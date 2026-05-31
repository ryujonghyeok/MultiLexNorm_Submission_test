#!/usr/bin/env python3
"""Generate a Codabench submission using a trained LoRA adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from train_lora import (
    as_list,
    load_multilexnorm,
    model_name_for_path,
    parse_json_list,
    render_prompt,
)
from utils import counting, evaluate, mfr


DEFAULT_LANG_BEST_MFR_LANGS = "id,iden,sr"



def check_torchao_version() -> None:
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return

    try:
        from packaging.version import Version
    except ImportError:
        return

    if Version(version) < Version("0.16.0"):
        raise RuntimeError(
            f"Found torchao=={version}, but PEFT requires torchao>=0.16.0 in this environment. "
            "In Colab, run: pip uninstall -y torchao, then rerun prediction. "
            "Do not upgrade torch just to satisfy torchao."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Codabench predictions with Gemma + LoRA adapter.")
    parser.add_argument("--adapter-dir", required=True, help="Path to the trained LoRA adapter folder.")
    parser.add_argument("--model-id", default=None, help="Base model ID. Defaults to adapter_config.json value.")
    parser.add_argument("--dataset-id", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Parent directory for timestamped prediction output folders.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Prefix for the timestamped output directory and ZIP file. Defaults to the adapter folder name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory. Prefer --output-root and --run-prefix for KST timestamped runs.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token for gated models. Defaults to HF_TOKEN.",
    )
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Rows per generation batch. Use 0 to choose automatically from available GPU memory.",
    )
    parser.add_argument("--max-examples", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--fallback", choices=["raw", "mfr"], default="mfr")
    parser.add_argument(
        "--prediction-strategy",
        choices=["model", "mfr", "mfr-known", "mfr-confidence", "lang-best"],
        default="mfr-known",
        help=(
            "How to combine model predictions with MFR. "
            "'model' uses valid model JSON directly. "
            "'mfr' ignores the model output. "
            "'mfr-known' uses MFR for tokens seen in public train/validation and model only for unseen tokens. "
            "'mfr-confidence' uses model only when MFR confidence is below --mfr-confidence-threshold. "
            "'lang-best' uses validation-derived language routing between model and mfr-known."
        ),
    )
    parser.add_argument(
        "--lang-best-mfr-langs",
        default=DEFAULT_LANG_BEST_MFR_LANGS,
        help=(
            "Comma-separated language codes that should use mfr-known under --prediction-strategy lang-best. "
            "All other languages use the model output. Use 'none' to force model-only."
        ),
    )
    parser.add_argument(
        "--force-model",
        nargs="*",
        default=[],
        metavar="LANG",
        help=(
            "Language codes that must use the model output even when the strategy would use MFR. "
            "Examples: --force-model ko ja or --force-model 'ko,ja'."
        ),
    )
    parser.add_argument(
        "--mfr-confidence-threshold",
        type=float,
        default=0.75,
        help="For --prediction-strategy mfr-confidence, use MFR when top replacement frequency is at least this value.",
    )
    parser.add_argument("--preview-examples", type=int, default=5)
    parser.add_argument(
        "--progress-steps",
        type=int,
        default=100,
        help="Print progress every N completed rows. Use 0 to disable progress logs.",
    )
    parser.add_argument(
        "--no-sort-by-length",
        action="store_true",
        help="Disable length-sorted batching. Sorting improves GPU throughput while preserving output order.",
    )
    parser.add_argument(
        "--disable-stop-after-json",
        action="store_true",
        help="Keep generating until max_new_tokens instead of stopping after a valid JSON token list.",
    )
    args = parser.parse_args()
    if args.batch_size < 0:
        parser.error("--batch-size must be 0 or greater")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than 0")
    if not 0.0 <= args.mfr_confidence_threshold <= 1.0:
        parser.error("--mfr-confidence-threshold must be between 0 and 1")
    return args


def resolve_output_dir(
    output_dir: str | None,
    output_root: str,
    run_prefix: str | None,
    adapter_dir: Path,
) -> tuple[Path, str]:
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m%d-%H%M")
    if output_dir:
        path = Path(output_dir)
        if not re.search(r"(\d{4}-\d{4}|\d{6}_\d{4})", path.name):
            raise ValueError(
                "Explicit --output-dir must include an MMDD-HHMM timestamp in the final folder name. "
                "Prefer --output-root plus --run-prefix so the script creates a KST timestamp automatically."
            )
        return path, timestamp
    prefix = run_prefix if run_prefix is not None else model_name_for_path(adapter_dir.name)
    return Path(output_root) / f"{prefix}_{timestamp}", timestamp


def model_id_from_adapter(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_id = config.get("base_model_name_or_path")
    if not model_id:
        raise ValueError(f"adapter_config.json does not contain base_model_name_or_path: {config_path}")
    return model_id


def make_mfr_counts(data: Any, target_split: str) -> dict[str, dict[str, dict[str, int]]]:
    from datasets import concatenate_datasets

    if target_split == "test":
        train = concatenate_datasets([data["train"], data["validation"]])
    else:
        train = data["train"]
    train_df = train.to_pandas()
    counts_by_lang = {}
    for lang, lang_df in train_df.groupby("lang"):
        counts_by_lang[lang] = counting(lang_df.to_dict(orient="records"))
    return counts_by_lang


def fallback_prediction(raw: list[str], lang: str, fallback: str, counts_by_lang: dict[str, Any] | None) -> list[str]:
    if fallback == "mfr" and counts_by_lang is not None:
        return mfr(raw, counts_by_lang.get(lang, {}))
    return list(raw)


def mfr_token_prediction(raw_token: str, token_counts: dict[str, dict[str, int]]) -> str:
    if raw_token not in token_counts:
        return raw_token
    return max(token_counts[raw_token], key=token_counts[raw_token].get)


def mfr_token_confidence(raw_token: str, token_counts: dict[str, dict[str, int]]) -> float:
    if raw_token not in token_counts:
        return 0.0
    replacements = token_counts[raw_token]
    total = sum(replacements.values())
    if total <= 0:
        return 0.0
    return max(replacements.values()) / total


def prediction_strategy_slug(prediction_strategy: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", prediction_strategy).replace("-", "_").strip("_")


def parse_lang_codes(value: str) -> set[str]:
    value = value.strip()
    if value.lower() in {"", "none"}:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_lang_code_args(values: list[str]) -> set[str]:
    return parse_lang_codes(",".join(values))


def parse_model_prediction(generated_text: str, raw: list[str]) -> list[str] | None:
    parsed = parse_json_list(generated_text)
    if isinstance(parsed, list) and len(parsed) == len(raw) and all(isinstance(token, str) for token in parsed):
        return parsed
    return None


def choose_prediction(
    model_pred: list[str] | None,
    raw: list[str],
    lang: str,
    fallback: str,
    counts_by_lang: dict[str, Any] | None,
    strategy: str,
    mfr_confidence_threshold: float,
    lang_best_mfr_langs: set[str] | None,
    force_model_langs: set[str],
) -> tuple[list[str], str]:
    fallback_pred = fallback_prediction(raw, lang, fallback, counts_by_lang)
    if strategy == "mfr":
        return fallback_pred, "mfr"
    if model_pred is None:
        return fallback_pred, "fallback"
    if lang in force_model_langs:
        return model_pred, "model"
    if strategy == "model" or counts_by_lang is None:
        return model_pred, "model"

    if strategy == "lang-best":
        if lang_best_mfr_langs is not None and lang in lang_best_mfr_langs:
            strategy = "mfr-known"
        else:
            return model_pred, "model"

    token_counts = counts_by_lang.get(lang, {})
    pred = []
    model_used = 0
    for raw_token, model_token in zip(raw, model_pred):
        if strategy == "mfr-known":
            use_mfr = raw_token in token_counts
        elif strategy == "mfr-confidence":
            use_mfr = mfr_token_confidence(raw_token, token_counts) >= mfr_confidence_threshold
        else:
            use_mfr = False

        if use_mfr:
            pred.append(mfr_token_prediction(raw_token, token_counts))
        else:
            pred.append(model_token)
            model_used += 1

    if model_used == 0:
        return pred, "mfr"
    if model_used == len(raw):
        return pred, "model"
    return pred, "hybrid"


def has_public_gold(records: list[dict[str, Any]]) -> bool:
    return any(any(token != "" for token in record["norm"]) for record in records)


def score_records(records: list[dict[str, Any]]) -> tuple[int, int, int, float, float, float | None]:
    correct = 0
    changed = 0
    total = 0
    for record in records:
        for raw_token, gold_token, pred_token in zip(record["raw"], record["norm"], record["pred"]):
            if raw_token != gold_token:
                changed += 1
            if gold_token == pred_token:
                correct += 1
            total += 1

    if total == 0:
        return 0, 0, 0, 0.0, 0.0, None

    lai = (total - changed) / total
    accuracy = correct / total
    err = None if changed == 0 else (accuracy - lai) / (1 - lai)
    return total, changed, correct, lai, accuracy, err


def print_score_by_language(records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["lang"], []).append(record)

    rows = []
    for lang, lang_records in grouped.items():
        total, changed, _correct, lai, accuracy, err = score_records(lang_records)
        rows.append(
            {
                "lang": lang,
                "sentences": len(lang_records),
                "tokens": total,
                "changed": changed,
                "lai": lai,
                "accuracy": accuracy,
                "err": err,
            }
        )

    rows.sort(key=lambda row: -1.0 if row["err"] is None else row["err"], reverse=True)

    print("\nValidation score by language:")
    print(f"{'lang':<8} {'sent':>6} {'tokens':>8} {'changed':>8} {'LAI':>8} {'Accuracy':>9} {'ERR':>8}")
    print("-" * 65)
    for row in rows:
        err_text = "n/a" if row["err"] is None else f"{row['err'] * 100:6.2f}"
        print(
            f"{row['lang']:<8} {row['sentences']:>6} {row['tokens']:>8} {row['changed']:>8} "
            f"{row['lai'] * 100:>8.2f} {row['accuracy'] * 100:>9.2f} {err_text:>8}"
        )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def choose_batch_size(requested_batch_size: int, max_new_tokens: int) -> tuple[int, str]:
    if requested_batch_size > 0:
        return requested_batch_size, "manual"

    try:
        import torch
    except ImportError:
        return 1, "auto: torch unavailable"

    if not torch.cuda.is_available():
        return 1, "auto: CUDA unavailable"

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)

    if free_gb >= 70:
        batch_size = 128
    elif free_gb >= 35:
        batch_size = 64
    elif free_gb >= 20:
        batch_size = 32
    elif free_gb >= 12:
        batch_size = 16
    else:
        batch_size = 8

    if max_new_tokens > 128:
        batch_size = max(1, batch_size // 2)

    reason = f"auto: {device_name}, free {free_gb:.1f}/{total_gb:.1f} GB GPU RAM"
    return batch_size, reason


def clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_out_of_memory(error: Exception) -> bool:
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def is_gemma4_model_id(model_id: str) -> bool:
    normalized = model_id.lower().replace("_", "-")
    return "gemma-4" in normalized or "gemma4" in normalized


def load_model_and_tokenizer(args: argparse.Namespace, model_id: str) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

    if is_gemma4_model_id(model_id):
        processor = AutoProcessor.from_pretrained(model_id, token=args.hf_token)
        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "chat_template", None) is None and getattr(processor, "chat_template", None):
            tokenizer.chat_template = processor.chat_template
        model_loader = AutoModelForImageTextToText
        print("Model loader: AutoModelForImageTextToText for Gemma 4 text-only inference")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.hf_token)
        model_loader = AutoModelForCausalLM
        print("Model loader: AutoModelForCausalLM")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    base_model = model_loader.from_pretrained(
        model_id,
        device_map="auto",
        dtype=compute_dtype,
        quantization_config=quantization_config,
        token=args.hf_token,
    )
    check_torchao_version()
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    return model, tokenizer


def generation_config_for_inference(model: Any, tokenizer: Any, max_new_tokens: int) -> Any:
    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.max_new_tokens = max_new_tokens
    return generation_config


def make_stop_after_json(tokenizer: Any, prompt_len: int, expected_lens: list[int]) -> Any:
    from transformers import StoppingCriteria

    class StopAfterValidJsonList(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            for row_idx, expected_len in enumerate(expected_lens):
                generated_ids = input_ids[row_idx][prompt_len:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                parsed = parse_json_list(text)
                valid = (
                    isinstance(parsed, list)
                    and len(parsed) == expected_len
                    and all(isinstance(token, str) for token in parsed)
                )
                if not valid:
                    return False
            return True

    return StopAfterValidJsonList()


def generate_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    stop_after_json: bool,
) -> list[str]:
    import torch
    from transformers import StoppingCriteriaList

    prompts = [render_prompt(tokenizer, row) for row in rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    stopping_criteria = None
    if stop_after_json:
        expected_lens = [len(as_list(row["raw"])) for row in rows]
        stopping_criteria = StoppingCriteriaList([make_stop_after_json(tokenizer, prompt_len, expected_lens)])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config_for_inference(model, tokenizer, max_new_tokens),
        )

    return [
        tokenizer.decode(output_ids[row_idx][prompt_len:], skip_special_tokens=True).strip()
        for row_idx in range(len(rows))
    ]


def write_submission(records: list[dict[str, Any]], output_dir: Path, prediction_strategy: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    strategy = prediction_strategy_slug(prediction_strategy)
    zip_path = output_dir.parent / f"{output_dir.name}({strategy}).zip"

    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, arcname="predictions.json")

    return predictions_path, zip_path


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    model_id = args.model_id or model_id_from_adapter(adapter_dir)
    output_dir, run_timestamp_kst = resolve_output_dir(args.output_dir, args.output_root, args.run_prefix, adapter_dir)

    print("Adapter directory:", adapter_dir)
    print("Base model:", model_id)
    print("Dataset:", args.dataset_id)
    print("Split:", args.split)
    print("Run timestamp (KST):", run_timestamp_kst)
    print("Output directory:", output_dir)
    print("Prediction strategy:", args.prediction_strategy)
    if args.prediction_strategy == "mfr-confidence":
        print("MFR confidence threshold:", args.mfr_confidence_threshold)
    force_model_langs = parse_lang_code_args(args.force_model)
    if force_model_langs and args.prediction_strategy == "mfr":
        raise ValueError("--force-model cannot be used with --prediction-strategy mfr because the model is skipped.")
    if force_model_langs:
        print("Force-model languages:", ", ".join(sorted(force_model_langs)))
    lang_best_mfr_langs = None
    if args.prediction_strategy == "lang-best":
        lang_best_mfr_langs = parse_lang_codes(args.lang_best_mfr_langs)
        print("Lang-best mfr-known languages:", ", ".join(sorted(lang_best_mfr_langs)) or "(none)")

    data = load_multilexnorm(args.dataset_id)
    target = data[args.split]
    if args.max_examples and args.max_examples > 0:
        target = target.select(range(min(args.max_examples, len(target))))
    print("Prediction rows:", len(target))
    print("Length-sorted batching:", "disabled" if args.no_sort_by_length else "enabled")

    needs_mfr_counts = args.fallback == "mfr" or args.prediction_strategy in {
        "mfr",
        "mfr-known",
        "mfr-confidence",
        "lang-best",
    }
    counts_by_lang = make_mfr_counts(data, args.split) if needs_mfr_counts else None
    model = None
    tokenizer = None
    if args.prediction_strategy == "mfr":
        batch_size = args.batch_size if args.batch_size > 0 else max(1, len(target))
        batch_size_reason = "manual" if args.batch_size > 0 else "auto: model skipped for MFR-only prediction"
    else:
        model, tokenizer = load_model_and_tokenizer(args, model_id)
        batch_size, batch_size_reason = choose_batch_size(args.batch_size, args.max_new_tokens)
    print(f"Batch size: {batch_size} ({batch_size_reason})")

    indexed_rows = list(enumerate(target))
    if not args.no_sort_by_length:
        indexed_rows.sort(key=lambda item: len(as_list(item[1]["raw"])))

    records: list[dict[str, Any] | None] = [None] * len(target)
    source_counts = {"model": 0, "hybrid": 0, "mfr": 0, "fallback": 0}
    completed = 0
    started_at = time.monotonic()
    next_progress = args.progress_steps if args.progress_steps > 0 else None
    batch_start = 0
    while batch_start < len(indexed_rows):
        batch_end = min(batch_start + batch_size, len(indexed_rows))
        indexed_batch = indexed_rows[batch_start:batch_end]
        batch_rows = [row for _, row in indexed_batch]
        if args.prediction_strategy == "mfr":
            generated_texts = [""] * len(batch_rows)
        else:
            try:
                generated_texts = generate_batch(
                    model,
                    tokenizer,
                    batch_rows,
                    args.max_new_tokens,
                    stop_after_json=not args.disable_stop_after_json,
                )
            except RuntimeError as error:
                if is_cuda_out_of_memory(error) and batch_size > 1:
                    clear_cuda_cache()
                    old_batch_size = batch_size
                    batch_size = max(1, batch_size // 2)
                    print(
                        f"CUDA OOM at batch size {old_batch_size}; retrying from row {completed} "
                        f"with batch size {batch_size}",
                        flush=True,
                    )
                    continue
                raise

        for (idx, row), generated_text in zip(indexed_batch, generated_texts):
            raw = as_list(row["raw"])
            norm = as_list(row["norm"]) if "norm" in row else [""] * len(raw)
            if len(norm) != len(raw):
                norm = [""] * len(raw)

            model_pred = parse_model_prediction(generated_text, raw)
            pred, source = choose_prediction(
                model_pred,
                raw,
                row["lang"],
                args.fallback,
                counts_by_lang,
                args.prediction_strategy,
                args.mfr_confidence_threshold,
                lang_best_mfr_langs,
                force_model_langs,
            )
            source_counts[source] += 1

            records[idx] = {"raw": raw, "norm": norm, "lang": row["lang"], "pred": pred}

            if idx < args.preview_examples:
                print(f"\nExample {idx + 1}")
                print("lang:", row["lang"])
                print("raw: ", raw)
                print("model_text:", generated_text)
                print("pred:", pred)
                print("source:", source)

        completed += len(indexed_batch)
        batch_start = batch_end
        should_print_progress = completed == len(target)
        if next_progress is not None and completed >= next_progress:
            should_print_progress = True
            while next_progress is not None and completed >= next_progress:
                next_progress += args.progress_steps
        if should_print_progress:
            elapsed = time.monotonic() - started_at
            rows_per_second = completed / elapsed if elapsed > 0 else 0.0
            remaining = len(target) - completed
            eta = remaining / rows_per_second if rows_per_second > 0 else 0.0
            percent = (completed / len(target) * 100) if len(target) else 100.0
            print(
                "Progress: "
                f"{completed}/{len(target)} rows ({percent:.1f}%) | "
                f"{rows_per_second:.2f} rows/s | "
                f"elapsed {format_duration(elapsed)} | "
                f"ETA {format_duration(eta)} | "
                f"model {source_counts['model']} / hybrid {source_counts['hybrid']} / "
                f"mfr {source_counts['mfr']} / fallback {source_counts['fallback']}",
                flush=True,
            )

    final_records: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if record is None:
            raise ValueError(f"Missing prediction at row {idx}")
        if set(record) != {"raw", "norm", "lang", "pred"}:
            raise ValueError(f"Bad keys at row {idx}: {record.keys()}")
        if len(record["raw"]) != len(record["norm"]) or len(record["raw"]) != len(record["pred"]):
            raise ValueError(f"Length mismatch at row {idx}")
        final_records.append(record)

    predictions_path, zip_path = write_submission(final_records, output_dir, args.prediction_strategy)
    if has_public_gold(final_records):
        print("\nValidation score:")
        evaluate(
            raw=[record["raw"] for record in final_records],
            gold=[record["norm"] for record in final_records],
            pred=[record["pred"] for record in final_records],
        )
        print_score_by_language(final_records)
    print("\nPrediction source counts:", source_counts)
    print(f"Wrote {predictions_path}")
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
