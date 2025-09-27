#!/usr/bin/env python3
"""
run_selfrag_hotpot_eval.py

Single file that:
 - Implements the exact HotpotQA forgiving evaluator (kept verbatim).
 - Downloads HotpotQA (distractor / validation).
 - Loads selfrag/selfrag_llama2_7b from Hugging Face.
 - Generates predictions (configurable: num samples, batch size, max tokens).
 - Saves predictions JSONL in the format expected by the evaluator.
 - Runs the evaluator on those predictions.

Defaults:
  --num-samples 2000
  --batch-size 4
  --max-new-tokens 512

Requirements:
  pip install transformers datasets bert-score tqdm accelerate
  huggingface-cli login  # if model/dataset requires auth
"""

# ------------------ Begin: EXACT evaluator code you provided ------------------
import re
import string
import json
from collections import Counter
from difflib import SequenceMatcher
# Note: we will import evaluate_hotpot_official.f1_score, normalize_answer later
# in the evaluate_predictions function to avoid top-level import issues.

# -------------------------------
# Helper: strict normalization
# -------------------------------
def normalize_strict(s):
    return normalize_answer(s)
# -------------------------------
# Balanced forgiving normalization
# -------------------------------
def normalize_balanced(s):
    s = normalize_answer(s)
    tokens = []
    for t in s.split():
        if t.endswith("ing"):
            t = t[:-3]
        elif t.endswith("ed"):
            t = t[:-2]
        elif t.endswith("ly"):
            t = t[:-2]
        elif t.endswith("s") and len(t) > 3:
            t = t[:-1]
        elif t.endswith("n") and len(t) > 4:
            t = t[:-1]
        if t in ["armenian", "armenia"]:
            t = "armenia"
        tokens.append(t)
    return " ".join(tokens)
# -------------------------------
# Numeric overlap
# -------------------------------
def numeric_overlap(pred, gold):
    pred_nums = set(re.findall(r"\d+", pred))
    gold_nums = set(re.findall(r"\d+", gold))
    return len(pred_nums & gold_nums) > 0
# -------------------------------
# Balanced F1 with soft token similarity
# -------------------------------
def token_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()
def balanced_f1(pred, gold):
    pred_tokens = normalize_balanced(pred).split()
    gold_tokens = normalize_balanced(gold).split()
    if numeric_overlap(pred, gold):
        return 1.0, 1.0, 1.0
    total_rec = 0.0
    for gt in gold_tokens:
        best = max((token_similarity(pt, gt) for pt in pred_tokens), default=0.0)
        total_rec += best
    rec = total_rec / len(gold_tokens) if gold_tokens else 0.0
    total_prec = 0.0
    for pt in pred_tokens:
        best = max((token_similarity(gt, pt) for gt in gold_tokens), default=0.0)
        total_prec += best
    prec = total_prec / len(pred_tokens) if pred_tokens else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec
def balanced_em(pred, gold):
    pred_norm = normalize_balanced(pred)
    gold_norm = normalize_balanced(gold)
    if numeric_overlap(pred_norm, gold_norm):
        return 1
    return int(pred_norm == gold_norm)
# -------------------------------
# BERTScore (semantic similarity) - lazy import to avoid heavy deps during balanced eval
# -------------------------------
def semantic_scores(preds, golds, model_type="roberta-large-mnli"):
    from bert_score import score as bertscore_score
    P, R, F1 = bertscore_score(
        preds,
        golds,
        model_type=model_type,
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
    )
    return {
        "precision": [p.item() for p in P],
        "recall": [r.item() for r in R],
        "f1": [f.item() for f in F1],
    }

# ------------------ End: EXACT evaluator code you provided ------------------


# We'll provide a function that runs that exact evaluator logic on a given predictions_file.
# The internal calculations and functions above are not changed; we simply drive them here.
def evaluate_predictions(predictions_file):
    """
    Runs the exact evaluation logic you provided on `predictions_file`.
    This function keeps the evaluation calculations identical to your original file.
    """
    # Import required names from evaluate_hotpot_official as used in original script
    # This matches the top-level import in your original script: from evaluate_hotpot_official import f1_score, normalize_answer
    global f1_score, normalize_answer
    try:
        from evaluate_hotpot_official import f1_score, normalize_answer  # noqa: F401
    except Exception as e:
        raise ImportError(
            "Failed to import evaluate_hotpot_official. "
            "Make sure evaluate_hotpot_official.py is available in PYTHONPATH. "
            f"Original import error: {e}"
        )

    print(f"Loading predictions from {predictions_file}")
    results = []
    with open(predictions_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    print(f"Loaded {len(results)} predictions")
    # Extract predictions and gold answers
    preds = []
    golds = []
    for result in results:
        prediction = result.get('prediction', '')
        gold_list = result.get('gold', [])
        # Handle gold as list (take first element) or string
        if isinstance(gold_list, list) and len(gold_list) > 0:
            gold = gold_list[0]
        else:
            gold = str(gold_list) if gold_list else ''
        preds.append(prediction)
        golds.append(gold)
    # Calculate metrics
    total_strict_em = 0
    total_strict_f1 = 0
    total_balanced_em = 0
    total_balanced_f1 = 0
    total_balanced_precision = 0
    total_balanced_recall = 0
    print("\nEvaluating predictions...")
    for i, (pred, gold) in enumerate(zip(preds, golds)):
        if i % 50 == 0:
            print(f"Processing {i+1}/{len(preds)}")
        # Strict
        strict_f1, _, _ = f1_score(pred, gold)
        strict_em = int(normalize_strict(pred) == normalize_strict(gold))
        # Balanced
        bal_f1, bal_prec, bal_rec = balanced_f1(pred, gold)
        bal_em = balanced_em(pred, gold)
        total_strict_em += strict_em
        total_strict_f1 += strict_f1
        total_balanced_em += bal_em
        total_balanced_f1 += bal_f1
        total_balanced_precision += bal_prec
        total_balanced_recall += bal_rec
    # Calculate averages
    num_samples = len(preds)
    avg_strict_em = total_strict_em / num_samples
    avg_strict_f1 = total_strict_f1 / num_samples
    avg_balanced_em = total_balanced_em / num_samples
    avg_balanced_f1 = total_balanced_f1 / num_samples
    avg_balanced_precision = total_balanced_precision / num_samples
    avg_balanced_recall = total_balanced_recall / num_samples
    print("\n" + "="*60)
    print("HOTPOTQA FORGIVING EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {num_samples}")
    print(f"Strict EM: {avg_strict_em:.4f}")
    print(f"Strict F1: {avg_strict_f1:.4f}")
    print(f"Balanced EM: {avg_balanced_em:.4f}")
    print(f"Balanced F1: {avg_balanced_f1:.4f}")
    print(f"Balanced Precision: {avg_balanced_precision:.4f}")
    print(f"Balanced Recall: {avg_balanced_recall:.4f}")
    print("="*60)
    # Semantic (batch) - sample first 50 for speed
    print("\nComputing BERTScore (sampling first 50 for speed)...")
    sample_size = min(50, len(preds))
    sem_results = semantic_scores(preds[:sample_size], golds[:sample_size])
    print("Semantic BERTScore averages:")
    print(f"Precision: {sum(sem_results['precision'])/len(sem_results['precision']):.4f}")
    print(f"Recall:    {sum(sem_results['recall'])/len(sem_results['recall']):.4f}")
    print(f"F1:        {sum(sem_results['f1'])/len(sem_results['f1']):.4f}")
    # Save results
    output_file = predictions_file.replace('.jsonl', '_forgiving_evaluation.json')
    metrics = {
        'num_samples': num_samples,
        'strict_em': avg_strict_em,
        'strict_f1': avg_strict_f1,
        'balanced_em': avg_balanced_em,
        'balanced_f1': avg_balanced_f1,
        'balanced_precision': avg_balanced_precision,
        'balanced_recall': avg_balanced_recall,
        'bertscore_precision': sum(sem_results['precision'])/len(sem_results['precision']),
        'bertscore_recall': sum(sem_results['recall'])/len(sem_results['recall']),
        'bertscore_f1': sum(sem_results['f1'])/len(sem_results['f1'])
    }
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_file}")
    return metrics

# ------------------ Runner: downloads dataset, model, generates predictions ------------------
def build_prompt_hotpot(example):
    """
    Build a simple prompt for HotpotQA example.
    We'll include question and a joined context (sentences).
    """
    question = example.get("question", "")
    # HotpotQA 'context' structure from HF is usually a list of [title, [sentences...]] pairs.
    ctx = example.get("context", [])
    # Try to build a readable context string:
    ctx_str_parts = []
    # ctx items may be [title, sentences_list] or dicts; handle common patterns.
    try:
        for item in ctx:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                sents = item[1]
                if isinstance(sents, (list, tuple)):
                    ctx_str_parts.append(" ".join(sents))
                else:
                    ctx_str_parts.append(str(sents))
            elif isinstance(item, dict):
                # sometimes context entries are dict-like
                if "title" in item and "sentences" in item:
                    ctx_str_parts.append(" ".join(item.get("sentences", [])))
                else:
                    # fallback: convert to string
                    ctx_str_parts.append(str(item))
            else:
                ctx_str_parts.append(str(item))
    except Exception:
        # Safe fallback: stringify context
        ctx_str_parts = [str(ctx)]
    context_text = "\n".join(ctx_str_parts)
    prompt = f"Context: {context_text}\n\nQuestion: {question}\nAnswer:"
    return prompt

def generate_and_save_predictions(model_name, num_samples, batch_size, max_new_tokens, out_file, device="auto"):
    """
    Downloads HotpotQA (distractor / validation), runs model generation, and writes predictions JSONL.
    Saved JSONL line format (matching evaluator expectation): {"prediction": "...", "gold": ["..."]}
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import math
    import torch

    print("Loading HotpotQA (distractor, validation) from Hugging Face...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    total_available = len(ds)
    if num_samples is None:
        num_samples = total_available
    if num_samples > total_available:
        print(f"Requested num_samples={num_samples} > available={total_available}, truncating to {total_available}")
        num_samples = total_available
    ds = ds.select(range(num_samples))
    print(f"Selected {len(ds)} examples (out of {total_available})")

    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # Use device_map='auto' if available to place weights on GPU if present
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    except Exception:
        # fallback to CPU if device_map not supported / fails
        model = AutoModelForCausalLM.from_pretrained(model_name)
    # Determine pipeline device
    if device == "auto":
        device_arg = 0 if torch.cuda.is_available() else -1
    elif device == "cuda":
        device_arg = 0
    else:
        device_arg = -1

    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device_arg)

    print("Generating answers...")
    results = []
    for start in range(0, len(ds), batch_size):
        batch = ds[start: start + batch_size]
        prompts = [build_prompt_hotpot(ex) for ex in batch]
        try:
            outputs = gen_pipe(prompts, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False, batch_size=len(prompts))
        except TypeError:
            # some transformers versions may not accept return_full_text param
            outputs = gen_pipe(prompts, max_new_tokens=max_new_tokens, do_sample=False, batch_size=len(prompts))
        # Normalize outputs into generated_text strings
        normalized_texts = []
        for o in outputs:
            if isinstance(o, dict) and 'generated_text' in o:
                normalized_texts.append(o['generated_text'])
            elif isinstance(o, list) and len(o) > 0 and isinstance(o[0], dict) and 'generated_text' in o[0]:
                normalized_texts.append(o[0]['generated_text'])
            elif isinstance(o, str):
                normalized_texts.append(o)
            else:
                normalized_texts.append(str(o))
        # Save per example
        for ex, gen_text in zip(batch, normalized_texts):
            # Extract answer the same way as your SQuAD example (split on "Answer:")
            if "Answer:" in gen_text:
                pred = gen_text.split("Answer:")[-1].strip()
            else:
                # If pipeline returned only new tokens (no prompt), use entire string
                pred = gen_text.strip()
            # gold: put as list to match evaluator handling (it accepts list or string)
            gold_answer = ex.get("answer", "")
            # Some HF variants have answer as list; normalize to first string
            if isinstance(gold_answer, (list, tuple)) and len(gold_answer) > 0:
                gold_entry = [str(gold_answer[0])]
            else:
                gold_entry = [str(gold_answer)]
            results.append({"prediction": pred, "gold": gold_entry})
    # Write JSONL
    with open(out_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(results)} predictions to {out_file}")
    return out_file

# ------------------ CLI entrypoint ------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Run selfrag on HotpotQA and evaluate with forgiving Hotpot evaluator (integrated).")
    p.add_argument("--model", type=str, default="selfrag/selfrag_llama2_7b", help="Hugging Face model name (default selfrag/selfrag_llama2_7b)")
    p.add_argument("--num-samples", type=int, default=2000, help="Total number of validation samples to evaluate (default 2000).")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size for generation (default 4).")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate per sample (default 512).")
    p.add_argument("--out-file", type=str, default="llama_7b_hotpot_200_predictions.jsonl", help="Output predictions JSONL file.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    p.add_argument("--only-eval", action="store_true", help="If set, only run evaluation on --out-file (skip model/dataset downloads/generation).")
    return p.parse_args()

def main():
    args = parse_args()
    if args.only_eval:
        # Only run evaluation on provided predictions file
        evaluate_predictions(args.out_file)
        return

    # Generate predictions and save
    pred_file = generate_and_save_predictions(
        model_name=args.model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        out_file=args.out_file,
        device=args.device
    )

    # Run the exact evaluation on the produced predictions file
    evaluate_predictions(pred_file)

if __name__ == "__main__":
    main()
