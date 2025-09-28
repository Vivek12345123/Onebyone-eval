#!/usr/bin/env python3
"""
ragtruth_eval.py - Fixed for Self-RAG text generation evaluation

- Uses Self-RAG HF repo id ("selfrag/selfrag_llama2_7b") for all tokenizer/model loads.
- Supports evaluating an existing predictions JSONL OR generating predictions
- For text generation mode: compares generated text with reference response
- Skips span-level metrics when doing text generation (not hallucination detection)
"""

import argparse
import json
import os
import sys
from ast import literal_eval
import re
import string
import collections
from typing import Any, Dict, List, Tuple

from bert_score import score as bert_score

# ----------------------------------------
# Text comparison metrics
# ----------------------------------------
def normalize_answer(s: str) -> str:
    """Standard normalization for F1/EM calculation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def token_similarity(a: str, b: str) -> float:
    """Return similarity 0–1 between two tokens."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def normalize_balanced(s: str) -> str:
    """Balanced normalization with minimal suffix stripping."""
    s = normalize_answer(s)
    tokens = []
    for t in s.split():
        if t.endswith("ing") and len(t) > 4:
            t = t[:-3]
        elif t.endswith("ed") and len(t) > 3:
            t = t[:-2]
        elif t.endswith("ly") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
            t = t[:-1]
        tokens.append(t)
    return " ".join(tokens)

def numeric_overlap(pred: str, gold: str) -> bool:
    """Returns True if numeric tokens overlap."""
    nums_pred = set(re.findall(r"\d+", pred))
    nums_gold = set(re.findall(r"\d+", gold))
    return len(nums_pred & nums_gold) > 0

def balanced_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """Balanced F1 with soft token similarity."""
    pred_tokens = normalize_balanced(pred).split()
    gold_tokens = normalize_balanced(gold).split()
    
    if not gold_tokens:
        return (1.0, 1.0, 1.0) if not pred_tokens else (0.0, 0.0, 0.0)
    if not pred_tokens:
        return (0.0, 0.0, 0.0)
    
    # Check numeric overlap
    if numeric_overlap(pred, gold):
        return (1.0, 1.0, 1.0)
    
    # Soft recall: for each gold token, find best match in pred
    total_rec = 0.0
    for gt in gold_tokens:
        best = max((token_similarity(pt, gt) for pt in pred_tokens), default=0.0)
        total_rec += best
    rec = total_rec / len(gold_tokens) if gold_tokens else 0.0
    
    # Soft precision: for each pred token, find best match in gold
    total_prec = 0.0
    for pt in pred_tokens:
        best = max((token_similarity(gt, pt) for gt in gold_tokens), default=0.0)
        total_prec += best
    prec = total_prec / len(pred_tokens) if pred_tokens else 0.0
    
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec

def balanced_em(pred: str, gold: str) -> int:
    """Balanced EM with numeric overlap consideration."""
    pred_norm = normalize_balanced(pred)
    gold_norm = normalize_balanced(gold)
    if numeric_overlap(pred, gold):
        return 1
    return int(pred_norm == gold_norm)

# ----------------------------------------
# Helpers: load dataset or predictions file
# ----------------------------------------
def load_predictions_from_jsonl(filepath: str) -> List[Dict[str,Any]]:
    """Load a JSONL predictions file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    return data

# ----------------------------------------
# Optional generation from Self-RAG
# ----------------------------------------
SELF_RAG_REPO = "selfrag/selfrag_llama2_7b"

def _prepare_tokenizer(model_source: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None:
        try:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        except Exception:
            pass
    return tokenizer

def generate_predictions_on_dataset(dataset, batch_size:int, max_tokens:int, model_source: str = SELF_RAG_REPO):
    """Generate predictions using Self-RAG model."""
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[generate] Loading tokenizer+model from {model_source} on device {device} ...")
    tokenizer = _prepare_tokenizer(model_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    try:
        model.to(device)
    except Exception:
        pass
    model.eval()

    out_list = []
    with torch.no_grad():
        total = len(dataset)
        batch_prompts = []
        batch_idxs = []
        
        for i, raw_ex in enumerate(dataset):
            # Get the prompt/question from the dataset
            if isinstance(raw_ex, dict):
                ex = dict(raw_ex)
            elif hasattr(raw_ex, "to_dict"):
                try:
                    ex = dict(raw_ex.to_dict())
                except Exception:
                    ex = {"prompt": str(raw_ex)}
            else:
                ex = {"prompt": str(raw_ex)}
            
            # Extract prompt text
            prompt_text = ex.get("prompt") or ex.get("question") or ex.get("input") or ""
            
            # Create Self-RAG style prompt
            prompt = f"### Instruction: {prompt_text}\n\n### Response:"
            
            batch_prompts.append(prompt)
            batch_idxs.append(i)
            
            if len(batch_prompts) >= batch_size or (i + 1) == total:
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(max_tokens, 256),
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                
                for dec, prompt, idx in zip(decoded, batch_prompts, batch_idxs):
                    # Extract the generated response
                    ans = ""
                    if "### Response:" in dec:
                        ans = dec.split("### Response:", 1)[-1].strip()
                    elif prompt in dec:
                        ans = dec.replace(prompt, "").strip()
                    else:
                        ans = dec.strip()
                    
                    # Get original example
                    orig = dataset[idx]
                    if isinstance(orig, dict):
                        exrec = dict(orig)
                    elif hasattr(orig, "to_dict"):
                        try:
                            exrec = dict(orig.to_dict())
                        except Exception:
                            exrec = {}
                    else:
                        exrec = {}
                    
                    entry = {
                        "response": exrec.get("response", ""),  # Reference response
                        "labels": exrec.get("labels", []),      # Hallucination labels (not used for text gen)
                        "task_type": exrec.get("task_type", exrec.get("task", "")),
                        "output": ans,  # Generated text
                        "prompt": prompt_text  # Original prompt
                    }
                    out_list.append(entry)
                
                batch_prompts = []
                batch_idxs = []
                
                if (i+1) % 100 == 0:
                    print(f"[generate] Generated for {i+1}/{total}")
    
    return out_list

# ----------------------------------------
# Main evaluation
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default=None,
                        help='Path to predictions JSONL file.')
    parser.add_argument('--generate', action='store_true',
                        help='Generate predictions from Self-RAG on the HF test split.')
    parser.add_argument('--model_source', type=str, default=SELF_RAG_REPO,
                        help='HF repo id or local path for Self-RAG model/tokenizer.')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--bert_model', type=str, default='roberta-large')
    args = parser.parse_args()

    # Load or generate data
    data: List[Dict[str,Any]] = []
    
    if args.generate:
        from datasets import load_dataset
        ds = load_dataset("leobianco/ragtruth_final_test_set", split="test")
        total = len(ds)
        use_n = min(total, args.max_samples) if args.max_samples and args.max_samples > 0 else total
        if use_n < total:
            ds = ds.select(range(use_n))
        print(f"[main] Generating predictions with Self-RAG for {len(ds)} examples...")
        data = generate_predictions_on_dataset(ds, batch_size=args.batch_size, max_tokens=args.max_tokens, model_source=args.model_source)
        
        # Save predictions
        out_file = args.filepath if args.filepath else "ragtruth_preds_selfrag.jsonl"
        with open(out_file, "w", encoding="utf-8") as fout:
            for rec in data:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[main] Wrote generated predictions to {out_file}")
    else:
        if not args.filepath:
            print("Either pass --filepath <preds.jsonl> or --generate to create predictions.")
            sys.exit(1)
        if not os.path.exists(args.filepath):
            print(f"Predictions file not found: {args.filepath}")
            sys.exit(1)
        print(f"[main] Loading predictions from {args.filepath}...")
        data = load_predictions_from_jsonl(args.filepath)

    # Collect texts for evaluation
    all_pred_texts = []
    all_gold_texts = []
    
    # Traditional metrics
    total_f1 = 0.0
    total_em = 0.0
    
    # Balanced metrics
    balanced_f1_sum = 0.0
    balanced_prec_sum = 0.0
    balanced_rec_sum = 0.0
    balanced_em_sum = 0.0
    
    count = 0
    
    print(f"\n[main] Evaluating {len(data)} predictions...")
    
    for i, sample in enumerate(data):
        # For text generation: compare generated output with reference response
        pred_text = str(sample.get('output', "")).strip()
        gold_text = str(sample.get('response', "")).strip()
        
        if not pred_text:
            pred_text = ""
        if not gold_text:
            gold_text = ""
        
        all_pred_texts.append(pred_text)
        all_gold_texts.append(gold_text)
        
        # Traditional F1 and EM
        if pred_text and gold_text:
            f1 = f1_score(pred_text, gold_text)
            em = exact_match_score(pred_text, gold_text)
            total_f1 += f1
            total_em += em
            
            # Balanced metrics
            b_f1, b_prec, b_rec = balanced_f1(pred_text, gold_text)
            b_em = balanced_em(pred_text, gold_text)
            
            balanced_f1_sum += b_f1
            balanced_prec_sum += b_prec
            balanced_rec_sum += b_rec
            balanced_em_sum += b_em
        
        count += 1
        
        # Debug first few examples
        if i < 3:
            print(f"\nExample {i}:")
            print(f"  Pred: {pred_text[:100]}...")
            print(f"  Gold: {gold_text[:100]}...")
            print(f"  F1: {f1:.3f}, EM: {em:.3f}")
    
    # Calculate averages
    avg_f1 = total_f1 / count if count > 0 else 0.0
    avg_em = total_em / count if count > 0 else 0.0
    balanced_f1_avg = balanced_f1_sum / count if count > 0 else 0.0
    balanced_prec_avg = balanced_prec_sum / count if count > 0 else 0.0
    balanced_rec_avg = balanced_rec_sum / count if count > 0 else 0.0
    balanced_em_avg = balanced_em_sum / count if count > 0 else 0.0
    
    # Compute BERTScore
    print("\nComputing BERTScore (this may take a while)...")
    try:
        # Filter out empty pairs for BERTScore
        valid_pairs = [(p, g) for p, g in zip(all_pred_texts, all_gold_texts) if p and g]
        if valid_pairs:
            valid_preds = [p for p, g in valid_pairs]
            valid_golds = [g for p, g in valid_pairs]
            
            P, R, F1 = bert_score(
                valid_preds, 
                valid_golds, 
                model_type=args.bert_model, 
                lang='en', 
                rescale_with_baseline=False, 
                verbose=False
            )
            bert_p = float(P.mean())
            bert_r = float(R.mean())
            bert_f1 = float(F1.mean())
        else:
            bert_p = bert_r = bert_f1 = 0.0
    except Exception as e:
        print(f"[warning] BERTScore failed: {e}")
        bert_p = bert_r = bert_f1 = 0.0

    # Summarize results
    results = {
        "num_samples": count,
        "traditional_f1": avg_f1,
        "traditional_em": avg_em,
        "balanced_f1": balanced_f1_avg,
        "balanced_precision": balanced_prec_avg,
        "balanced_recall": balanced_rec_avg,
        "balanced_em": balanced_em_avg,
        "bertscore_precision": bert_p,
        "bertscore_recall": bert_r,
        "bertscore_f1": bert_f1,
        "bertscore_model": args.bert_model,
    }

    print("\n" + "="*60)
    print("SELF-RAG TEXT GENERATION EVALUATION RESULTS")
    print("="*60)
    print(f"Num samples: {results['num_samples']}")
    print(f"Traditional F1: {results['traditional_f1']:.4f}")
    print(f"Traditional EM: {results['traditional_em']:.4f}")
    print(f"Balanced F1: {results['balanced_f1']:.4f}")
    print(f"Balanced Precision: {results['balanced_precision']:.4f}")
    print(f"Balanced Recall: {results['balanced_recall']:.4f}")
    print(f"Balanced EM: {results['balanced_em']:.4f}")
    print(f"BERTScore Precision: {results['bertscore_precision']:.4f}")
    print(f"BERTScore Recall: {results['bertscore_recall']:.4f}")
    print(f"BERTScore F1: {results['bertscore_f1']:.4f}")
    print("="*60)

    # Write summary JSON
    summary_file = args.filepath.replace(".jsonl", "_summary.json") if args.filepath and args.filepath.endswith(".jsonl") else "summary.json"
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(results, sf, indent=2)
    print(f"\nSummary written to {summary_file}")

if __name__ == '__main__':
    main()
