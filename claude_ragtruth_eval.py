#!/usr/bin/env python3
"""
ragtruth_eval.py - Fixed for Self-RAG text generation vs hallucination detection

This script handles BOTH:
1. Hallucination detection evaluation (span-level metrics) - when output is a list of spans
2. Text generation evaluation (text comparison metrics) - when output is generated text

For Self-RAG (text generation), it will use text comparison metrics.
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
from difflib import SequenceMatcher

# ----------------------------------------
# Span-based metrics (for hallucination detection)
# ----------------------------------------
def compute_multi_spans(gold_spans: List[Tuple[int,int]], pred_spans: List[Tuple[int,int]]):
    """Compute overlap for multiple spans."""
    if not gold_spans:
        gold_sets = [set()]
    else:
        gold_sets = [set(range(int(start), int(end))) for start, end in gold_spans]
    if not pred_spans:
        pred_sets = [set()]
    else:
        pred_sets = [set(range(int(start), int(end))) for start, end in pred_spans]

    try:
        gold_union = set.union(*gold_sets) if gold_sets else set()
    except TypeError:
        gold_union = set()
    try:
        pred_union = set.union(*pred_sets) if pred_sets else set()
    except TypeError:
        pred_union = set()

    tp = len(gold_union.intersection(pred_union))
    pred_len = len(pred_union)
    gold_len = len(gold_union)
    return tp, pred_len, gold_len

def find_start_end_index(text: str, sub_str: str, strict: bool = False) -> Tuple[int,int]:
    """Find (start,end) indexes for sub_str inside text."""
    if text is None:
        return (-1, -1)
    s = text.lower()
    sub = sub_str.lower().strip()
    start = s.find(sub)
    if start == -1:
        if len(sub) >= 10 and not strict:
            match_length = 10
            prefix_match = s.find(sub[:match_length])
            suffix_match = s.find(sub[-match_length:])
            if prefix_match != -1 and suffix_match != -1:
                new_start = prefix_match
                new_end = suffix_match + match_length
                if 0.7 < (new_end - new_start) / max(1, len(sub)) < 1.3:
                    return (new_start, new_end)
            sub2 = ' '.join(sub.split())
            start = s.find(sub2)
            if start == -1:
                return (-1, -1)
    end = start + len(sub)
    return (start, end)

def cal_span_metrics(response: str, labels: List[Dict], output: str):
    """Calculate span-level metrics for hallucination detection."""
    try:
        output_list = literal_eval(output)
        if not isinstance(output_list, (list, tuple)):
            output_list = []
    except Exception:
        output_list = []

    predict_span = []
    for out in output_list:
        if not isinstance(out, str):
            out = str(out)
        tup = find_start_end_index(response or "", out, strict=False)
        if tup != (-1, -1):
            predict_span.append(tup)

    ground_truth_span = []
    if labels:
        for label in labels:
            try:
                ground_truth_span.append((int(label['start']), int(label['end'])))
            except Exception:
                continue

    if len(predict_span) == 0:
        predict_span.append((0,0))
    if len(ground_truth_span) == 0:
        ground_truth_span.append((0,0))

    match_size, prediction, groud_truth = compute_multi_spans(ground_truth_span, predict_span)

    # case level
    pos_true_cnt = neg_true_cnt = neg_false_cnt = pos_false_cnt = 0
    if output_list != [] and labels != []:
        pos_true_cnt += 1
    elif output_list == [] and labels != []:
        neg_true_cnt += 1
    elif output_list == [] and labels == []:
        neg_false_cnt += 1
    elif output_list != [] and labels == []:
        pos_false_cnt += 1

    return match_size, prediction, groud_truth, pos_true_cnt, neg_true_cnt, neg_false_cnt, pos_false_cnt

# ----------------------------------------
# Text comparison metrics (for text generation)
# ----------------------------------------
def normalize_answer(s: str) -> str:
    """Standard normalization for text comparison."""
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

# --------------------------
# Reworked balanced metrics
# --------------------------
def normalize_balanced(s: str) -> str:
    """
    Conservative normalization for balanced metrics:
    - Use same base normalization as normalize_answer
    - Light stemming only for longer words (very conservative)
    """
    s = normalize_answer(s)
    tokens = []
    for t in s.split():
        # conservative stemming for longer tokens only
        if len(t) > 6:
            if t.endswith("ing") and len(t) > 7:
                t = t[:-3]
            elif t.endswith("ed") and len(t) > 6:
                t = t[:-2]
        tokens.append(t)
    return " ".join(tokens)

def token_similarity(a: str, b: str) -> float:
    """Token-level similarity [0..1] using SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()

def balanced_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """
    Balanced F1 with proportional soft matching:
    Returns (f1, precision, recall).
    Conservative thresholds provide partial credit for near-matches.
    """
    pred_tokens = normalize_balanced(pred).split()
    gold_tokens = normalize_balanced(gold).split()
    
    if not gold_tokens:
        if not pred_tokens:
            return (1.0, 1.0, 1.0)
        else:
            return (0.0, 0.0, 0.0)
    if not pred_tokens:
        return (0.0, 0.0, 0.0)

    # Tunable thresholds for conservative partial credit
    HIGH = 0.85
    MID = 0.70

    # Recall: how well gold tokens are covered by pred tokens
    total_rec = 0.0
    for gt in gold_tokens:
        best = 0.0
        for pt in pred_tokens:
            sim = token_similarity(gt, pt)
            if sim > best:
                best = sim
        if best >= HIGH:
            total_rec += best
        elif best >= MID:
            total_rec += best * 0.6
        # below MID: no credit

    rec = total_rec / len(gold_tokens)

    # Precision: how many pred tokens are correct w.r.t gold tokens
    total_prec = 0.0
    for pt in pred_tokens:
        best = 0.0
        for gt in gold_tokens:
            sim = token_similarity(pt, gt)
            if sim > best:
                best = sim
        if best >= HIGH:
            total_prec += best
        elif best >= MID:
            total_prec += best * 0.6

    prec = total_prec / len(pred_tokens)

    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec

def balanced_em(pred: str, gold: str) -> float:
    """
    Conservative balanced exact match:
    - Return 1.0 if normalized strings equal
    - Else compute average best-token similarity and Jaccard; only count as EM if extremely high
    """
    pred_norm = normalize_balanced(pred)
    gold_norm = normalize_balanced(gold)
    
    if pred_norm == gold_norm:
        return 1.0
    
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    # average best similarity from gold->pred
    sum_best = 0.0
    for gt in gold_tokens:
        best = max((token_similarity(gt, pt) for pt in pred_tokens), default=0.0)
        sum_best += best
    avg_sim = sum_best / len(gold_tokens)

    # jaccard similarity
    intersection = len(set(pred_tokens) & set(gold_tokens))
    union = len(set(pred_tokens) | set(gold_tokens))
    jaccard = intersection / union if union > 0 else 0.0

    # Decision thresholds (conservative)
    if avg_sim >= 0.90 or jaccard >= 0.90:
        return 1.0
    return 0.0

# ----------------------------------------
# Detection of task type
# ----------------------------------------
def detect_task_type(output: str) -> str:
    """Detect whether output is for hallucination detection or text generation."""
    if not output or output.strip() == "":
        return "empty"
    
    # Try to parse as list (hallucination detection)
    try:
        parsed = literal_eval(output)
        if isinstance(parsed, (list, tuple)):
            return "hallucination_detection"
    except Exception:
        pass
    
    # Otherwise it's text generation
    return "text_generation"

# ----------------------------------------
# Self-RAG generation
# ----------------------------------------
SELF_RAG_REPO = "selfrag/selfrag_llama2_7b"

def _prepare_tokenizer(model_source: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def generate_predictions_on_dataset(dataset, batch_size:int, max_tokens:int, model_source: str = SELF_RAG_REPO):
    """Generate predictions using Self-RAG model."""
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[generate] Loading model from {model_source} on {device}...")
    tokenizer = _prepare_tokenizer(model_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.eval()

    out_list = []
    with torch.no_grad():
        total = len(dataset)
        for i in range(0, total, batch_size):
            batch = dataset[i:min(i+batch_size, total)]
            prompts = []
            
            for raw_ex in batch:
                if isinstance(raw_ex, dict):
                    ex = dict(raw_ex)
                else:
                    ex = {"prompt": str(raw_ex)}
                
                prompt_text = ex.get("prompt") or ex.get("question") or ""
                prompt = f"### Instruction: {prompt_text}\n\n### Response:"
                prompts.append(prompt)
            
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            for j, (dec, prompt) in enumerate(zip(decoded, prompts)):
                idx = i + j
                if idx >= total:
                    break
                ans = ""
                if "### Response:" in dec:
                    ans = dec.split("### Response:", 1)[-1].strip()
                else:
                    ans = dec.replace(prompt, "").strip()
                orig = dataset[idx]
                if isinstance(orig, dict):
                    exrec = dict(orig)
                else:
                    exrec = {}
                entry = {
                    "response": exrec.get("response", ""),
                    "labels": exrec.get("labels", []),
                    "task_type": exrec.get("task_type", ""),
                    "output": ans
                }
                out_list.append(entry)
            
            if (i + batch_size) % 100 == 0:
                print(f"[generate] Processed {min(i+batch_size, total)}/{total}")
    
    return out_list

# ----------------------------------------
# Main evaluation
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default=None)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--model_source', type=str, default=SELF_RAG_REPO)
    parser.add_argument('--chosen_task', type=str, default='QA,Data2txt,Summary')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--bert_model', type=str, default='roberta-large')
    args = parser.parse_args()

    chosen_task = args.chosen_task.split(',')
    
    # Load or generate data
    data = []
    if args.generate:
        from datasets import load_dataset
        ds = load_dataset("leobianco/ragtruth_final_test_set", split="test")
        if args.max_samples > 0:
            ds = ds.select(range(min(len(ds), args.max_samples)))
        print(f"Generating predictions for {len(ds)} examples...")
        data = generate_predictions_on_dataset(ds, args.batch_size, args.max_tokens, args.model_source)
        
        out_file = args.filepath or "ragtruth_preds_selfrag.jsonl"
        with open(out_file, "w") as fout:
            for rec in data:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {out_file}")
    else:
        if not args.filepath or not os.path.exists(args.filepath):
            print("Provide --filepath or use --generate")
            sys.exit(1)
        print(f"Loading predictions from {args.filepath}...")
        with open(args.filepath, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    
    # Detect evaluation mode
    sample_outputs = [d.get('output', '') for d in data[:10]]
    task_types = [detect_task_type(o) for o in sample_outputs]
    is_text_gen = task_types.count('text_generation') > task_types.count('hallucination_detection')
    
    if is_text_gen:
        print("\n=== TEXT GENERATION MODE DETECTED ===")
        print("Evaluating Self-RAG text generation quality\n")
    else:
        print("\n=== HALLUCINATION DETECTION MODE ===")
        print("Evaluating span-level hallucination detection\n")
    
    # Initialize counters
    task_list = chosen_task + ['All']
    record = collections.defaultdict(int)
    
    # For text generation metrics
    text_metrics = {
        'f1': collections.defaultdict(float),
        'em': collections.defaultdict(float),
        'balanced_f1': collections.defaultdict(float),
        'balanced_prec': collections.defaultdict(float),
        'balanced_rec': collections.defaultdict(float),
        'balanced_em': collections.defaultdict(float),
        'count': collections.defaultdict(int)
    }
    
    # For BERTScore
    all_preds = []
    all_golds = []
    
    # Process each example
    for i, sample in enumerate(data):
        response = sample.get('response', '')
        labels = sample.get('labels', [])
        output = sample.get('output', '')
        task = sample.get('task_type', sample.get('task', ''))
        
        if is_text_gen:
            # Text generation evaluation
            pred_text = str(output).strip()
            gold_text = str(response).strip()
            
            # Show first few examples for debugging
            if i < 3:
                print(f"\nExample {i+1}:")
                print(f"Predicted: {pred_text[:150]}..." if len(pred_text) > 150 else f"Predicted: {pred_text}")
                print(f"Reference: {gold_text[:150]}..." if len(gold_text) > 150 else f"Reference: {gold_text}")
            
            if pred_text and gold_text:
                # Traditional metrics
                f1 = f1_score(pred_text, gold_text)
                em = exact_match_score(pred_text, gold_text)
                
                # Balanced metrics
                b_f1, b_prec, b_rec = balanced_f1(pred_text, gold_text)
                b_em = balanced_em(pred_text, gold_text)
                
                # Show scores for first example
                if i == 0:
                    print(f"  Traditional F1: {f1:.3f}, EM: {em:.3f}")
                    print(f"  Balanced F1: {b_f1:.3f}, Prec: {b_prec:.3f}, Rec: {b_rec:.3f}, EM: {b_em:.3f}")
                
                for task_cur in task_list:
                    if task_cur == task or task_cur == 'All':
                        text_metrics['f1'][task_cur] += f1
                        text_metrics['em'][task_cur] += em
                        text_metrics['balanced_f1'][task_cur] += b_f1
                        text_metrics['balanced_prec'][task_cur] += b_prec
                        text_metrics['balanced_rec'][task_cur] += b_rec
                        text_metrics['balanced_em'][task_cur] += b_em
                        text_metrics['count'][task_cur] += 1
            
            all_preds.append(pred_text if pred_text else "no response")
            all_golds.append(gold_text if gold_text else "no reference")
            
        else:
            # Hallucination detection evaluation (span-level)
            res = cal_span_metrics(response, labels, output)
            temp_tp, temp_pred_len, temp_gold_len, pos_true, neg_true, neg_false, pos_false = res
            
            for task_cur in task_list:
                if task_cur == task or task_cur == 'All':
                    record['match_size_'+task_cur] += temp_tp
                    record['prediction_'+task_cur] += temp_pred_len
                    record['ground_truth_'+task_cur] += temp_gold_len
                    record['pos_true_cnt_'+task_cur] += pos_true
                    record['neg_true_cnt_'+task_cur] += neg_true
                    record['neg_false_cnt_'+task_cur] += neg_false
                    record['pos_false_cnt_'+task_cur] += pos_false
    
    # Print results
    if is_text_gen:
        # Text generation results
        print("\n" + "="*60)
        print("TEXT GENERATION EVALUATION RESULTS")
        print("="*60)
        
        for task_cur in task_list:
            if text_metrics['count'][task_cur] > 0:
                count = text_metrics['count'][task_cur]
                print(f"\nTask: {task_cur} ({count} examples)")
                print(f"Traditional F1: {text_metrics['f1'][task_cur]/count:.4f}")
                print(f"Traditional EM: {text_metrics['em'][task_cur]/count:.4f}")
                print(f"Balanced F1: {text_metrics['balanced_f1'][task_cur]/count:.4f}")
                print(f"Balanced Precision: {text_metrics['balanced_prec'][task_cur]/count:.4f}")
                print(f"Balanced Recall: {text_metrics['balanced_rec'][task_cur]/count:.4f}")
                print(f"Balanced EM: {text_metrics['balanced_em'][task_cur]/count:.4f}")
        
        # Compute BERTScore
        print("\nComputing BERTScore...")
        try:
            P, R, F1 = bert_score(
                all_preds,
                all_golds,
                model_type=args.bert_model,
                lang='en',
                rescale_with_baseline=True,  # Use rescaling for more interpretable scores
                verbose=False
            )
            print(f"\nBERTScore (with {args.bert_model}):")
            print(f"Precision: {float(P.mean()):.4f}")
            print(f"Recall: {float(R.mean()):.4f}")
            print(f"F1: {float(F1.mean()):.4f}")
        except Exception as e:
            print(f"BERTScore failed: {e}")
        
    else:
        # Hallucination detection results
        print("\n" + "="*60)
        print("HALLUCINATION DETECTION RESULTS")
        print("="*60)
        
        for task_cur in task_list:
            if record['ground_truth_'+task_cur] > 0 or record['prediction_'+task_cur] > 0:
                print(f"\nTask: {task_cur}")
                
                # Span-level metrics
                rec = record['match_size_'+task_cur] / record['ground_truth_'+task_cur] if record['ground_truth_'+task_cur] else 0
                prec = record['match_size_'+task_cur] / record['prediction_'+task_cur] if record['prediction_'+task_cur] else 0
                f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
                print(f"Span-level - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
                
                # Case-level metrics
                total_pos = record['pos_true_cnt_'+task_cur] + record['pos_false_cnt_'+task_cur]
                total_with_labels = record['pos_true_cnt_'+task_cur] + record['neg_true_cnt_'+task_cur]
                case_prec = record['pos_true_cnt_'+task_cur] / total_pos if total_pos else 0
                case_rec = record['pos_true_cnt_'+task_cur] / total_with_labels if total_with_labels else 0
                case_f1 = 2 * case_prec * case_rec / (case_prec + case_rec) if (case_prec + case_rec) > 0 else 0
                print(f"Case-level - Precision: {case_prec:.4f}, Recall: {case_rec:.4f}, F1: {case_f1:.4f}")
    
    print("="*60)
    print(f"\nTotal examples evaluated: {len(data)}")

if __name__ == '__main__':
    main()
