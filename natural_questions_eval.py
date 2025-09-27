#!/usr/bin/env python3
"""
Single-file pipeline: prepare NQ validation, run Self-RAG HF inference (batched),
and run the provided Natural Questions evaluation (unchanged).

This file:
 - keeps the original evaluation logic and metrics (traditional F1 and EM, BERTScore)
 - fixes a robustness bug where gold/predictions could be lists/dicts (coerces to string)
 - *adds* the "balanced" forgiving EM and F1 implementation you provided,
   and computes balanced_em and balanced_f1 alongside the traditional metrics
 - provides pipeline helper functions to download NQ validation, run HF Self-RAG
   batched inference, and run the evaluator on the generated predictions.

Usage examples:
  # prepare gold, run inference on 2000 samples, batch size 4, 512 tokens, and evaluate:
  python run_naturalquestions_eval.py \
    --prepare_gold \
    --run_inference \
    --gold_out gold.json \
    --pred_out predictions.json \
    --model_name selfrag/selfrag_llama2_7b \
    --num_samples 2000 \
    --batch_size 4 \
    --max_new_tokens 512 \
    --eval_output_file results.json
"""

# -------------------------
# === BEGIN: Unmodified NQ evaluator code (kept exactly, with small safe robustness fixes) ===
# -------------------------
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score as bertscore
import time
import re
from pathlib import Path

def extract_core_answer(prediction: str, question: str) -> str:
    """Extract the core answer from the model prediction, removing fluff and repetition."""
    # Remove any [Relevant], [Fully supported], etc. tags
    prediction = re.sub(r'\[.*?\]', '', prediction)
    # Remove repetition by splitting into sentences and keeping unique ones
    sentences = re.split(r'(?<=[.!?])\s+', prediction)
    unique_sentences = []
    seen_normalized = set()
    for sent in sentences:
        normalized_sent = sent.strip().lower()
        if normalized_sent and normalized_sent not in seen_normalized:
            unique_sentences.append(sent.strip())
            seen_normalized.add(normalized_sent)
    prediction = ' '.join(unique_sentences).strip()
    # Try to extract answer directly following the question
    question_pattern = re.escape(question.strip())
    match = re.search(rf'{question_pattern}.*?([A-Z][^.!?]*(?:[.!?]|$))', prediction, re.IGNORECASE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted
    # Fallback: take the first sentence
    first_sentence = re.split(r'[.!?]', prediction, 1)[0].strip()
    if first_sentence:
        return first_sentence
    return prediction.strip()

def load_creamrag_model():
    """Load the CREAMRAG merged model using transformers."""
    model_path = "/workspace/cream_rag_data/backups/20250902_021122/CREAMRAG/checkpoints/creamrag_merged_model"
    print(f"Loading CREAMRAG merged model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, question, max_new_tokens=512):
    """Generate response using the model."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# -------------------------
# === BEGIN: Robust coercion + normalization helpers (fix for 'list' object has no attribute 'lower') ===
# -------------------------
def answer_to_text(ans):
    """
    Coerce various annotation/prediction formats to a single string.
    Handles: str, dict, list (of strings or dicts), or other types.
    """
    if ans is None:
        return ""
    if isinstance(ans, str):
        return ans
    if isinstance(ans, dict):
        # try common string fields first
        for k in ('text', 'answer_text', 'short_answer', 'long_answer'):
            if k in ans:
                v = ans[k]
                return answer_to_text(v)
        # sometimes short_answers is a list inside the dict
        if 'short_answers' in ans:
            return answer_to_text(ans['short_answers'])
        # fallback
        try:
            return json.dumps(ans)
        except Exception:
            return str(ans)
    if isinstance(ans, list):
        # Try to pick the first non-empty coerced value
        for item in ans:
            t = answer_to_text(item)
            if t:
                return t
        return ""
    # fallback for numbers, booleans, etc.
    return str(ans)

def normalize_answer(s):
    """Normalize answer for F1/EM calculation."""
    # ensure we operate on a string
    if not isinstance(s, str):
        s = answer_to_text(s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Traditional exact-match / F1 (same semantics as original evaluator)
def f1_score(prediction, ground_truth):
    """Calculate traditional F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def exact_match_score(prediction, ground_truth):
    """Calculate exact match score."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# -------------------------
# === END: Robust coercion + normalization helpers ===
# -------------------------

# -------------------------
# === BEGIN: Balanced/flexible metrics (from user's example) ===
# -------------------------
from difflib import SequenceMatcher

def normalize_strict(s):
    """Alias to strict normalization (keeps original strict behavior)."""
    return normalize_answer(s)

def normalize_balanced(s):
    """Balanced forgiving normalization (as provided in the example)."""
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
        # small domain-specific example mapping from the user's snippet
        if t in ["armenian", "armenia"]:
            t = "armenia"
        tokens.append(t)
    return " ".join(tokens)

def numeric_overlap(pred, gold):
    pred_nums = set(re.findall(r"\d+", pred))
    gold_nums = set(re.findall(r"\d+", gold))
    return len(pred_nums & gold_nums) > 0

def token_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def balanced_f1(pred, gold):
    """Return (f1, precision, recall) using forgiving token-similarity matching."""
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

def semantic_scores(preds, golds, model_type="roberta-large-mnli"):
    """Compute BERTScore (semantic) for a batch."""
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
# -------------------------
# === END: Balanced/flexible metrics ===
# -------------------------

def evaluate_nq_with_bertscore(predictions_file, gold_file, model_path=None, generate_predictions=False):
    """Evaluate Natural Questions with BERTScore and both traditional and balanced metrics."""
    # Load or generate predictions (the original code supports generating with CREAMRAG; keep it)
    if generate_predictions and model_path:
        print("Generating predictions with CREAMRAG model...")
        model, tokenizer = load_creamrag_model()
        # Load questions from gold file
        with open(gold_file, 'r') as f:
            gold_data = json.load(f)
        predictions = {}
        for i, item in enumerate(gold_data):
            if i >= 100:  # Limit to first 100 for demo generation (unchanged behavior)
                break
            question = item.get('question', '')
            print(f"Processing {i+1}/100: {question[:50]}...")
            start_time = time.time()
            try:
                response = generate_response(model, tokenizer, question)
                core_answer = extract_core_answer(response, question)
                predictions[item['id']] = core_answer
            except Exception as e:
                print(f"Error processing {item['id']}: {e}")
                predictions[item['id']] = ""
        # Save predictions
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions to {predictions_file}")

    # Load predictions and gold data
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    with open(gold_file, 'r') as f:
        gold_data = json.load(f)

    # Align predictions and gold answers (coerce to strings)
    pred_texts, gold_texts, ids = [], [], []
    for item in gold_data:
        qid = item['id']
        if qid in predictions:
            pred_texts.append(answer_to_text(predictions[qid]))
            # Handle multiple possible answers robustly (take first)
            answers = item.get('answers', [])
            if answers:
                gold_texts.append(answer_to_text(answers[0]))
            else:
                gold_texts.append("")
            ids.append(qid)

    print(f"Evaluating {len(pred_texts)} question-answer pairs...")

    # Traditional metrics (unchanged semantics)
    total_f1 = 0.0
    total_em = 0.0
    # Balanced metrics accumulators
    total_bal_f1 = 0.0
    total_bal_em = 0.0
    total_bal_prec = 0.0
    total_bal_rec = 0.0

    for pred, gold in zip(pred_texts, gold_texts):
        # Traditional
        total_f1 += f1_score(pred, gold)
        total_em += int(exact_match_score(pred, gold))
        # Balanced
        bf1, bprec, brec = balanced_f1(pred, gold)
        bem = balanced_em(pred, gold)
        total_bal_f1 += bf1
        total_bal_prec += bprec
        total_bal_rec += brec
        total_bal_em += bem

    n = len(pred_texts) if pred_texts else 0
    avg_f1 = total_f1 / n if n else 0.0
    avg_em = total_em / n if n else 0.0
    avg_bal_f1 = total_bal_f1 / n if n else 0.0
    avg_bal_em = total_bal_em / n if n else 0.0
    avg_bal_prec = total_bal_prec / n if n else 0.0
    avg_bal_rec = total_bal_rec / n if n else 0.0

    # Compute BERTScore (semantic similarity) for all pairs (matches original evaluator's use)
    print("Computing BERTScore (this may take a while)...")
    P, R, F1 = bertscore(
        pred_texts,
        gold_texts,
        model_type='roberta-large',
        lang='en',
        rescale_with_baseline=False,
        verbose=True
    )

    # Optionally compute rescaled semantic sampling like the hotpot example (sample first 50)
    sample_size = min(50, len(pred_texts))
    sem_sample = semantic_scores(pred_texts[:sample_size], gold_texts[:sample_size]) if sample_size > 0 else None

    results = {
        'num_samples': n,
        # traditional
        'traditional_f1': avg_f1,
        'traditional_em': avg_em,
        # balanced / forgiving
        'balanced_f1': avg_bal_f1,
        'balanced_em': avg_bal_em,
        'balanced_precision': avg_bal_prec,
        'balanced_recall': avg_bal_rec,
        # bertscore (main)
        'bertscore_precision': float(P.mean()) if n else 0.0,
        'bertscore_recall': float(R.mean()) if n else 0.0,
        'bertscore_f1': float(F1.mean()) if n else 0.0,
        'bertscore_model': 'roberta-large',
        'bertscore_rescale_with_baseline': False,
    }

    # Add semantic sample averages if computed (mirrors the Hotpot example behavior)
    if sem_sample:
        results.update({
            'semantic_sample_precision': sum(sem_sample['precision'])/len(sem_sample['precision']),
            'semantic_sample_recall': sum(sem_sample['recall'])/len(sem_sample['recall']),
            'semantic_sample_f1': sum(sem_sample['f1'])/len(sem_sample['f1']),
            'semantic_sample_size': sample_size
        })

    return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced NQ evaluation with BERTScore')
    parser.add_argument('--predictions_file', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--gold_file', type=str, required=True,
                       help='Path to gold data JSON file')
    parser.add_argument('--generate_predictions', action='store_true',
                       help='Generate predictions using CREAMRAG model')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save results JSON file')
    args = parser.parse_args()
    results = evaluate_nq_with_bertscore(
        args.predictions_file,
        args.gold_file,
        generate_predictions=args.generate_predictions
    )
    print("\n" + "="*60)
    print("NATURAL QUESTIONS EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Traditional F1: {results['traditional_f1']:.4f}")
    print(f"Traditional EM: {results['traditional_em']:.4f}")
    print(f"Balanced F1: {results['balanced_f1']:.4f}")
    print(f"Balanced EM: {results['balanced_em']:.4f}")
    print(f"Balanced Precision: {results['balanced_precision']:.4f}")
    print(f"Balanced Recall: {results['balanced_recall']:.4f}")
    print(f"BERTScore Precision: {results['bertscore_precision']:.4f}")
    print(f"BERTScore Recall: {results['bertscore_recall']:.4f}")
    print(f"BERTScore F1: {results['bertscore_f1']:.4f}")
    if 'semantic_sample_f1' in results:
        print(f"Semantic sample (n={results['semantic_sample_size']}) F1: {results['semantic_sample_f1']:.4f}")
    print("="*60)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")

# -------------------------
# === END: Unmodified NQ evaluator code (with balanced metrics integrated) ===
# -------------------------

# -------------------------
# === BEGIN: Additional helper functions and CLI wiring (non-evaluative) ===
# -------------------------
from datasets import load_dataset
import sys

def prepare_nq_validation(gold_out_path: str = "gold.json", max_samples: int = None):
    """
    Download Natural Questions validation split using:
      load_dataset("google-research-datasets/natural_questions", "default")
    and write to gold_out_path as JSON list of objects with fields:
      - id
      - question
      - answers (list)
    This version is robust to various annotation formats (dict, list, or JSON string).
    """
    import json as _json  # local alias so we don't shadow outer json
    print("Loading Natural Questions validation (config 'default') via datasets.load_dataset ...")
    ds = load_dataset("google-research-datasets/natural_questions", "default", split="validation")

    out = []
    for i, ex in enumerate(ds):
        if max_samples and max_samples > 0 and i >= max_samples:
            break
        q = ex.get("question_text") or ex.get("question") or ex.get("query") or ""
        # try common annotation fields to collect answers, otherwise empty list
        answers = []

        anns = ex.get("annotations") or []
        # anns can be: list of dicts, list of strings (serialized), a single dict, etc.
        if anns:
            # normalize to a list
            if isinstance(anns, dict):
                anns = [anns]
            for ann in anns:
                # If ann is a JSON string, try to parse it
                if isinstance(ann, str):
                    try:
                        ann_parsed = _json.loads(ann)
                    except Exception:
                        # not JSON - skip or treat as a direct short answer string
                        text = ann.strip()
                        if text:
                            answers.append(text)
                        continue
                    ann = ann_parsed

                # If ann is now a dict, look for short_answers
                if isinstance(ann, dict):
                    short_answers = ann.get("short_answers") or []
                    # short_answers can be list of dicts or list of strings
                    for sa in short_answers:
                        if isinstance(sa, dict):
                            # some variants store text under 'text' or 'answer_text'
                            t = sa.get("text") or sa.get("answer_text") or None
                        else:
                            t = sa
                        if t:
                            answers.append(t)
                    # also try 'long_answer' text if present and no short answers found
                    if not short_answers:
                        # long_answer might be dict with 'text'
                        la = ann.get("long_answer")
                        if isinstance(la, dict):
                            t = la.get("text")
                            if t:
                                answers.append(t)
                elif isinstance(ann, list):
                    # ann is a list of short-answer entries (strings or dicts)
                    for sa in ann:
                        if isinstance(sa, dict):
                            t = sa.get("text") or sa.get("answer_text") or None
                        else:
                            t = sa
                        if t:
                            answers.append(t)

        # fallback: some dataset variants have 'answers' field
        if not answers and ex.get("answers"):
            if isinstance(ex["answers"], list):
                answers = [a for a in ex["answers"] if isinstance(a, str)]

        item = {
            "id": str(ex.get("example_id", i)),
            "question": q,
            "answers": answers
        }
        out.append(item)

    Path(gold_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(gold_out_path, "w") as f:
        _json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} items to {gold_out_path}")

def run_selfrag_inference_and_save(pred_out_path: str = "predictions.json",
                                   gold_file: str = None,
                                   model_name: str = "selfrag/selfrag_llama2_7b",
                                   num_samples: int = None,
                                   batch_size: int = 4,
                                   max_new_tokens: int = 256,
                                   device: str = None):
    """
    Load questions from gold_file (if provided) or from NQ validation directly,
    load Self-RAG HF model (exact API calls you specified) and generate predictions
    in batches. Predictions are saved as JSON mapping {id: generated_answer}.
    """
    # collect questions
    ids = []
    questions = []
    if gold_file:
        print(f"Loading questions from gold file: {gold_file}")
        with open(gold_file, "r") as f:
            gold = json.load(f)
        for i, item in enumerate(gold):
            if num_samples and num_samples > 0 and i >= num_samples:
                break
            ids.append(item["id"])
            questions.append(item.get("question", ""))
    else:
        print("No gold file provided, loading NQ validation directly.")
        ds = load_dataset("google-research-datasets/natural_questions", "default", split="validation")
        for i, ex in enumerate(ds):
            if num_samples and num_samples > 0 and i >= num_samples:
                break
            ids.append(str(ex.get("example_id", i)))
            q = ex.get("question_text") or ex.get("question") or ex.get("query") or ""
            questions.append(q)

    total = len(ids)
    print(f"Preparing to generate {total} items (batch_size={batch_size}, max_new_tokens={max_new_tokens})")

    print(f"Loading tokenizer and model from HF id: {model_name}")
    # Use exactly the loading calls you requested:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # move to device if possible
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving model to device: {device}")
    model.to(device)
    model.eval()

    preds = {}
    i = 0
    while i < total:
        batch_ids = ids[i: i + batch_size]
        batch_qs = questions[i: i + batch_size]
        print(f"Generating batch {i//batch_size + 1}: items {i+1}-{i+len(batch_qs)}")

        # tokenize prompts as a batch
        prompts = [f"Question: {q}\nAnswer:" for q in batch_qs]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_lengths = inputs['attention_mask'].sum(dim=1).tolist()  # tokens per example before generation

        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        except Exception as e:
            print(f"Error during generation for batch starting at index {i}: {e}")
            # on failure, fill batch ids with empty strings and continue
            for bid in batch_ids:
                preds[bid] = ""
            i += batch_size
            continue

        # outputs shape: (batch_size, seq_len)
        # decode each output and cut off the prompt by using input_lengths
        for j, out in enumerate(outputs):
            inp_len = int(input_lengths[j])
            # guard in case generated length <= inp_len
            if out.shape[0] <= inp_len:
                gen_text = ""
            else:
                gen_tokens = out[inp_len:]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            qid = batch_ids[j]
            preds[qid] = gen_text
            print(f"  -> id={qid} generated {len(gen_text)} chars")

        i += batch_size

    Path(pred_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pred_out_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {pred_out_path}")

# -------------------------
# === END: Additional helpers ===
# -------------------------

# -------------------------
# === Unified CLI: run pipeline or the original evaluator ===
# -------------------------
def pipeline_entry():
    parser = argparse.ArgumentParser(description="All-in-one: prepare NQ validation, run Self-RAG inference (batched), and evaluate with the provided NQ evaluator.")
    parser.add_argument("--prepare_gold", action="store_true", help="Download NQ validation and write gold file")
    parser.add_argument("--gold_out", type=str, default="gold.json", help="Path to write gold JSON file")
    parser.add_argument("--max_gold_samples", type=int, default=None, help="Limit number of gold samples (for quick tests). Use -1 or 0 for all")

    parser.add_argument("--run_inference", action="store_true", help="Run HF Self-RAG inference to produce predictions")
    parser.add_argument("--pred_out", type=str, default="predictions.json", help="Path to write predictions mapping {id: answer}")
    parser.add_argument("--model_name", type=str, default="selfrag/selfrag_llama2_7b", help="HF model id for Self-RAG (default matches your instruction)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate (set 0 or -1 for all)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation length per prompt")

    parser.add_argument("--eval_predictions_file", type=str, default=None, help="Predictions file to pass to evaluator (if omitted uses --pred_out)")
    parser.add_argument("--eval_gold_file", type=str, default=None, help="Gold file to pass to evaluator (if omitted uses --gold_out)")
    parser.add_argument("--eval_output_file", type=str, default=None, help="If set, evaluator will save JSON results here")

    parser.add_argument("--no_gpu_warn", action="store_true", help="Suppress GPU availability warning")

    args = parser.parse_args()

    # Prepare gold
    if args.prepare_gold:
        max_g = None if (args.max_gold_samples in (None, -1, 0)) else args.max_gold_samples
        print("=== Preparing gold file ===")
        prepare_nq_validation(args.gold_out, max_samples=max_g)

    # Run inference
    if args.run_inference:
        if not args.no_gpu_warn and not torch.cuda.is_available():
            print("WARNING: CUDA not available — inference will run on CPU and may be very slow for large models.")
        num_samples = None if args.num_samples in (None, -1, 0) else args.num_samples
        print("=== Running Self-RAG HF inference (batched) ===")
        gold_file_for_inference = args.gold_out if args.eval_gold_file is None else args.eval_gold_file
        run_selfrag_inference_and_save(pred_out_path=args.pred_out,
                                       gold_file=gold_file_for_inference,
                                       model_name=args.model_name,
                                       num_samples=num_samples,
                                       batch_size=args.batch_size,
                                       max_new_tokens=args.max_new_tokens)

    # Evaluate (call your evaluator exactly as-is)
    eval_preds = args.eval_predictions_file if args.eval_predictions_file else args.pred_out
    eval_gold = args.eval_gold_file if args.eval_gold_file else args.gold_out

    if eval_preds is None or eval_gold is None:
        print("Skipping evaluation: predictions or gold file not provided.")
        return

    print("=== Running evaluation (using your evaluator code unchanged except balanced metrics integrated) ===")
    # Re-use the evaluator's function directly
    results = evaluate_nq_with_bertscore(
        eval_preds,
        eval_gold,
        generate_predictions=False  # keep as False: we generated predictions with the HF model above
    )

    # Print results (same output format as original main)
    print("\n" + "="*60)
    print("NATURAL QUESTIONS EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Traditional F1: {results['traditional_f1']:.4f}")
    print(f"Traditional EM: {results['traditional_em']:.4f}")
    print(f"Balanced F1: {results['balanced_f1']:.4f}")
    print(f"Balanced EM: {results['balanced_em']:.4f}")
    print(f"Balanced Precision: {results['balanced_precision']:.4f}")
    print(f"Balanced Recall: {results['balanced_recall']:.4f}")
    print(f"BERTScore Precision: {results['bertscore_precision']:.4f}")
    print(f"BERTScore Recall: {results['bertscore_recall']:.4f}")
    print(f"BERTScore F1: {results['bertscore_f1']:.4f}")
    if 'semantic_sample_f1' in results:
        print(f"Semantic sample (n={results['semantic_sample_size']}) F1: {results['semantic_sample_f1']:.4f}")
    print("="*60)
    if args.eval_output_file:
        with open(args.eval_output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.eval_output_file}")

# Entrypoint switch: if pipeline-related flags are present, run pipeline_entry().
# Otherwise fall back to the original evaluator entrypoint (main) which expects
# --predictions_file and --gold_file.
if __name__ == "__main__":
    # Detect pipeline-style flags in argv
    pipeline_flags = {
        "--prepare_gold", "--run_inference", "--model_name", "--max_gold_samples",
        "--pred_out", "--eval_predictions_file", "--eval_gold_file", "--eval_output_file",
        "--num_samples", "--batch_size", "--max_new_tokens", "--no_gpu_warn"
    }
    args_present = set(arg for arg in sys.argv[1:] if arg.startswith("--"))
    if args_present & pipeline_flags:
        pipeline_entry()
    else:
        # fall back to original evaluator behaviour (unchanged)
        main()
