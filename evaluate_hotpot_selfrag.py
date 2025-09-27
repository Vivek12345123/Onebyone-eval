#!/usr/bin/env python3
# evaluate_hotpot_selfrag.py
"""
Runs a Self-RAG style retrieval+generation evaluation on HotpotQA (distractor val).
Keeps all original metric calculations exactly the same as provided.
Defaults: TOTAL_SAMPLES=2000, BATCH_SIZE=4, MAX_TOKENS=512
"""
import os
import sys
import json
import re
import string
import textwrap
from collections import Counter
from difflib import SequenceMatcher
from typing import List

# Attempt to import evaluate helpers; if not present, provide a minimal compatible fallback.
try:
    from evaluate_hotpot_official import f1_score, normalize_answer
except Exception:
    # Minimal compatible fallback for normalize_answer and f1_score (keeps same contract)
    import re as _re
    import string as _string

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return _re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(_string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s or ""))))

    def f1_score(prediction, ground_truth):
        pred_tokens = normalize_answer(prediction).split()
        gold_tokens = normalize_answer(ground_truth).split()
        # count matches like SQuAD
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return (int(pred_tokens == gold_tokens), 0.0, 0.0)
        if num_same == 0:
            return (0.0, 0.0, 0.0)
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return (f1, precision, recall)

# -------------------------------
# User-enforced parameters (defaults)
# -------------------------------
DEFAULT_TOTAL_SAMPLES = 2000
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_TOKENS = 512
RETRIEVE_K = 5
DEFAULT_PREDICTIONS_FILE = "llama_7b_hotpot_200_predictions.jsonl"
DEFAULT_MODEL_NAME = "selfrag/selfrag_llama2_7b"
DEFAULT_TFIDF_MAX_FEATURES = 20000

# -------------------------------
# Helper: strict normalization
# -------------------------------
def normalize_strict(s):
    return normalize_answer(s or "")

# -------------------------------
# Balanced forgiving normalization (identical logic)
# -------------------------------
def normalize_balanced(s):
    s = normalize_answer(s or "")
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
# Numeric overlap (identical)
# -------------------------------
def numeric_overlap(pred, gold):
    pred_nums = set(re.findall(r"\d+", pred or ""))
    gold_nums = set(re.findall(r"\d+", gold or ""))
    return len(pred_nums & gold_nums) > 0

# -------------------------------
# Balanced F1 with soft token similarity (identical)
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
# BERTScore wrapper (identical)
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

# -------------------------------
# Retrieval helpers
# -------------------------------
def retrieve_top_k(corpus_matrix, vectorizer, question, corpus_texts, k=5):
    q_vec = vectorizer.transform([question])
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_vec, corpus_matrix).flatten()
    topk_idx = sims.argsort()[::-1][:k]
    return [corpus_texts[i] for i in topk_idx], topk_idx, sims[topk_idx]

# -------------------------------
# Generation helper — fixed token/device handling
# -------------------------------
def generate_batch_prompts_and_answers(model, tokenizer, prompts: List[str], device, max_new_tokens=512):
    """
    Generate answers for a list of prompts (batch).
    Deterministic (greedy). Returns list[str] answers.
    """
    import torch
    # Tokenize (return PyTorch tensors)
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    # Move tensors to device
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc.get("input_ids")
    attention_mask = enc.get("attention_mask", None)

    # Ensure pad/eos token ids are provided
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    # decode
    decoded = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    answers = []
    for txt, prompt in zip(decoded, prompts):
        if "Answer:" in txt:
            txt = txt.split("Answer:")[-1].strip()
        elif "answer:" in txt:
            txt = txt.split("answer:")[-1].strip()
        else:
            if txt.startswith(prompt):
                txt = txt[len(prompt):].strip()
        answers.append(txt)
    return answers

# -------------------------------
# Main pipeline
# -------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA with Self-RAG (prompted retrieval+generation).")
    parser.add_argument("--predictions-file", type=str, default=DEFAULT_PREDICTIONS_FILE,
                        help="Filename to save predictions JSONL (overwritten).")
    parser.add_argument("--run-inference", action="store_true",
                        help="If set, run retrieval+generation to create predictions. Otherwise expects predictions file present and will only evaluate.")
    parser.add_argument("--device", type=str, default=None, help="torch device override, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="HF model name to load")
    parser.add_argument("--retrieve-k", type=int, default=RETRIEVE_K, help="Top-k paragraphs to retrieve")
    parser.add_argument("--tfidf-max-features", type=int, default=DEFAULT_TFIDF_MAX_FEATURES, help="TF-IDF max features")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_TOTAL_SAMPLES, help="Max eval samples (default 2000)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Generation batch size (default 4)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max new tokens to generate (default 512)")
    args = parser.parse_args()

    predictions_file = args.predictions_file
    TOTAL_SAMPLES = args.max_samples
    BATCH_SIZE = args.batch_size
    MAX_TOKENS = args.max_tokens

    # If requested, run inference (retrieval + generation)
    if args.run_inference:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Load HotpotQA (distractor validation)
        print("Loading HotpotQA (distractor validation split)...")
        try:
            ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        except Exception as e:
            # Try alternative HF id for compatibility
            print("Warning: load_dataset default failed:", e)
            ds = load_dataset("hotpot_qa", "distractor", split="validation")

        total_available = len(ds)
        cap = min(TOTAL_SAMPLES, total_available)
        print(f"Dataset size (validation distractor): {total_available}. Processing cap: {cap}")

        # Build paragraph corpus for retrieval (robust handling)
        print("Building paragraph corpus for retrieval...")
        corpus_texts = []
        for ex_idx, ex in enumerate(ds):
            # canonical HF shape: ex['context'] = list of [title, paragraph_text]
            if 'context' in ex and isinstance(ex['context'], list):
                for item in ex['context']:
                    try:
                        # item is often [title, paragraph_text]
                        title = item[0]
                        second = item[1]
                    except Exception:
                        continue
                    if isinstance(second, list):
                        # second may be list of sentences or list of paras
                        for p in second:
                            para_text = " ".join(p).strip() if isinstance(p, list) else str(p).strip()
                            if para_text:
                                corpus_texts.append(para_text)
                    else:
                        para_text = str(second).strip()
                        if para_text:
                            corpus_texts.append(para_text)

            # alternate key 'paragraphs'
            elif 'paragraphs' in ex and isinstance(ex['paragraphs'], list):
                for p in ex['paragraphs']:
                    para_text = " ".join(p).strip() if isinstance(p, list) else str(p).strip()
                    if para_text:
                        corpus_texts.append(para_text)

            # fallback: scan for long string fields or lists
            else:
                for k, v in ex.items():
                    if isinstance(v, str):
                        v_s = v.strip()
                        if len(v_s) > 50:
                            corpus_texts.append(v_s)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                s = item.strip()
                                if len(s) > 50:
                                    corpus_texts.append(s)
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                second = item[1]
                                if isinstance(second, list):
                                    for p in second:
                                        para_text = " ".join(p).strip() if isinstance(p, list) else str(p).strip()
                                        if para_text:
                                            corpus_texts.append(para_text)
                                else:
                                    para_text = str(second).strip()
                                    if len(para_text) > 0:
                                        corpus_texts.append(para_text)

        # dedupe & filter
        corpus_texts = [p for p in dict.fromkeys(corpus_texts) if len(p) > 20]

        if not corpus_texts:
            print("DEBUG: corpus_texts is empty. Inspecting first dataset example to help debug:")
            try:
                example0 = ds[0]
                print("Example keys:", list(example0.keys()))
                ex_str = json.dumps(example0, indent=2, ensure_ascii=False)
                for line in textwrap.wrap(ex_str, width=200):
                    print(line)
            except Exception as e:
                print("Couldn't print first example:", repr(e))
            raise RuntimeError("Failed to build corpus from HotpotQA dataset - unexpected dataset format.")

        # Fit TF-IDF vectorizer
        print(f"Fitting TF-IDF vectorizer over corpus (max_features={args.tfidf_max_features})...")
        vectorizer = TfidfVectorizer(max_features=args.tfidf_max_features, ngram_range=(1,2))
        corpus_matrix = vectorizer.fit_transform(corpus_texts)

        # Model + tokenizer load (safe / try to use device_map if GPU available)
        device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Preparing model '{args.model_name}' on device preference '{device}' ...")
        tokenizer = None
        model = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
            tokenizer.padding_side = 'left'  # Ensure left-padding for decoder-only models
        except Exception as e:
            print("Tokenizer load failed:", e)
            raise

        # Try to load model with device_map="auto" and float16 if GPU present (safer for large models)
        try:
            if "cuda" in device and torch.cuda.is_available():
                print("Attempting to load model with device_map='auto' and torch_dtype=float16 (GPU available)...")
                model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
            else:
                print("Loading model to CPU (no GPU or device override). This may be slow.")
                model = AutoModelForCausalLM.from_pretrained(args.model_name)
        except Exception as e:
            print("Model.load attempt with device_map failed or raised. Falling back to default load:", e)
            try:
                model = AutoModelForCausalLM.from_pretrained(args.model_name)
            except Exception as e2:
                print("Model loading failed:", e2)
                raise

        # If model is loaded with device_map='auto', it will already be on correct devices.
        # If not, move model to chosen device (only for non-sharded loads)
        try:
            # If model has 'is_loaded_in_8bit' or uses device_map, skip .to()
            if not hasattr(model, "is_loaded_in_8bit") and not getattr(model, "hf_device_map", None):
                model.to(device)
        except Exception:
            # ignore if model is already sharded/loaded with accelerate
            pass

        model.eval()

        # Run retrieval+generation for each question, batching prompts for generation
        out_file = open(predictions_file, "w", encoding="utf-8")
        processed = 0
        prompts_buffer = []
        ids_buffer = []
        golds_buffer = []

        for ex_idx in range(cap):
            ex = ds[ex_idx]
            qid = ex.get("id", f"hotpot_{ex_idx}")
            question = ex.get("question", "") or ""
            top_paras, top_idx, top_sims = retrieve_top_k(corpus_matrix, vectorizer, question, corpus_texts, k=args.retrieve_k)
            prompt_parts = []
            for i, p in enumerate(top_paras):
                prompt_parts.append(f"Context {i+1}:\n{p}\n")
            prompt_parts.append(f"Question: {question}\nAnswer:")
            prompt = "\n".join(prompt_parts)

            prompts_buffer.append(prompt)
            ids_buffer.append(qid)
            gold_val = ex.get("answer", "")
            golds_buffer.append([gold_val] if gold_val else [])

            # When buffer full => generate
            if len(prompts_buffer) >= BATCH_SIZE:
                answers = generate_batch_prompts_and_answers(model, tokenizer, prompts_buffer, device, max_new_tokens=MAX_TOKENS)
                for i, ans in enumerate(answers):
                    answer = ans.strip() if ans else ""
                    gold_entry = golds_buffer[i]
                    rec = {"id": ids_buffer[i], "prediction": answer, "gold": gold_entry}
                    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    processed += 1
                    if processed % 50 == 0 or processed == cap:
                        print(f"Generated/Stored {processed}/{cap}")
                prompts_buffer = []
                ids_buffer = []
                golds_buffer = []

        # flush remaining
        if prompts_buffer:
            answers = generate_batch_prompts_and_answers(model, tokenizer, prompts_buffer, device, max_new_tokens=MAX_TOKENS)
            for i, ans in enumerate(answers):
                answer = ans.strip() if ans else ""
                gold_entry = golds_buffer[i]
                rec = {"id": ids_buffer[i], "prediction": answer, "gold": gold_entry}
                out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1
                if processed % 50 == 0 or processed == cap:
                    print(f"Generated/Stored {processed}/{cap}")

        out_file.close()
        print(f"Inference complete. Predictions saved to {predictions_file}")

    # -------------------------------
    # Load predictions file and run original evaluation logic (unchanged)
    # -------------------------------
    print(f"Loading predictions from {predictions_file}")
    results = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    print(f"Loaded {len(results)} predictions")

    # Extract predictions and gold answers (same logic as original)
    preds = []
    golds = []
    for result in results:
        prediction = result.get('prediction', '')
        gold_list = result.get('gold', [])
        if isinstance(gold_list, list) and len(gold_list) > 0:
            gold = gold_list[0]
        else:
            gold = str(gold_list) if gold_list else ''
        preds.append(prediction)
        golds.append(gold)

    # Calculate metrics (identical math)
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
        strict_f1, _, _ = f1_score(pred, gold)
        strict_em = int(normalize_strict(pred) == normalize_strict(gold))
        bal_f1, bal_prec, bal_rec = balanced_f1(pred, gold)
        bal_em = balanced_em(pred, gold)
        total_strict_em += strict_em
        total_strict_f1 += strict_f1
        total_balanced_em += bal_em
        total_balanced_f1 += bal_f1
        total_balanced_precision += bal_prec
        total_balanced_recall += bal_rec

    num_samples = len(preds)
    if num_samples == 0:
        raise RuntimeError("No predictions to evaluate (num_samples == 0)")

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

    # Semantic (batch) - sample first 50 for speed (unchanged)
    print("\nComputing BERTScore (sampling first 50 for speed)...")
    sample_size = min(50, len(preds))
    sem_results = semantic_scores(preds[:sample_size], golds[:sample_size])
    print("Semantic BERTScore averages:")
    print(f"Precision: {sum(sem_results['precision'])/len(sem_results['precision']):.4f}")
    print(f"Recall:    {sum(sem_results['recall'])/len(sem_results['recall']):.4f}")
    print(f"F1:        {sum(sem_results['f1'])/len(sem_results['f1']):.4f}")

    # Save results (unchanged format)
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

if __name__ == "__main__":
    main()
