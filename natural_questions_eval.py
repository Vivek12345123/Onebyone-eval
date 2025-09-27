#!/usr/bin/env python3
"""
Enhanced Natural Questions evaluation with BERTScore and balanced (forgiving) EM/F1.
This file preserves the original evaluator's processes and calculations and adds
balanced_em and balanced_f1 computed alongside the traditional metrics.
"""
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score as bertscore
import time
import re
from pathlib import Path
from difflib import SequenceMatcher
from datasets import load_dataset
import json

ds = load_dataset("google-research-datasets/natural_questions", "default", split="validation[:2000]")

# Convert to your script's expected format
gold = []
for i, item in enumerate(ds):
    gold.append({
        "id": str(i),
        "question": item["question"]["text"],
        "answers": [a["text"] for a in item["annotations"]["short_answers"]]
    })

with open("nq_dev.json", "w") as f:
    json.dump(gold, f, indent=2)


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

def load_selfrag_model():
    """Load the SelfRAG model from Hugging Face."""
    model_id = "selfrag/selfrag_llama2_7b"
    print(f"Loading model from Hugging Face id: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16,        # instead of torch_dtype
    low_cpu_mem_usage=True
)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question, max_new_tokens=512):
    """Generate response using the model (single-question helper kept for backwards compat)."""
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

def normalize_answer(s):
    """Normalize answer for F1/EM calculation."""
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

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    """Calculate exact match score."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

# -------------------------
# Balanced / forgiving metrics
# -------------------------
def normalize_balanced(s):
    """
    Balanced forgiving normalization:
    - start from normalize_answer
    - apply light heuristics (simple suffix stripping and small mapping)
    This is intentionally lightweight and forgiving.
    """
    if s is None:
        s = ""
    s = normalize_answer(s)
    tokens = []
    for t in s.split():
        # simple suffix stripping heuristics
        if t.endswith("ing") and len(t) > 4:
            t = t[:-3]
        elif t.endswith("ed") and len(t) > 3:
            t = t[:-2]
        elif t.endswith("ly") and len(t) > 3:
            t = t[:-2]
        elif t.endswith("s") and len(t) > 3:
            t = t[:-1]
        # small domain mapping example (preserve meaning)
        if t in {"armenian", "armenia"}:
            t = "armenia"
        tokens.append(t)
    return " ".join(tokens)

def numeric_overlap(pred, gold):
    """Return True if there's at least one shared numeric token between pred and gold."""
    pred_nums = set(re.findall(r"\d+", pred))
    gold_nums = set(re.findall(r"\d+", gold))
    return len(pred_nums & gold_nums) > 0

def token_similarity(a, b):
    """Return a token-level similarity in [0,1] using SequenceMatcher."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def balanced_f1(prediction, ground_truth):
    """
    Compute forgiving/fuzzy F1:
      - If numeric overlap -> perfect score
      - Compare token-level similarity between normalized_balanced tokens
      - Compute precision & recall as average best-token-similarity (mirrors symmetric style)
    Returns (f1, precision, recall)
    """
    pred_norm = normalize_balanced(prediction)
    gold_norm = normalize_balanced(ground_truth)

    pred_tokens = [t for t in pred_norm.split() if t]
    gold_tokens = [t for t in gold_norm.split() if t]

    # Mirror official f1 behavior: if no gold tokens, yield 1.0 if no pred tokens else 0.0
    if not gold_tokens:
        if not pred_tokens:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    # numeric overlap shortcut
    if numeric_overlap(pred_norm, gold_norm):
        return 1.0, 1.0, 1.0

    # recall: for each gold token, best matching pred token
    total_rec = 0.0
    for gt in gold_tokens:
        best = 0.0
        for pt in pred_tokens:
            sim = token_similarity(pt, gt)
            if sim > best:
                best = sim
        total_rec += best
    recall = total_rec / len(gold_tokens) if gold_tokens else 0.0

    # precision: for each pred token, best matching gold token
    total_prec = 0.0
    for pt in pred_tokens:
        best = 0.0
        for gt in gold_tokens:
            sim = token_similarity(gt, pt)
            if sim > best:
                best = sim
        total_prec += best
    precision = total_prec / len(pred_tokens) if pred_tokens else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

def balanced_em(prediction, ground_truth):
    """
    Forgiving EM:
      - If numeric overlap -> 1
      - Else compare normalized_balanced strings exactly
    """
    pred_norm = normalize_balanced(prediction)
    gold_norm = normalize_balanced(ground_truth)
    if numeric_overlap(pred_norm, gold_norm):
        return 1
    return int(pred_norm == gold_norm)

def evaluate_nq_with_bertscore(predictions_file, gold_file, model_path=None, generate_predictions=False,
                               num_samples=100, max_new_tokens=512, batch_size=4):
    """Evaluate Natural Questions with BERTScore and both traditional and balanced metrics."""
    # Load or generate predictions
    if generate_predictions:
        print("Generating predictions with CREAMRAG model...")
        model, tokenizer = load_selfrag_model()

        # Load questions from gold file
        with open(gold_file, 'r') as f:
            gold_data = json.load(f)

        # Determine total to generate
        total_available = len(gold_data)
        if num_samples is None or num_samples <= 0:
            total_to_gen = total_available
        else:
            total_to_gen = min(num_samples, total_available)

        predictions = {}

        # Batched generation loop
        for start in range(0, total_to_gen, batch_size):
            batch = gold_data[start: start + batch_size]
            prompts = []
            ids = []
            for item in batch:
                q = item.get('question', '')
                prompts.append(f"Question: {q}\nAnswer:")
                ids.append(item.get('id', str(start)))

            print(f"Generating batch {start//batch_size + 1}: items {start+1}-{start+len(batch)}")
            # Tokenize batch
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
                print(f"Error during generation for batch starting at index {start}: {e}")
                # on failure, fill batch ids with empty strings and continue
                for bid in ids:
                    predictions[bid] = ""
                continue

            # decode each output and cut off the prompt by using input_lengths
            for j, out in enumerate(outputs):
                inp_len = int(input_lengths[j])
                if out.shape[0] <= inp_len:
                    gen_text = ""
                else:
                    gen_tokens = out[inp_len:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                qid = ids[j]
                core_answer = extract_core_answer(gen_text, batch[j].get('question', ''))
                predictions[qid] = core_answer
                print(f"  -> id={qid} generated {len(gen_text)} chars")

        # Save predictions
        Path(predictions_file).parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions to {predictions_file}")

    # Load predictions and gold data
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    with open(gold_file, 'r') as f:
        gold_data = json.load(f)

    # Align predictions and gold answers
    pred_texts, gold_texts, ids = [], [], []
    for item in gold_data:
        qid = item['id']
        if qid in predictions:
            # keep original behavior but ensure we have strings
            pred_val = predictions[qid]
            if pred_val is None:
                pred_val = ""
            pred_texts.append(str(pred_val))
            # Handle multiple possible answers
            answers = item.get('answers', [])
            if answers:
                gold_val = answers[0]
                if gold_val is None:
                    gold_val = ""
                gold_texts.append(str(gold_val))
            else:
                gold_texts.append("")
            ids.append(qid)

    print(f"Evaluating {len(pred_texts)} question-answer pairs...")

    # Traditional metrics (as original)
    total_f1 = 0.0
    total_em = 0.0
    for pred, gold in zip(pred_texts, gold_texts):
        total_f1 += f1_score(pred, gold)
        # original exact_match_score returns bool; sum of bools works as intended
        total_em += exact_match_score(pred, gold)

    avg_f1 = total_f1 / len(pred_texts) if pred_texts else 0.0
    avg_em = total_em / len(pred_texts) if pred_texts else 0.0

    # Balanced metrics accumulators
    total_bal_f1 = 0.0
    total_bal_em = 0.0
    total_bal_prec = 0.0
    total_bal_rec = 0.0

    for pred, gold in zip(pred_texts, gold_texts):
        bf1, bprec, brec = balanced_f1(pred, gold)
        bem = balanced_em(pred, gold)
        total_bal_f1 += bf1
        total_bal_prec += bprec
        total_bal_rec += brec
        total_bal_em += bem

    n = len(pred_texts) if pred_texts else 0
    avg_bal_f1 = total_bal_f1 / n if n else 0.0
    avg_bal_em = total_bal_em / n if n else 0.0
    avg_bal_prec = total_bal_prec / n if n else 0.0
    avg_bal_rec = total_bal_rec / n if n else 0.0

    # Calculate BERTScore
    print("Computing BERTScore...")
    P, R, F1 = bertscore(
        pred_texts,
        gold_texts,
        model_type='roberta-large',
        lang='en',
        rescale_with_baseline=False,
        verbose=True
    )

    # Compile results
    results = {
        'num_samples': len(pred_texts),
        'traditional_f1': avg_f1,
        'traditional_em': avg_em,
        'balanced_f1': avg_bal_f1,
        'balanced_em': avg_bal_em,
        'balanced_precision': avg_bal_prec,
        'balanced_recall': avg_bal_rec,
        'bertscore_precision': float(P.mean()) if len(pred_texts) else 0.0,
        'bertscore_recall': float(R.mean()) if len(pred_texts) else 0.0,
        'bertscore_f1': float(F1.mean()) if len(pred_texts) else 0.0,
        'bertscore_model': 'roberta-large',
        'bertscore_rescale_with_baseline': False,
    }

    return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced NQ evaluation with BERTScore and balanced metrics')
    parser.add_argument('--predictions_file', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--gold_file', type=str, required=True,
                       help='Path to gold data JSON file')
    parser.add_argument('--generate_predictions', action='store_true',
                       help='Generate predictions using CREAMRAG model')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save results JSON file')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate (default 100)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='Max new tokens to generate per answer (default 512)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for generation (default 4)')
    args = parser.parse_args()
    results = evaluate_nq_with_bertscore(
        args.predictions_file,
        args.gold_file,
        generate_predictions=args.generate_predictions,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
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
    print("="*60)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
