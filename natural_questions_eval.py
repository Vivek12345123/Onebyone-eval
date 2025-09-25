#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete evaluation script for Self-RAG Llama 2 7B on Natural Questions dataset.
Returns: Balanced F1, Balanced EM, BERT Precision, BERT Recall, and BERT F1.
"""

import json
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm

# Natural Questions imports
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

# BERTScore imports
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from bert_score import score as bert_score_func


class NQEvaluator:
    """Natural Questions evaluation metrics."""
    
    def __init__(self):
        self.LONG_NO_NULL_THRESHOLD = 2
        self.SHORT_NO_NULL_THRESHOLD = 2
    
    def safe_divide(self, x, y):
        """Compute x / y, but return 0 if y is zero."""
        return 0 if y == 0 else x / y
    
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        import re
        import string
        
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
    
    def exact_match_score(self, prediction, ground_truth):
        """Check if prediction exactly matches ground truth after normalization."""
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
    
    def f1_score(self, prediction, ground_truth):
        """Compute token-level F1 score between prediction and ground truth."""
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            return int(prediction_tokens == ground_truth_tokens)
        
        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = sum((1 for token in common))
        
        if num_same == 0:
            return 0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def compute_metrics(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """Compute EM and F1 scores for predictions against references."""
        total_em = 0
        total_f1 = 0
        total_count = 0
        
        for pred, refs in zip(predictions, references):
            # Get best scores across all reference answers
            em_scores = [self.exact_match_score(pred, ref) for ref in refs]
            f1_scores = [self.f1_score(pred, ref) for ref in refs]
            
            total_em += max(em_scores) if em_scores else 0
            total_f1 += max(f1_scores) if f1_scores else 0
            total_count += 1
        
        return {
            'exact_match': self.safe_divide(total_em, total_count),
            'f1': self.safe_divide(total_f1, total_count),
            'total': total_count
        }


class SelfRAGEvaluator:
    """Self-RAG model evaluator for Natural Questions."""
    
    def __init__(self, model_path: str = "selfrag/selfrag_llama2_7b", 
                 download_dir: str = "/gscratch/h2lab/akari/model_cache",
                 use_retrieval: bool = False,
                 max_tokens: int = 100):
        """Initialize Self-RAG model with vLLM."""
        print(f"Loading Self-RAG model from {model_path}...")
        self.model = LLM(model_path, download_dir=download_dir, dtype="half")
        self.sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=1.0, 
            max_tokens=max_tokens, 
            skip_special_tokens=False
        )
        self.use_retrieval = use_retrieval
    
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Format prompt for Self-RAG model."""
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        if context is not None and self.use_retrieval:
            prompt += f"[Retrieval]<paragraph>{context}</paragraph>"
        return prompt
    
    def extract_answer_from_generation(self, generated_text: str) -> str:
        """Extract clean answer from Self-RAG generation."""
        # Remove special tokens and markers
        answer = generated_text
        
        # Remove Self-RAG specific markers
        markers = ['[Retrieval]', '[No Retrieval]', '[Relevant]', '[Irrelevant]', 
                  '[Fully supported]', '[Partially supported]', '[No support]',
                  '[Utility:1]', '[Utility:2]', '[Utility:3]', '[Utility:4]', '[Utility:5]']
        
        for marker in markers:
            answer = answer.replace(marker, '')
        
        # Remove paragraph tags if present
        import re
        answer = re.sub(r'<paragraph>.*?</paragraph>', '', answer)
        
        # Remove </s> token
        answer = answer.replace('</s>', '')
        
        # Clean up whitespace
        answer = ' '.join(answer.split())
        
        return answer.strip()
    
    def generate_answers(self, questions: List[str], contexts: Optional[List[str]] = None,
                        batch_size: int = 8) -> List[str]:
        """Generate answers for a batch of questions."""
        answers = []
        
        # Process in batches
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating answers"):
            batch_questions = questions[i:i+batch_size]
            
            if contexts:
                batch_contexts = contexts[i:i+batch_size]
                prompts = [self.format_prompt(q, c) for q, c in zip(batch_questions, batch_contexts)]
            else:
                prompts = [self.format_prompt(q) for q in batch_questions]
            
            # Generate predictions
            preds = self.model.generate(prompts, self.sampling_params)
            
            # Extract answers
            for pred in preds:
                raw_answer = pred.outputs[0].text
                clean_answer = self.extract_answer_from_generation(raw_answer)
                answers.append(clean_answer)
        
        return answers


def prepare_nq_data(dataset, max_samples: Optional[int] = None) -> Tuple[List[str], List[List[str]], List[Optional[str]]]:
    """Prepare Natural Questions data for evaluation."""
    questions = []
    all_answers = []
    contexts = []
    
    # Limit samples if specified
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    for example in tqdm(samples, desc="Preparing NQ data"):
        # Extract question
        question = example['question']['text']
        questions.append(question)
        
        # Extract all annotated short answers (from multiple annotators)
        short_answers = []
        annotations = example['annotations']
        
        for annotation in annotations['short_answers']:
            if annotation and len(annotation) > 0:
                # Each annotator may provide multiple short answer spans
                for answer_span in annotation:
                    start_idx = answer_span['start_token']
                    end_idx = answer_span['end_token']
                    
                    # Extract answer text from document tokens
                    if start_idx >= 0 and end_idx >= 0:
                        tokens = example['document']['tokens']
                        answer_tokens = []
                        for idx in range(start_idx, min(end_idx, len(tokens['token']))):
                            if tokens['is_html'][idx]:
                                continue
                            answer_tokens.append(tokens['token'][idx])
                        
                        answer_text = ' '.join(answer_tokens).strip()
                        if answer_text:
                            short_answers.append(answer_text)
        
        # Also check for yes/no answers
        yes_no_answers = [ans for ans in annotations['yes_no_answer'] if ans and ans != 'NONE']
        short_answers.extend(yes_no_answers)
        
        # If no short answers found, use long answers as fallback
        if not short_answers:
            for annotation_idx, long_answer in enumerate(annotations['long_answer']):
                if long_answer and long_answer['start_token'] >= 0:
                    start_idx = long_answer['start_token']
                    end_idx = long_answer['end_token']
                    
                    tokens = example['document']['tokens']
                    answer_tokens = []
                    for idx in range(start_idx, min(end_idx, len(tokens['token']))):
                        if tokens['is_html'][idx]:
                            continue
                        answer_tokens.append(tokens['token'][idx])
                    
                    answer_text = ' '.join(answer_tokens[:50]).strip()  # Limit long answers
                    if answer_text:
                        short_answers.append(answer_text)
        
        # Use a default if still no answers
        if not short_answers:
            short_answers = ["No answer found"]
        
        all_answers.append(short_answers)
        
        # Extract context (simplified - using first part of document)
        tokens = example['document']['tokens']
        context_tokens = []
        for idx in range(min(500, len(tokens['token']))):  # Use first 500 tokens as context
            if not tokens['is_html'][idx]:
                context_tokens.append(tokens['token'][idx])
        
        context = ' '.join(context_tokens).strip()
        contexts.append(context if context else None)
    
    return questions, all_answers, contexts


def main():
    parser = argparse.ArgumentParser(description='Evaluate Self-RAG on Natural Questions')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for generation')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Maximum number of tokens to generate (default: 512)')
    parser.add_argument('--use_retrieval', action='store_true',
                       help='Use retrieval augmentation')
    parser.add_argument('--output_file', type=str, default='selfrag_nq_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Self-RAG Natural Questions Evaluation")
    print("=" * 80)
    
    # Load Natural Questions dataset
    print("\n1. Loading Natural Questions dataset...")
    dataset = load_dataset("google-research-datasets/natural_questions", "default", split="validation")
    print(f"Loaded {len(dataset)} validation examples")
    
    # Prepare data
    print("\n2. Preparing data...")
    questions, reference_answers, contexts = prepare_nq_data(dataset, args.max_samples)
    print(f"Prepared {len(questions)} questions for evaluation")
    
    # Initialize Self-RAG model
    print("\n3. Initializing Self-RAG model...")
    evaluator = SelfRAGEvaluator(use_retrieval=args.use_retrieval, max_tokens=args.max_tokens)
    
    # Generate predictions
    print("\n4. Generating predictions...")
    start_time = time.time()
    predictions = evaluator.generate_answers(
        questions, 
        contexts if args.use_retrieval else None,
        batch_size=args.batch_size
    )
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Compute NQ metrics
    print("\n5. Computing NQ metrics...")
    nq_evaluator = NQEvaluator()
    nq_metrics = nq_evaluator.compute_metrics(predictions, reference_answers)
    
    # Compute BERTScore metrics
    print("\n6. Computing BERTScore metrics...")
    # Flatten references for BERTScore (use best matching reference)
    flat_references = [refs[0] if refs else "" for refs in reference_answers]
    
    # Calculate BERTScore
    P, R, F1 = bert_score_func(
        predictions, 
        flat_references, 
        lang='en',
        model_type='roberta-large',
        verbose=True
    )
    
    bert_metrics = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }
    
    # Compile all results
    results = {
        'dataset': 'Natural Questions',
        'model': 'Self-RAG Llama 2 7B',
        'num_examples': len(predictions),
        'metrics': {
            'balanced_exact_match': nq_metrics['exact_match'],
            'balanced_f1': nq_metrics['f1'],
            'bert_precision': bert_metrics['precision'],
            'bert_recall': bert_metrics['recall'],
            'bert_f1': bert_metrics['f1']
        },
        'generation_time_seconds': generation_time,
        'use_retrieval': args.use_retrieval
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Dataset: Natural Questions (Validation)")
    print(f"Model: Self-RAG Llama 2 7B")
    print(f"Number of examples: {results['num_examples']}")
    print(f"Use retrieval: {args.use_retrieval}")
    print("\n" + "-" * 40)
    print("Metrics:")
    print("-" * 40)
    print(f"Balanced Exact Match: {results['metrics']['balanced_exact_match']:.4f}")
    print(f"Balanced F1:          {results['metrics']['balanced_f1']:.4f}")
    print(f"BERT Precision:       {results['metrics']['bert_precision']:.4f}")
    print(f"BERT Recall:          {results['metrics']['bert_recall']:.4f}")
    print(f"BERT F1:              {results['metrics']['bert_f1']:.4f}")
    print("=" * 80)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")
    
    return results


if __name__ == "__main__":
    results = main()
