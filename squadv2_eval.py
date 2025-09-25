#!/usr/bin/env python3
"""
Self-RAG Evaluation Script for SQuAD v2
Evaluates Self-RAG model on SQuAD v2 with BertScore and Balanced EM/F1 metrics
"""

import json
import string
import re
import argparse
import collections
from collections import Counter
from typing import List, Dict, Tuple, Optional
import sys

import torch
from vllm import LLM, SamplingParams
from bert_score import BERTScorer
from tqdm import tqdm
import numpy as np
from datasets import load_dataset


class SelfRAGSquadEvaluator:
    """Evaluator for Self-RAG model on SQuAD v2 dataset"""
    
    def __init__(self, model_path: str = "selfrag/selfrag_llama2_7b", 
                 download_dir: str = "/gscratch/h2lab/akari/model_cache",
                 device: str = "cuda", max_tokens: int = 100, batch_size: int = 1):
        """
        Initialize Self-RAG evaluator
        
        Args:
            model_path: Path to Self-RAG model
            download_dir: Directory for model cache
            device: Device for BertScore computation
            max_tokens: Maximum tokens for generation
            batch_size: Batch size for inference
        """
        print("Loading Self-RAG model...")
        self.model = LLM(model_path, download_dir=download_dir, dtype="half")
        self.sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=1.0, 
            max_tokens=max_tokens, 
            skip_special_tokens=False
        )
        self.batch_size = batch_size
        
        print("Initializing BertScore...")
        self.bert_scorer = BERTScorer(
            model_type='microsoft/deberta-xlarge-mnli',
            lang='en',
            rescale_with_baseline=True,
            device=device
        )
        
        # Special tokens that Self-RAG uses
        self.special_tokens_pattern = re.compile(
            r'\[(?:Retrieval|No Retrieval|Relevant|Irrelevant|'
            r'Fully supported|Partially supported|No support|Utility:\d+)\]'
        )
    
    def format_prompt(self, question: str, context: str = None) -> str:
        """
        Format prompt for Self-RAG model
        
        Args:
            question: The question to answer
            context: Optional context paragraph
        
        Returns:
            Formatted prompt string
        """
        instruction = f"Answer the following question based on the given context. " \
                     f"If the question cannot be answered based on the context, say 'unanswerable'.\n\n" \
                     f"Question: {question}"
        
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        if context:
            # Add context as a retrieved paragraph for Self-RAG
            prompt += f"[Retrieval]<paragraph>{context}</paragraph>"
        
        return prompt
    
    def clean_prediction(self, text: str) -> str:
        """
        Clean Self-RAG prediction by removing special tokens
        
        Args:
            text: Raw prediction from Self-RAG
        
        Returns:
            Cleaned text
        """
        # Remove special tokens
        cleaned = self.special_tokens_pattern.sub('', text)
        
        # Remove </s> token
        cleaned = cleaned.replace('</s>', '')
        
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def normalize_answer(self, s: str) -> str:
        """Normalize answer for evaluation"""
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
    
    def get_tokens(self, s: str) -> List[str]:
        """Get normalized tokens from string"""
        if not s:
            return []
        return self.normalize_answer(s).split()
    
    def compute_exact(self, a_gold: str, a_pred: str) -> int:
        """Compute exact match"""
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))
    
    def compute_f1(self, a_gold: str, a_pred: str) -> float:
        """Compute F1 score"""
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def evaluate_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """
        Evaluate a batch of samples
        
        Args:
            batch_data: List of dictionaries containing question, context, answers, etc.
        
        Returns:
            List of evaluation results
        """
        # Generate prompts for the batch
        prompts = []
        for sample in batch_data:
            prompt = self.format_prompt(sample['question'], sample['context'])
            prompts.append(prompt)
        
        # Generate predictions for the batch
        pred_outputs = self.model.generate(prompts, self.sampling_params)
        
        # Process each sample in the batch
        batch_results = []
        for i, sample in enumerate(batch_data):
            raw_prediction = pred_outputs[i].outputs[0].text
            prediction = self.clean_prediction(raw_prediction)
            
            # Check if model predicted unanswerable
            pred_is_impossible = 'unanswerable' in prediction.lower()
            
            # Compute exact match and F1
            if sample['is_impossible']:
                # For unanswerable questions
                if pred_is_impossible:
                    exact_match = 1
                    f1 = 1.0
                else:
                    exact_match = 0
                    f1 = 0.0
            else:
                # For answerable questions
                if pred_is_impossible:
                    exact_match = 0
                    f1 = 0.0
                else:
                    # Compare against all gold answers and take max
                    exact_scores = [self.compute_exact(ans, prediction) for ans in sample['gold_answers']]
                    f1_scores = [self.compute_f1(ans, prediction) for ans in sample['gold_answers']]
                    exact_match = max(exact_scores) if exact_scores else 0
                    f1 = max(f1_scores) if f1_scores else 0.0
            
            result = {
                'prediction': prediction,
                'raw_prediction': raw_prediction,
                'exact_match': exact_match,
                'f1': f1,
                'is_impossible': sample['is_impossible'],
                'pred_is_impossible': pred_is_impossible,
                'qid': sample['qid']
            }
            batch_results.append(result)
        
        return batch_results
    
    def evaluate_sample(self, question: str, context: str, 
                       gold_answers: List[str], is_impossible: bool) -> Dict:
        """
        Evaluate a single sample
        
        Args:
            question: Question text
            context: Context paragraph
            gold_answers: List of gold answer texts
            is_impossible: Whether question is unanswerable
        
        Returns:
            Dictionary with prediction and scores
        """
        # Generate prompt and get prediction
        prompt = self.format_prompt(question, context)
        pred_output = self.model.generate([prompt], self.sampling_params)[0]
        raw_prediction = pred_output.outputs[0].text
        
        # Clean prediction
        prediction = self.clean_prediction(raw_prediction)
        
        # Check if model predicted unanswerable
        pred_is_impossible = 'unanswerable' in prediction.lower()
        
        # Compute exact match and F1
        if is_impossible:
            # For unanswerable questions
            if pred_is_impossible:
                exact_match = 1
                f1 = 1.0
            else:
                exact_match = 0
                f1 = 0.0
        else:
            # For answerable questions
            if pred_is_impossible:
                exact_match = 0
                f1 = 0.0
            else:
                # Compare against all gold answers and take max
                exact_scores = [self.compute_exact(ans, prediction) for ans in gold_answers]
                f1_scores = [self.compute_f1(ans, prediction) for ans in gold_answers]
                exact_match = max(exact_scores) if exact_scores else 0
                f1 = max(f1_scores) if f1_scores else 0.0
        
        return {
            'prediction': prediction,
            'raw_prediction': raw_prediction,
            'exact_match': exact_match,
            'f1': f1,
            'is_impossible': is_impossible,
            'pred_is_impossible': pred_is_impossible
        }
    
    def compute_bert_scores(self, predictions: List[str], 
                          references: List[List[str]]) -> Tuple[float, float, float]:
        """
        Compute BertScore metrics
        
        Args:
            predictions: List of predictions
            references: List of reference answer lists
        
        Returns:
            Tuple of (precision, recall, f1) averages
        """
        # For multiple references, compute against each and take max
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for pred, ref_list in zip(predictions, references):
            if not ref_list:  # For unanswerable questions
                ref_list = ['unanswerable']
            
            # Compute scores against each reference
            P_list, R_list, F1_list = [], [], []
            for ref in ref_list:
                P, R, F1 = self.bert_scorer.score([pred], [ref])
                P_list.append(P.item())
                R_list.append(R.item())
                F1_list.append(F1.item())
            
            # Take max across references
            all_precisions.append(max(P_list))
            all_recalls.append(max(R_list))
            all_f1s.append(max(F1_list))
        
        return np.mean(all_precisions), np.mean(all_recalls), np.mean(all_f1s)
    
    def compute_balanced_metrics(self, results: List[Dict]) -> Tuple[float, float]:
        """
        Compute balanced EM and F1 scores
        
        Args:
            results: List of evaluation results
        
        Returns:
            Tuple of (balanced_em, balanced_f1)
        """
        # Separate answerable and unanswerable
        answerable_results = [r for r in results if not r['is_impossible']]
        unanswerable_results = [r for r in results if r['is_impossible']]
        
        # Compute metrics for each group
        if answerable_results:
            ans_em = np.mean([r['exact_match'] for r in answerable_results])
            ans_f1 = np.mean([r['f1'] for r in answerable_results])
        else:
            ans_em = ans_f1 = 0.0
        
        if unanswerable_results:
            unans_em = np.mean([r['exact_match'] for r in unanswerable_results])
            unans_f1 = np.mean([r['f1'] for r in unanswerable_results])
        else:
            unans_em = unans_f1 = 0.0
        
        # Compute balanced metrics (average of answerable and unanswerable)
        balanced_em = (ans_em + unans_em) / 2
        balanced_f1 = (ans_f1 + unans_f1) / 2
        
        return balanced_em, balanced_f1
    
    def evaluate(self, limit: Optional[int] = None) -> Dict:
        """
        Evaluate Self-RAG on SQuAD v2 dataset
        
        Args:
            limit: Optional limit on number of samples to evaluate
        
        Returns:
            Dictionary with all evaluation metrics
        """
        # Load dataset using HuggingFace datasets
        print("Loading SQuAD v2 dataset...")
        ds = load_dataset("rajpurkar/squad_v2")
        validation_data = ds['validation']
        
        # Process samples with batching
        all_results = []
        all_predictions = []
        all_references = []
        
        # Prepare batch data
        batch_data = []
        total_samples = 0
        
        for sample in tqdm(validation_data, desc="Processing samples"):
            if limit and total_samples >= limit:
                break
                
            # Extract data from HuggingFace dataset format
            question = sample['question']
            context = sample['context']
            qid = sample['id']
            is_impossible = len(sample['answers']['text']) == 0
            
            # Get gold answers
            if is_impossible:
                gold_answers = []
            else:
                gold_answers = sample['answers']['text']
            
            # Add to batch
            batch_item = {
                'question': question,
                'context': context,
                'qid': qid,
                'is_impossible': is_impossible,
                'gold_answers': gold_answers
            }
            batch_data.append(batch_item)
            
            # Process batch when it's full
            if len(batch_data) == self.batch_size:
                batch_results = self.evaluate_batch(batch_data)
                
                for result in batch_results:
                    all_results.append(result)
                    all_predictions.append(result['prediction'])
                    all_references.append(batch_data[len(all_results)-1]['gold_answers'] 
                                        if batch_data[len(all_results)-1]['gold_answers'] 
                                        else ['unanswerable'])
                
                batch_data = []
            
            total_samples += 1
        
        # Process remaining samples in the last batch
        if batch_data:
            batch_results = self.evaluate_batch(batch_data)
            
            for i, result in enumerate(batch_results):
                all_results.append(result)
                all_predictions.append(result['prediction'])
                all_references.append(batch_data[i]['gold_answers'] 
                                    if batch_data[i]['gold_answers'] 
                                    else ['unanswerable'])
        
        print(f"\nEvaluated {len(all_results)} samples")
        
        # Compute overall metrics
        overall_em = np.mean([r['exact_match'] for r in all_results])
        overall_f1 = np.mean([r['f1'] for r in all_results])
        
        # Compute balanced metrics
        balanced_em, balanced_f1 = self.compute_balanced_metrics(all_results)
        
        # Compute BertScore
        print("\nComputing BertScore metrics...")
        bert_precision, bert_recall, bert_f1 = self.compute_bert_scores(
            all_predictions, all_references
        )
        
        # Compile final metrics
        metrics = {
            'overall': {
                'exact_match': overall_em * 100,
                'f1': overall_f1 * 100,
                'total': len(all_results)
            },
            'balanced': {
                'exact_match': balanced_em * 100,
                'f1': balanced_f1 * 100
            },
            'bert_score': {
                'precision': bert_precision,
                'recall': bert_recall,
                'f1': bert_f1
            },
            'breakdown': {
                'answerable': {
                    'total': sum(1 for r in all_results if not r['is_impossible']),
                    'exact_match': np.mean([r['exact_match'] for r in all_results 
                                           if not r['is_impossible']]) * 100,
                    'f1': np.mean([r['f1'] for r in all_results 
                                 if not r['is_impossible']]) * 100
                },
                'unanswerable': {
                    'total': sum(1 for r in all_results if r['is_impossible']),
                    'exact_match': np.mean([r['exact_match'] for r in all_results 
                                           if r['is_impossible']]) * 100,
                    'f1': np.mean([r['f1'] for r in all_results 
                                 if r['is_impossible']]) * 100
                }
            }
        }
        
        return metrics, all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Self-RAG on SQuAD v2')
    parser.add_argument('--model-path', default='selfrag/selfrag_llama2_7b',
                       help='Path to Self-RAG model')
    parser.add_argument('--download-dir', default='/gscratch/h2lab/akari/model_cache',
                       help='Directory for model cache')
    parser.add_argument('--output', default='selfrag_squad_results.json',
                       help='Output file for results')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples to evaluate')
    parser.add_argument('--device', default='cuda',
                       help='Device for BertScore computation')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens for generation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SelfRAGSquadEvaluator(
        model_path=args.model_path,
        download_dir=args.download_dir,
        device=args.device,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    metrics, results = evaluator.evaluate(limit=args.limit)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\n📊 Overall Metrics:")
    print(f"  Exact Match: {metrics['overall']['exact_match']:.2f}%")
    print(f"  F1 Score: {metrics['overall']['f1']:.2f}%")
    print(f"  Total Samples: {metrics['overall']['total']}")
    
    print("\n⚖️  Balanced Metrics:")
    print(f"  Balanced EM: {metrics['balanced']['exact_match']:.2f}%")
    print(f"  Balanced F1: {metrics['balanced']['f1']:.2f}%")
    
    print("\n🎯 BertScore Metrics:")
    print(f"  Precision: {metrics['bert_score']['precision']:.4f}")
    print(f"  Recall: {metrics['bert_score']['recall']:.4f}")
    print(f"  F1: {metrics['bert_score']['f1']:.4f}")
    
    print("\n📋 Breakdown by Question Type:")
    print(f"\nAnswerable Questions ({metrics['breakdown']['answerable']['total']} samples):")
    print(f"  EM: {metrics['breakdown']['answerable']['exact_match']:.2f}%")
    print(f"  F1: {metrics['breakdown']['answerable']['f1']:.2f}%")
    
    print(f"\nUnanswerable Questions ({metrics['breakdown']['unanswerable']['total']} samples):")
    print(f"  EM: {metrics['breakdown']['unanswerable']['exact_match']:.2f}%")
    print(f"  F1: {metrics['breakdown']['unanswerable']['f1']:.2f}%")
    
    # Save results
    output_data = {
        'metrics': metrics,
        'predictions': {r['qid']: r['prediction'] for r in results}
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")


if __name__ == '__main__':
    main()
