#!/usr/bin/env python3
"""
Test script to compare accuracy between base Qwen2.5-VL and fine-tuned model
Uses test.csv for evaluation
"""

import os
import csv
import json
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import time

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Qwen2VLForConditionalGeneration,
)
from peft import PeftModel

# ==================== Configuration ====================
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
TEST_CSV = "test_v2.csv"

# Improved prompt from app_v4.py - ensures concise food name output
PROMPT = (
    "너는 한국 음식 이미지 분류기다.\n"
    "이미지에서 '가장 중심이 되는 음식 1개'의 이름만 한국어로 출력해라.\n\n"
    "출력 규칙(매우 중요):\n"
    "1) 한국어 음식명만 출력\n"
    "2) 조사/문장/설명 금지 (예: '입니다', '같아요', '.' 금지)\n"
    "3) 따옴표/괄호/슬래시/이모지 금지\n"
    "4) 공백 없이 음식명만 출력 (최대 12자)\n\n"
    "예시(정답 형식):\n"
    "김밥\n"
    "계란말이\n"
    "떡볶이\n\n"
    "혼동 주의 규칙:\n"
    "- 계란말이: 달걀을 말아 네모/원통 형태, 단면이 말린 층\n"
    "- 오믈렛: 접힌 형태, 둥글고 납작함\n\n"
    "중요: 출력할 음식명이 없다고 판단되더라도 '-', '없음', '?'를 출력하지 말고\n"
    "가장 유사한 한국 음식명 1개를 반드시 출력하라.\n\n"
    "정답(음식명만):"
)



# ==================== Model Loading ====================
def load_base_model():
    """Load base Qwen2.5-VL model"""
    print(f"Loading base model: {BASE_MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16,
    )
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    model.eval()
    print("✓ Base model loaded")
    return model, processor


def load_finetuned_model(adapter_path: str):
    """Load fine-tuned model with LoRA adapter"""
    print(f"Loading fine-tuned model from: {adapter_path}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # Load base model
    base_model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    model.eval()
    print("✓ Fine-tuned model loaded")
    return model, processor


# ==================== Inference ====================
def predict_food(
    model,
    processor,
    image_path: str,
    prompt: str = PROMPT,
    max_new_tokens: int = 50
) -> str:
    """Predict food name from image"""
    
    # Normalize path
    image_path = image_path.replace('\\', '/')
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Warning: Failed to load {image_path}: {e}")
        return ""
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    # Decode
    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    
    # Extract answer - improved parsing
    # The output format is usually: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n[ANSWER]<|im_end|>
    answer = generated_text.strip()
    
    # Remove special tokens
    answer = answer.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    
    # Extract assistant's response
    if "assistant" in answer.lower():
        parts = answer.lower().split("assistant")
        if len(parts) > 1:
            answer = parts[-1].strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "음식 이름:",
        "음식의 이름:",
        "한국 음식:",
        "답:",
        "answer:",
        "정답:",
    ]
    
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            break
    
    # Extract first quoted text if exists
    if '"' in answer or "'" in answer:
        import re
        # Try to find quoted food name
        quoted = re.findall(r'["\']([^"\']+)["\']', answer)
        if quoted:
            answer = quoted[0].strip()
    
    # If answer contains multiple lines or sentences, take first line
    if '\n' in answer:
        answer = answer.split('\n')[0].strip()
    
    # Remove any trailing explanations (everything after period, comma, or parenthesis)
    for delimiter in ['.', '!', '?', '(', ',']:
        if delimiter in answer:
            # But keep if it's part of Korean food name (like "떡_만두국")
            parts = answer.split(delimiter)
            first_part = parts[0].strip()
            # Only use first part if it contains Korean characters
            if any('\uac00' <= c <= '\ud7a3' for c in first_part):
                answer = first_part
                break
    
    # Clean up any remaining special characters at the end
    answer = answer.rstrip('.,!?()[]{}')
    
    return answer.strip()



# ==================== Evaluation ====================
def evaluate_model(
    model,
    processor,
    test_csv_path: str,
    model_name: str = "Model"
) -> Dict:
    """Evaluate model on test set"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Load test data
    test_data = []
    with open(test_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append({
                'image_path': row['image_path'],
                'label': row['label']
            })
    
    print(f"Test samples: {len(test_data)}")
    
    # Evaluation metrics
    correct = 0
    total = 0
    results = []
    
    # Track per-label accuracy
    label_stats = {}
    
    # Evaluate
    for item in tqdm(test_data, desc=f"Testing {model_name}"):
        image_path = item['image_path']
        true_label = item['label'].strip()
        
        # Predict
        predicted = predict_food(model, processor, image_path)
        
        # Check if correct (exact match or contains)
        is_correct_exact = (predicted == true_label)
        is_correct_contains = (true_label in predicted)
        
        # Use contains as primary metric (more lenient)
        is_correct = is_correct_contains
        
        if is_correct:
            correct += 1
        
        total += 1
        
        # Track per-label stats
        if true_label not in label_stats:
            label_stats[true_label] = {'correct': 0, 'total': 0}
        
        label_stats[true_label]['total'] += 1
        if is_correct:
            label_stats[true_label]['correct'] += 1
        
        # Store result
        results.append({
            'image_path': image_path,
            'true_label': true_label,
            'predicted': predicted,
            'correct_exact': is_correct_exact,
            'correct_contains': is_correct_contains,
        })
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Calculate per-label accuracy
    label_accuracies = {}
    for label, stats in label_stats.items():
        label_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        label_accuracies[label] = label_acc
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"  Unique labels tested: {len(label_stats)}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results,
        'label_accuracies': label_accuracies,
        'label_stats': label_stats,
    }


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned Qwen2.5-VL models")
    parser.add_argument(
        "--finetuned_path",
        type=str,
        default="./qwen25_kfood_output/final",
        help="Path to fine-tuned model adapter"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=TEST_CSV,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip base model evaluation"
    )
    parser.add_argument(
        "--skip_finetuned",
        action="store_true",
        help="Skip fine-tuned model evaluation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_summary = {}
    
    # Evaluate base model
    if not args.skip_base:
        base_model, base_processor = load_base_model()
        base_results = evaluate_model(base_model, base_processor, args.test_csv, "Base Model")
        results_summary['base'] = base_results
        
        # Save base results
        with open(os.path.join(args.output_dir, "base_results.json"), 'w', encoding='utf-8') as f:
            json.dump(base_results, f, ensure_ascii=False, indent=2)
        
        # Free memory
        del base_model, base_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    if not args.skip_finetuned and os.path.exists(args.finetuned_path):
        finetuned_model, finetuned_processor = load_finetuned_model(args.finetuned_path)
        finetuned_results = evaluate_model(
            finetuned_model,
            finetuned_processor,
            args.test_csv,
            "Fine-tuned Model"
        )
        results_summary['finetuned'] = finetuned_results
        
        # Save fine-tuned results
        with open(os.path.join(args.output_dir, "finetuned_results.json"), 'w', encoding='utf-8') as f:
            json.dump(finetuned_results, f, ensure_ascii=False, indent=2)
        
        # Free memory
        del finetuned_model, finetuned_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if 'base' in results_summary:
        print(f"Base Model Accuracy: {results_summary['base']['accuracy']:.2f}%")
    
    if 'finetuned' in results_summary:
        print(f"Fine-tuned Model Accuracy: {results_summary['finetuned']['accuracy']:.2f}%")
    
    if 'base' in results_summary and 'finetuned' in results_summary:
        improvement = results_summary['finetuned']['accuracy'] - results_summary['base']['accuracy']
        print(f"\nImprovement: {improvement:+.2f}%")
    
    # Save comparison
    with open(os.path.join(args.output_dir, "comparison.json"), 'w', encoding='utf-8') as f:
        comparison = {
            'base_accuracy': results_summary.get('base', {}).get('accuracy', None),
            'finetuned_accuracy': results_summary.get('finetuned', {}).get('accuracy', None),
            'improvement': improvement if 'base' in results_summary and 'finetuned' in results_summary else None,
            'test_samples': results_summary.get('base', results_summary.get('finetuned', {})).get('total', 0),
        }
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {args.output_dir}/")
    print("  - base_results.json")
    print("  - finetuned_results.json")
    print("  - comparison.json")


if __name__ == "__main__":
    main()
