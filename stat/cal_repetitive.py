import torch
from transformers import AutoTokenizer

import json
import torch
from typing import Dict, Any

def calculate_repetition_rate(text: str, tokenizer: Any, minimal_repetition: int = 2) -> float:
    """Calculate repetition rate for a given text while ignoring padding tokens.
    
    Args:
        text: Input text to analyze
        tokenizer: Tokenizer instance to use for encoding
        
    Returns:
        float: Repetition rate between 0 and 1
    """
    
    common_words = {
    "the", "a", "an", "he", "she", "they", "is", "are", "was", "were",
    "in", "on", "at", "to", "of", "for", "with", "by", "from", "and",
    "or", "but", "so", "that", "this", "these", "those", "it", "its"}

    # Encode the text
    tokens = tokenizer.encode(text)
    
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, device='cuda')
        
    # Get the padding token id
    pad_token_id = tokenizer.pad_token_id

    # mask out the common words during the calculation
    common_words_ids = set()
    for word in common_words:
        words_ids = tokenizer.encode(word, add_special_tokens=False)
        common_words_ids.update(words_ids)

    # create a mask 
    mask = torch.ones_like(tokens, dtype=torch.bool)

    for token_id in common_words_ids:
        mask &= (tokens != token_id)
    mask &= (tokens != pad_token_id)

    # Mask out the common words and padding tokens
    tokens = tokens[mask]

    # Handle empty text case, in case no token besides special tokens were generated
    if len(tokens) == 0:
        return 0.0
    
    # Count occurrences of each token
    token_counts = torch.bincount(tokens)
    
    # Calculate metrics
    repeated_tokens = torch.sum(token_counts >= minimal_repetition)
    total_tokens = len(tokens)
    unique_tokens = len(torch.unique(tokens))
    
    # Calculate repetition rate
    repetition_rate = float(repeated_tokens / total_tokens)
    
    return repetition_rate

def process_json_file(json_path: str, tokenizer: Any) -> Dict[str, Any]:
    """Process JSON file and calculate repetition rates for generated text.
    
    Args:
        json_path: Path to JSON file
        tokenizer: Tokenizer instance
        
    Returns:
        Dict containing rates, averages, and repetition counts:
        {
            'samples': Dict[int, float],     # Individual sample rates
            'average': float,                # Average rate across all samples
            'sentences_with_repetitions': int,# Count of sentences with repetitions
            'total_sentences': int,          # Total number of sentences analyzed
            'repetition_ratio': float        # Ratio of sentences with repetitions
        }
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {
        'samples': {},
        'average': 0.0,
        'sentences_with_repetitions': 0,
        'total_sentences': 0,
        'repetition_ratio': 0.0
    }
    total_rate = 0.0
    sample_count = 0

    for idx, sample in enumerate(data):
        if 'generated_result' in sample and '0' in sample['generated_result']:
            text = sample['generated_result']['0']
            
            # Calculate repetition rate
            rep_rate = calculate_repetition_rate(text, tokenizer)
            results['samples'][idx] = rep_rate
            
            total_rate += rep_rate
            sample_count += 1
    
    # We report average repetition rate in our paper
    if sample_count > 0:
        results['average'] = total_rate / sample_count
            
    return results


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    results = process_json_file(args.file_path, tokenizer)

    # print out the repetition rate
    print(f"\nAverage repetition rate: {results['average']:.4f}")


