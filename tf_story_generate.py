import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from guard_logits_processor import EntropyBasedLogitsProcessor

import numpy as np
import json

from helpers.process_data import load_data

import argparse
from tqdm import trange

import os

def cal_repetitive(text, tokenizer):
    """Calculate repetition rate of generated text"""
    generated_tokens = tokenizer.encode(text)
    
    if not isinstance(generated_tokens, torch.Tensor):
        generated_tokens = torch.tensor(generated_tokens)
    
    total_tokens = len(generated_tokens)
    unique_tokens = len(torch.unique(generated_tokens))
    
    repetition_rate = float((total_tokens - unique_tokens) / total_tokens)
    
    return repetition_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--w", type=int, default=7)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    args = parser.parse_args()
    
    # 1. Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    
    if torch.cuda.is_available():
        print("cuda is available")
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda_available else 'cpu')
    model = model.to(device)
    print(f"model is on {device}")
    
    # 2. Set model to eval mode
    model.eval()

    # 3. Load data
    data_path = f"data/{args.dataset_name}_contrastive_gpt2-xl_256.jsonl"  # data path
    pre_text_list, pre_token_id_list, reference_text_list = load_data(data_path, tokenizer, mode='wikitext')

    # 4. Start Inference
    print("Starting Inference ⏰⏰⏰...")

    data_num = len(pre_text_list)
    print(f"Total data number for inference: {data_num}")
    results_list = []
    repetitive_list = []

    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    with torch.inference_mode():
        for index in trange(data_num):
            one_pre_text = pre_text_list[index]
            one_reference_text = reference_text_list[index]
            
            # Define GUARD logits processor
            entropy_processor = EntropyBasedLogitsProcessor(w=args.w)
            logits_processor = LogitsProcessorList([entropy_processor])
            
            # Tokenize input
            inputs = tokenizer(one_pre_text, return_tensors="pt").to(device)
            
            # Generate text by using `model.generate` function
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=256, 
                logits_processor=logits_processor, # Use GUARD logits processor
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, # Must be defined as True for GUARD algorithm
                temperature=1.0,
            )
            
            # Decode the generated text
            one_generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True)
            
            # Calculate repetition rate
            repetitive_rate = cal_repetitive(one_generated_text, tokenizer)
            
            one_res_list = {
                'prefix_text': one_pre_text,
                'reference_text': one_reference_text,
                'generated_text': {
                    '0': one_generated_text
                }
            }
            results_list.append(one_res_list)
            repetitive_list.append(repetitive_rate)
    
    print("Inference completed! 🎉🎉🎉")
    print(f"Average Repetitive Rate: {np.mean(repetitive_list)}")

    # Create directory if it doesn't exist
    save_path_prefix = f"{args.model_name}_result"
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_path = f"{save_path_prefix}/inference_result_wsize_{args.w}_dataset_{args.dataset_name}.json" 

    with open(save_path, 'w') as outfile:
        json.dump(results_list, outfile, indent=4)

    print(f"Inference results saved to {save_path} 🔥🔥🔥")
