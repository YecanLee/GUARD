import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
import json

from generate import generate_text
from helpers.process_data import load_data
from global_entropy import *

import argparse
from tqdm import trange

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--w", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    args = parser.parse_args()
    model_name = args.model_name
    # 1. Load model&tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
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
    #results = []

    pre_text_list , pre_token_id_list ,reference_text_list = load_data(data_path,tokenizer,mode='wikitext')

    # 4. Start Inference
    print("Start Inference!!!!!")

    data_num = len(pre_text_list)
    print(data_num)
    results_list = []
    repetitive_list = []


    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    with torch.inference_mode():
        #with torch.autocast('cuda',enabled=cuda_available,dtype=torch.float16,cache_enabled=True):
        for index in trange(data_num):
            # print(f'Inference {index + 1}/{data_num} ({np.round((index + 1) / data_num * 100, 2)} %)')
            one_pre_text = pre_text_list[index]
            one_reference_text = reference_text_list[index]
            input_ids = tokenizer(one_pre_text,return_tensors='pt').input_ids.to(device)
            if cuda_available:
                input_ids = input_ids.cuda(device)

            one_generated_text, repetitive_rate = generate_text(model, tokenizer, input_ids, w=args.w)

            one_res_list = {
                'prefix_text':one_pre_text,
                'reference_text':one_reference_text,
                'generated_text':{
                    '0':one_generated_text
                }
            }
            results_list.append(one_res_list)
            repetitive_list.append(repetitive_rate)
    print("Inference completed!")

    print(repetitive_list)

    print(f"Average Repetitive Rate {np.mean(repetitive_rate)}")

    save_path_prefix = f"{args.model_name}_result" if os.path.exists(f"{args.model_name}_result") else os.makedirs(f"{args.model_name}_result")

    save_path = f"{save_path_prefix}/inference_result_wsize_{args.w}_dataset_{args.dataset_name}.json" 

    with open(save_path, 'w') as outfile:
        json.dump(results_list, outfile, indent=4)

print("Congratulations!!!")
print("Inference Completed!!!!")
