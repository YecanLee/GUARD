import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from guard_logits_processor import EntropyBasedLogitsProcessor
from transformers import LogitsProcessorList

def generate_with_entropy_processor(model_name, prompt, w=4, max_new_tokens=256):
    """
    Generate text using the entropy-based logits processor
    Args:
        model_name (str): The name of the model to use
        prompt (str): The prompt to generate text from
        w (int): The window size parameter
        max_new_tokens (int): The maximum number of new tokens to generate
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Define GUARD logits processor
    entropy_processor = EntropyBasedLogitsProcessor(w=w)
    logits_processor = LogitsProcessorList([entropy_processor])
    
    # Generate text by using `model.generate` function
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, # Must be defined as True for GUARD algorithm
        temperature=1.0,
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there is a city called Suzhou.")
    parser.add_argument("--w", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    
    generated_text = generate_with_entropy_processor(args.model_name, args.prompt, args.w, args.max_new_tokens)
    print(f"Prompt: {args.prompt}")
    print(f"Generated text: {generated_text}")