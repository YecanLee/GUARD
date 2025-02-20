import torch
from global_entropy import *

from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Llama-3.1-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda') 


def generate_text(model, tokenizer, inputs_ids, max_length=256, temperature=1.0, w=4):
    """Generate text and return the generated text and global entropy"""
    _, prefix_len = inputs_ids.size()
    inputs = inputs_ids.to('cuda')  
    steps_entropy = []  # Used to store the entropy values at each step
    generated_sequence = inputs[0].tolist()  # Initial sequence
    generated_text = ""
    sum_t = 0 # Used to count the size of time step t

    with torch.inference_mode():  
        for _ in range(max_length):
            # Model predicts the distribution of the next word
            outputs = model(input_ids=torch.tensor([generated_sequence], device='cuda')) 
            next_token_logits = outputs.logits[:, -1, :]

            # Adjust the temperature
            next_token_logits = next_token_logits / temperature

            # Calculate the entropy value for the current step
            probabilities = torch.softmax(next_token_logits, dim=-1).squeeze()
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-5))
            steps_entropy.append(entropy)

            sum_t += 1

            # Calculate the global entropy and dynamically adjust k and alpha
            global_entropy = calculate_global_entropy(torch.stack(steps_entropy)) 
            max_entropy = torch.log(torch.tensor(len(probabilities), device='cuda'))  # Calculate the maximum entropy

            k_t, alpha_t = dynamic_adjustment(global_entropy, torch.stack(steps_entropy), max_entropy, sum_t, w=w)  # Obtain k_t and alpha_t

            # Get the probabilities of the top k_t candidate words
            top_k_probs, top_k_indices = torch.topk(probabilities, k_t)

            # Apply degeneration penalty to the already generated tokens
            adjusted_probs_full = apply_degeneration_penalty(probabilities, torch.tensor(generated_sequence, device='cuda'), alpha_t)

            # Retain only the probabilities of top_k_indices
            adjusted_top_k_probs = adjusted_probs_full[top_k_indices]

            # If the sum of the adjusted probabilities is 0, break the loop
            if torch.sum(adjusted_top_k_probs) == 0:
                break

            # Normalize the adjusted top_k probabilities
            adjusted_top_k_probs = adjusted_top_k_probs / torch.sum(adjusted_top_k_probs)

            # Sample the next token from the adjusted probability distribution
            next_token = torch.multinomial(adjusted_top_k_probs, 1).item()
            next_token = top_k_indices[next_token].item()

            # Add the next token to the generated sequence
            generated_sequence.append(next_token)

            # Check if the end token has been reached
            if next_token == tokenizer.eos_token_id:
                break

            # check the repetitive rate of the generated sentence
            repetitive_rate = cal_repetitive(generated_sequence)

            # Decode the newly generated token into text and add it to the generated text
            generated_text = tokenizer.decode(generated_sequence[prefix_len:], skip_special_tokens=True)

    return generated_text, repetitive_rate
