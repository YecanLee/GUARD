import torch
#from test_good import *
import numpy as np


def calculate_global_entropy(steps_entropy, decay_factor=0.95):
    """Calculate global entropy with decay-weighted entropy"""
    if not isinstance(steps_entropy, torch.Tensor):
        steps_entropy = torch.tensor(steps_entropy, device='cuda')

    # Calculate decay weight
    T = len(steps_entropy)
    decay_weights = torch.tensor([decay_factor ** (T - t - 1) for t in range(T)], device='cuda')

    # Calculate weighted entropy using weights
    weighted_entropy = torch.sum(decay_weights * steps_entropy) / torch.sum(decay_weights)

    global_entropy =  weighted_entropy

    return global_entropy

def dynamic_adjustment(global_entropy, step_entropy, max_entropy, sum_t, w=4):

    """Dynamically adjust the number of candidate words and degradation penalty weights based on global entropy and current local entropy"""
    epsilon = 1e-4  # Prevent numerical overflow.

    # Convert the input to a tensor of at least 1 dimension and place it on CUDA
    step_entropy = torch.tensor(step_entropy, device='cuda').clone().detach() if not isinstance(step_entropy,
                                                                                                torch.Tensor) else step_entropy
    global_entropy = torch.tensor(global_entropy, device='cuda').clone().detach() if not isinstance(global_entropy,
                                                                                                    torch.Tensor) else global_entropy
    if global_entropy.dim() == 0:
        global_entropy = global_entropy.unsqueeze(0)
    if step_entropy.dim() == 0:
        step_entropy = step_entropy.unsqueeze(0)

    median_entropy = torch.median(step_entropy) if len(step_entropy) > 0 else 0

    if sum_t < w:
        # Use the first formula: δ_loc = q * arctanh((H(X)^t - median(H(X)^{<t})) / (max_entropy))
        delta_t_raw = (step_entropy[-1] - median_entropy) / max_entropy
        delta_global_raw = 0  # δ_glob = 0
    else:
        # Use the second formula: δ_loc = q * arctanh((H(X)^t - median(H(X)^{t-w:t-1})) / (max_entropy))

        # Calculate the local entropy difference at time step t
        local_entropy_window = step_entropy[sum_t - w:sum_t]  # Get the local entropy values within the window
        median_local_entropy = torch.median(local_entropy_window) if len(local_entropy_window) > 0 else 0
        delta_t_raw = (step_entropy[-1] - median_local_entropy) / max_entropy


        # δ_glob = q * arctanh((median(H(X)^{t-w+1:t}) - median(H_{glob}^{<t})) / (max_entropy))
        # Calculate the global entropy difference.
        median_entropy_window_t = torch.median(step_entropy[sum_t - w + 1:sum_t+1])
        median_entropy_global = torch.median(global_entropy[:len(step_entropy)]) if len(global_entropy) > 0 else 0
        delta_global_raw = (median_entropy_window_t - median_entropy_global) / max_entropy

    # Ensure that delta_global_raw is a tensor.
    if isinstance(delta_global_raw, torch.Tensor):
        delta_global_raw_tensor = delta_global_raw.clone().detach()
    else:
        delta_global_raw_tensor = torch.tensor(delta_global_raw, device='cuda')

    # Calculate adaptive q
    if len(step_entropy) >1 :
        loc_entropy_change = torch.abs(step_entropy[-1] - step_entropy[-2]) / (step_entropy[-2] + epsilon)
        loc_entropy_change = torch.clamp(loc_entropy_change, min=0.0, max=1.0)  # Clip to the range [0, 1]
    else:
        loc_entropy_change = torch.tensor(0.0, device='cuda')

    if sum_t < w:
        glob_entropy_diff = torch.tensor(0.0, device='cuda')
    else:
        med_entropy_window_diff = torch.median(step_entropy[sum_t - w + 1:sum_t + 1])
        med_entropy_global_diff = torch.median(global_entropy[:len(step_entropy)]) if len(global_entropy) > 0 else 0
        glob_entropy_diff = torch.abs(med_entropy_window_diff - med_entropy_global_diff) / (med_entropy_window_diff + epsilon)
        glob_entropy_diff = torch.clamp(glob_entropy_diff, min=0.0, max=1.0)  # Clip to the range [0, 1]

    prev_q = 1.0
    q = prev_q * 0.9 + (1.0 + loc_entropy_change + glob_entropy_diff) * 0.1  # Use moving average to smooth q

    # Use the arctanh function to calculate the final δ_loc and δ_glob
    delta_t = q * torch.atanh(torch.clamp(delta_t_raw.clone().detach(), -1 + epsilon, 1 - epsilon))
    delta_global = q * torch.atanh(torch.clamp(delta_global_raw_tensor, -1 + epsilon, 1 - epsilon))
    
    # Introduce the parameter lambda_k to control the ratio of global entropy parameters to local entropy parameters
    
    lambda_k = abs(delta_t) / (abs(delta_global)+abs(delta_t)+epsilon)
    
    # Calculate k_t according to the formula
    
    exp_value = torch.exp(lambda_k * delta_t + (1 - lambda_k) * delta_global)
    k_t = 10 * exp_value / (exp_value + 1) + 5

    # Calculate δ_loc^k 和 δ_glob^k
    ln_k = torch.log((k_t + epsilon).clone().detach())
    if sum_t < w:
        # When the time step is less than the window size.
        median_entropy_a = median_entropy
        delta_loc_a_raw = (step_entropy[-1] - median_entropy_a) / ln_k
        delta_glob_a_raw = 0
    else:
        #Calculate the local entropy difference.
        local_entropy_window_a  = step_entropy[sum_t - w:sum_t]
        median_local_entropy_a = torch.median(local_entropy_window_a) if len(step_entropy) > 0 else 0
        delta_loc_a_raw = (step_entropy[-1] - median_local_entropy_a) / ln_k

        #Calculate the global entropy difference.
        median_entropy_window_a = torch.median(step_entropy[sum_t - w + 1:sum_t+1])
        median_entropy_global_a = torch.median(global_entropy[:len(step_entropy)]) if len(global_entropy) > 0 else 0
        delta_glob_a_raw = (median_entropy_window_a-median_entropy_global_a) / ln_k

    # Ensure that delta_global_raw is a tensor
    if isinstance(delta_glob_a_raw, torch.Tensor):
        delta_glob_a_raw_tensor = delta_glob_a_raw.clone().detach()
    else:
        delta_glob_a_raw_tensor = torch.tensor(delta_glob_a_raw, device='cuda')

    # Calculate the final δ_loc^k and δ_glob^k.
    delta_t_a = q * torch.atanh(torch.clamp(delta_loc_a_raw.clone().detach(), -1 + epsilon, 1 - epsilon))
    delta_glob_a = q * torch.atanh(torch.clamp(delta_glob_a_raw_tensor, -1 + epsilon, 1 - epsilon))

    #Introduce the parameter lambda_a to control the parameter ratio
    
    lambda_a = abs(delta_t_a) / (abs(delta_glob_a) + abs(delta_t_a) + epsilon)
    
    # Calculate a_t according to the formula
    
    exp_value_a = torch.exp(lambda_a * delta_t_a + (1 - lambda_a) * delta_glob_a)
    alpha_t = exp_value_a / (exp_value_a + 1)

    return int(k_t.item()), min(max(alpha_t.item(), 0), 1)  # 确保 alpha 在 [0, 1] 之间


def apply_degeneration_penalty(probabilities, generated_tokens, alpha_t):
    """Apply degeneration penalty to the already generated tokens"""
    # Ensure that probabilities is a PyTorch tensor and on the GPU
    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.tensor(probabilities, device='cuda')
    generated_tokens = torch.tensor(generated_tokens, device='cuda') if not isinstance(generated_tokens,
                                                                                       torch.Tensor) else generated_tokens

    # Get the frequency of the already generated tokens
    token_counts = torch.bincount(generated_tokens, minlength=len(probabilities))

    # Apply a penalty to the already generated tokens
    penalties = alpha_t ** token_counts
    adjusted_probs = probabilities * penalties

    # Normalize the probability distribution
    adjusted_probs = adjusted_probs / torch.sum(adjusted_probs)

    return adjusted_probs

def cal_repetitive(input_ids,tokenizer):
    generated_tokens = tokenizer.encode(input_ids)

    if not isinstance(generated_tokens, torch.Tensor):
        generated_tokens = torch.tensor(generated_tokens, device='cuda')

    token_counts = torch.bincount(generated_tokens)

    total_tokens = len(generated_tokens)
    unique_tokens = len(torch.unique(generated_tokens))

    repetition_rate = float((total_tokens - unique_tokens) / total_tokens)

    return repetition_rate
