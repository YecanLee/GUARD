import torch
from transformers import LogitsProcessorList, LogitsProcessor
import numpy as np

class EntropyBasedLogitsProcessor(LogitsProcessor):
    """
    Implements the entropy-based decoding algorithm as a LogitsProcessor for use with model.generate
    Args:
        w (int): The window size parameter for entropy calculation and dynamic adjustment
        decay_factor (float): The decay factor for entropy weight decay
    """
    def __init__(self, w=7, decay_factor=0.95):
        self.w = w
        self.decay_factor = decay_factor
        self.steps_entropy = []
        self.sum_t = 0
        
    def __call__(self, input_ids, scores):
        # Calculate probabilities from scores
        probabilities = torch.softmax(scores, dim=-1).squeeze()
        
        # Calculate entropy for current step
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-5))
        self.steps_entropy.append(entropy)
        self.sum_t += 1
        
        # Calculate global entropy
        steps_entropy_tensor = torch.stack(self.steps_entropy)
        global_entropy = self.calculate_global_entropy(steps_entropy_tensor)
        
        # Calculate maximum entropy
        max_entropy = torch.log(torch.tensor(scores.size(-1), device=scores.device))
        
        # Get dynamic k and alpha values
        k_t, alpha_t = self.dynamic_adjustment(global_entropy, steps_entropy_tensor, max_entropy, self.sum_t, w=self.w)
        
        # Get top-k probabilities and indices
        top_k_probs, top_k_indices = torch.topk(probabilities, k_t)
        
        # Apply degeneration penalty
        adjusted_probs_full = self.apply_degeneration_penalty(probabilities, input_ids[0], alpha_t)
        
        # Retain only the probabilities of top_k_indices
        adjusted_top_k_probs = adjusted_probs_full[top_k_indices]
        
        # If the sum of adjusted probabilities is 0, keep original scores
        if torch.sum(adjusted_top_k_probs) == 0:
            return scores
        
        # Normalize the adjusted top_k probabilities
        adjusted_top_k_probs = adjusted_top_k_probs / torch.sum(adjusted_top_k_probs)
        
        # Create a new scores tensor with the adjusted probabilities
        new_scores = torch.full_like(scores, float('-inf'))
        new_scores[0, top_k_indices] = torch.log(adjusted_top_k_probs + 1e-10)  # Convert back to log space
        
        return new_scores
    
    def calculate_global_entropy(self, steps_entropy):
        """Calculate global entropy with decay-weighted entropy"""
        # Calculate decay weight
        T = len(steps_entropy)
        decay_weights = torch.tensor([self.decay_factor ** (T - t - 1) for t in range(T)], device=steps_entropy.device)
        
        # Calculate weighted entropy using weights
        weighted_entropy = torch.sum(decay_weights * steps_entropy) / torch.sum(decay_weights)
        
        return weighted_entropy
    
    def dynamic_adjustment(self, global_entropy, step_entropy, max_entropy, sum_t, w=4):
        """Dynamically adjust the number of candidate words and degradation penalty weights"""
        epsilon = 1e-4  # Prevent numerical overflow, change this value may cause errors for generation
        
        # Ensure tensors are properly formatted
        if global_entropy.dim() == 0:
            global_entropy = global_entropy.unsqueeze(0)
        if step_entropy.dim() == 0:
            step_entropy = step_entropy.unsqueeze(0)
        
        median_entropy = torch.median(step_entropy) if len(step_entropy) > 0 else 0
        
        if sum_t < w:
            delta_t_raw = (step_entropy[-1] - median_entropy) / max_entropy
            delta_global_raw = 0
        else:
            local_entropy_window = step_entropy[sum_t - w:sum_t]
            median_local_entropy = torch.median(local_entropy_window) if len(local_entropy_window) > 0 else 0
            delta_t_raw = (step_entropy[-1] - median_local_entropy) / max_entropy
            
            median_entropy_window_t = torch.median(step_entropy[sum_t - w + 1:sum_t+1])
            median_entropy_global = torch.median(global_entropy[:len(step_entropy)]) if len(global_entropy) > 0 else 0
            delta_global_raw = (median_entropy_window_t - median_entropy_global) / max_entropy
        
        # Ensure delta_global_raw is a tensor
        if not isinstance(delta_global_raw, torch.Tensor):
            delta_global_raw = torch.tensor(delta_global_raw, device=step_entropy.device)
        
        # Calculate adaptive q
        if len(step_entropy) > 1:
            loc_entropy_change = torch.abs(step_entropy[-1] - step_entropy[-2]) / (step_entropy[-2] + epsilon)
            loc_entropy_change = torch.clamp(loc_entropy_change, min=0.0, max=1.0)
        else:
            loc_entropy_change = torch.tensor(0.0, device=step_entropy.device)
        
        if sum_t < w:
            glob_entropy_diff = torch.tensor(0.0, device=step_entropy.device)
        else:
            med_entropy_window_diff = torch.median(step_entropy[sum_t - w + 1:sum_t + 1])
            med_entropy_global_diff = torch.median(global_entropy[:len(step_entropy)]) if len(global_entropy) > 0 else 0
            glob_entropy_diff = torch.abs(med_entropy_window_diff - med_entropy_global_diff) / (med_entropy_window_diff + epsilon)
            glob_entropy_diff = torch.clamp(glob_entropy_diff, min=0.0, max=1.0)
        
        prev_q = 1.0
        q = prev_q * 0.9 + (1.0 + loc_entropy_change + glob_entropy_diff) * 0.1
        
        # Calculate delta_t and delta_global
        delta_t = q * torch.atanh(torch.clamp(delta_t_raw, -1 + epsilon, 1 - epsilon))
        delta_global = q * torch.atanh(torch.clamp(delta_global_raw, -1 + epsilon, 1 - epsilon))
        
        # Calculate lambda_k
        lambda_k = abs(delta_t) / (abs(delta_global) + abs(delta_t) + epsilon)
        
        # Calculate k_t
        exp_value = torch.exp(lambda_k * delta_t + (1 - lambda_k) * delta_global)
        k_t = 10 * exp_value / (exp_value + 1) + 5
        
        # Calculate alpha_t
        ln_k = torch.log(k_t + epsilon)
        
        if sum_t < w:
            median_entropy_a = median_entropy
            delta_loc_a_raw = (step_entropy[-1] - median_entropy_a) / ln_k
            delta_glob_a_raw = 0
        else:
            local_entropy_window_a = step_entropy[sum_t - w:sum_t]
            median_local_entropy_a = torch.median(local_entropy_window_a) if len(step_entropy) > 0 else 0
            delta_loc_a_raw = (step_entropy[-1] - median_local_entropy_a) / ln_k
            
            median_entropy_window_a = torch.median(step_entropy[sum_t - w + 1:sum_t+1])
            median_entropy_global_a = torch.median(global_entropy[:len(step_entropy)]) if len(global_entropy) > 0 else 0
            delta_glob_a_raw = (median_entropy_window_a - median_entropy_global_a) / ln_k
        
        # Ensure delta_glob_a_raw is a tensor
        if not isinstance(delta_glob_a_raw, torch.Tensor):
            delta_glob_a_raw = torch.tensor(delta_glob_a_raw, device=step_entropy.device)
        
        # Calculate delta_t_a and delta_glob_a
        delta_t_a = q * torch.atanh(torch.clamp(delta_loc_a_raw, -1 + epsilon, 1 - epsilon))
        delta_glob_a = q * torch.atanh(torch.clamp(delta_glob_a_raw, -1 + epsilon, 1 - epsilon))
        
        # Calculate lambda_a
        lambda_a = abs(delta_t_a) / (abs(delta_glob_a) + abs(delta_t_a) + epsilon)
        
        # Calculate alpha_t
        exp_value_a = torch.exp(lambda_a * delta_t_a + (1 - lambda_a) * delta_glob_a)
        alpha_t = exp_value_a / (exp_value_a + 1)
        
        return int(k_t.item()), min(max(alpha_t.item(), 0), 1)
    
    def apply_degeneration_penalty(self, probabilities, generated_tokens, alpha_t):
        """Apply degeneration penalty to the already generated tokens"""
        # Get the frequency of the already generated tokens
        token_counts = torch.bincount(generated_tokens, minlength=len(probabilities))
        
        # Apply a penalty to the already generated tokens
        penalties = alpha_t ** token_counts
        adjusted_probs = probabilities * penalties
        
        # Normalize the probability distribution
        adjusted_probs = adjusted_probs / torch.sum(adjusted_probs)
        
        return adjusted_probs