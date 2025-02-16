class DynamicLogitsWarper(LogitsProcessor):
    def __init__(
        self, 
        temperature: float = 1.0, 
        decay_factor: float = 0.95, 
        window_size: int = 4,
        q: float = 1.0,
        filter_value: float = -float("Inf"),
        repetition_penalty: float = 1.2,  # Added repetition penalty
        top_k: int = 50,  # Added top-k filtering
        min_tokens_to_keep: int = 1
    ):
        self.decay_factor = decay_factor
        self.temperature = temperature
        self.window_size = window_size
        self.q = q
        self.filter_value = filter_value
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.steps_entropy: List[torch.Tensor] = []
        self.time_step = 0

    def _calculate_global_entropy(self, steps_scores: torch.FloatTensor) -> torch.FloatTensor:
        T = len(steps_scores)
        decay_weights = torch.tensor(
            [self.decay_factor ** (T - t - 1) for t in range(T)],
            device=steps_scores.device
        )
        return torch.sum(decay_weights * steps_scores) / torch.sum(decay_weights)

    def _dynamic_adjustment(self, 
                          step_entropy: torch.Tensor,
                          global_entropy: torch.Tensor,
                          max_entropy: torch.Tensor) -> tuple[int, float]:
        epsilon = 1e-10
        
        # Improved local entropy calculation with smoothing
        if self.time_step < self.window_size:
            median_entropy = torch.median(step_entropy) if len(step_entropy) > 0 else 0
            delta_t_raw = (step_entropy[-1] - median_entropy) / (max_entropy + epsilon)
            delta_global_raw = 0
        else:
            # Enhanced local window analysis
            local_window = step_entropy[-self.window_size:]
            median_local = torch.median(local_window)
            mean_local = torch.mean(local_window)
            # Use both mean and median for more stable estimation
            local_stat = (median_local + mean_local) / 2
            delta_t_raw = (step_entropy[-1] - local_stat) / (max_entropy + epsilon)
            
            # Improved global entropy calculation
            window_stat = (torch.median(step_entropy[-self.window_size:]) + torch.mean(step_entropy[-self.window_size:])) / 2
            delta_global_raw = (window_stat - global_entropy) / (max_entropy + epsilon)

        # Convert to tensor and ensure device consistency
        delta_t_raw = torch.as_tensor(delta_t_raw, device=step_entropy.device)
        delta_global_raw = torch.as_tensor(delta_global_raw, device=step_entropy.device)

        # Enhanced delta calculations with smoothing
        delta_t = self.q * torch.atanh(torch.clamp(delta_t_raw, -1 + epsilon, 1 - epsilon))
        delta_global = self.q * torch.atanh(torch.clamp(delta_global_raw, -1 + epsilon, 1 - epsilon))

        # Improved k_t calculation with better scaling
        lambda_k = torch.abs(delta_t) / (torch.abs(delta_t) + torch.abs(delta_global) + epsilon)
        exp_value = torch.exp(lambda_k * delta_t + (1 - lambda_k) * delta_global)
        k_t = self.top_k * exp_value / (1 + exp_value)  # Dynamic k value scaled by top_k
        if torch.isnan(k_t):
            k_t = torch.tensor(self.top_k // 2, device=step_entropy.device)

        # Enhanced alpha_t calculation
        ln_k = torch.log(k_t + epsilon)
        if self.time_step < self.window_size:
            delta_loc_a = (step_entropy[-1] - torch.median(step_entropy)) / ln_k
            delta_glob_a = 0
        else:
            delta_loc_a = (step_entropy[-1] - local_stat) / ln_k
            delta_glob_a = (window_stat - global_entropy) / ln_k

        if isinstance(delta_loc_a, torch.Tensor):
            delta_loc_a = delta_loc_a.clone().detach()
        else:
            delta_loc_a = torch.tensor(delta_loc_a, device=step_entropy.device)
        if isinstance(delta_glob_a, torch.Tensor):
            delta_glob_a = delta_glob_a.clone().detach()
        else:
            delta_glob_a = torch.tensor(delta_glob_a, device=step_entropy.device)
        
        delta_t_a = self.q * torch.atanh(torch.clamp(delta_loc_a, -1 + epsilon, 1 - epsilon))
        delta_global_a = self.q * torch.atanh(torch.clamp(delta_glob_a, -1 + epsilon, 1 - epsilon))

        lambda_a = torch.abs(delta_t_a) / (torch.abs(delta_t_a) + torch.abs(delta_global_a) + epsilon)
        exp_value_a = torch.exp(lambda_a * delta_t_a + (1 - lambda_a) * delta_global_a)
        alpha_t = exp_value_a / (1 + exp_value_a)

        return int(k_t.item()), float(alpha_t.item())

    def _apply_repetition_penalty(self, scores: torch.FloatTensor, input_ids: torch.LongTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # If score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.repetition_penalty, score / self.repetition_penalty)
        scores.scatter_(1, input_ids, score)
        
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Apply temperature scaling
        scores = scores / self.temperature
        
        # Calculate probabilities and entropy
        probabilities = torch.softmax(scores, dim=-1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
        self.steps_entropy.append(entropy)

        # Calculate global statistics
        steps_entropy_tensor = torch.stack(self.steps_entropy)
        global_entropy = self._calculate_global_entropy(steps_entropy_tensor)
        max_entropy = torch.log(torch.tensor(scores.size(-1), device=scores.device))

        # Get dynamic parameters
        k_t, alpha_t = self._dynamic_adjustment(steps_entropy_tensor, global_entropy, max_entropy)
        
        # Apply repetition penalty
        scores = self._apply_repetition_penalty(scores, input_ids)
        
        # Apply top-k filtering
        top_k = min(k_t, scores.size(-1))  # Safety check
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        
        self.time_step += 1
        return scores