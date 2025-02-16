import torch
from global_entropy import *

from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Llama-3.1-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')  # 将模型移动到 GPU

# 5. 使用GPT-2 XL进行文本推理
def generate_text(model, tokenizer, inputs_ids, max_length=256, temperature=1.0, w=4):
    """生成文本并返回生成的文本和全局熵"""
    _, prefix_len = inputs_ids.size()
    inputs = inputs_ids.to('cuda')  # 确保输入在 GPU 上
    steps_entropy = []  # 用于存储每一步的熵值
    generated_sequence = inputs[0].tolist()  # 初始序列
    generated_text = ""
    sum_t = 0 #用来统计时间步t的大小

    with torch.inference_mode():  # 使用 inference_mode 代替 no_grad，以进一步优化推理性能
        for _ in range(max_length):
            # 模型预测下一个词的分布
            outputs = model(input_ids=torch.tensor([generated_sequence], device='cuda'))  # 输入也放到 GPU
            next_token_logits = outputs.logits[:, -1, :]

            # 调整温度
            next_token_logits = next_token_logits / temperature

            # 计算当前步骤的熵值（改为 GPU 上计算）
            probabilities = torch.softmax(next_token_logits, dim=-1).squeeze()
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-5))
            steps_entropy.append(entropy)

            sum_t += 1

            # 计算全局熵并动态调整 k 和 alpha
            global_entropy = calculate_global_entropy(torch.stack(steps_entropy))  # 确保在 GPU 上计算
            max_entropy = torch.log(torch.tensor(len(probabilities), device='cuda'))  # 计算最大熵

            k_t, alpha_t = dynamic_adjustment(global_entropy, torch.stack(steps_entropy), max_entropy, sum_t, w=w)  # 获取 k_t 和 alpha_t

            # 获取前 k_t 个候选词的概率
            top_k_probs, top_k_indices = torch.topk(probabilities, k_t)

            # 对已生成的 token 应用退化惩罚
            adjusted_probs_full = apply_degeneration_penalty(probabilities, torch.tensor(generated_sequence, device='cuda'), alpha_t)

            # 只保留 top_k_indices 的概率
            adjusted_top_k_probs = adjusted_probs_full[top_k_indices]

            # 如果调整后的概率之和为 0，跳出循环
            if torch.sum(adjusted_top_k_probs) == 0:
                break

            # 归一化调整后的 top_k 概率
            adjusted_top_k_probs = adjusted_top_k_probs / torch.sum(adjusted_top_k_probs)

            # 从调整后的概率分布中采样下一个 token
            next_token = torch.multinomial(adjusted_top_k_probs, 1).item()
            next_token = top_k_indices[next_token].item()

            # 将下一个 token 添加到生成序列中
            generated_sequence.append(next_token)

            # 检查是否达到结束标记
            if next_token == tokenizer.eos_token_id:
                break

            # check the repetitive rate of the generated sentence
            repetitive_rate = cal_repetitive(generated_sequence)

            # 将新生成的 token 解码为文本并添加到生成文本中
            generated_text = tokenizer.decode(generated_sequence[prefix_len:], skip_special_tokens=True)

    return generated_text, repetitive_rate
