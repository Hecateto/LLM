import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType


# 1. CoT
SYSTEM_PROMPT = """
按照如下格式生成：
<think>
...
</think>
<answer>
...
</answer>
"""

def process_data(data):
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh-cn']}
        ],
        'answer': x['answer_only']
    })
    return data

def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# 2. Rewards
def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
    if text.count("</think>\n") == 1:
        reward += 0.125
    if text.count("<answer>\n") == 1:
        reward += 0.125
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

def reward_correctness(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0][-1]['content']}",
          f"\n答案:\n{answer[0]}",
          f"\n模型输出:\n{responses[0]}",
          f"\n提取后的答案:\n{extracted[0]}")
    return [2.0 if res == str(ans) else 0.0 for res, ans in zip(extracted, answer)]

# 稀疏性补充
def reward_digit(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer(r) for r in responses]
    return [0.5 if res.isdigit() else 0.0 for res in extracted]

def reward_hard_format(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, res) for res in responses]
    return [0.5 if match else 0.0 for match in matches]

def reward_soft_format(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, res) for res in responses]
    return [0.5 if match else 0.0 for match in matches]

def reward_mark(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    return [mark_num(res) for res in responses]

if __name__ == '__main__':
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # # lora
    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=256,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_dropout=0.1,
    # )
    # model = get_peft_model(model, lora_config)

    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_path = "gsm8k"
    ds = load_dataset(data_path)
    data = process_data(ds['train'])
    output_dir = 'output'
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,     # 每个样本生成16个不同的回答, GRPO特有参数
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard"
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correctness,
            reward_digit,
            reward_hard_format,
            reward_soft_format,
            reward_mark
        ],
        args=training_args,
        train_dataset=data
    )
    trainer.train()
    trainer.save_model(output_dir)





