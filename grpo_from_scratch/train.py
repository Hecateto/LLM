from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']

    def _len(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        prompt = sample['question_zh-cn']
        answer = sample['answer_only']
        return {'prompt': prompt, 'answer': answer}

@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attn_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int

class GRPOArguments:
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-6
    save_steps = 100
    epoch = 3
    num_generations = 4 # 组内采样数量
    max_prompt_len = 256
    max_generate_len = 256
    reward_weights : List[float] = None # 多奖励函数时的权重
    beta = 0.0 # KL惩罚系数
    clip_eps = 0.2
    gradient_accumulation_steps = 2
    num_iters = 1   # 采样一次样本训练模型轮数
    batch_size = 1

class GRPOTrainer:
    def __init__(
        self,
        model: None,
        reward_fns: Union[Callable, List[Callable]] = None,
        args = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer = None,
        reward_tokenizers = None
    ):
        self.args = args
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)

        # self.ref_model
        self.ref_model = None
        if self.args.beta > 0.0:
            self.ref_model = deepcopy(self.model)
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()

        # self.tokenizer
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = self.get_tokenizer(tokenizer)

        # self.reward_fns
        if isinstance(reward_fns, str):
            reward_fns = [reward_fns]
        for i, reward_func in enumerate(reward_fns):
            if isinstance(reward_func, str):
                # AutoModelForSequenceClassification适用于分类任务的奖励模型
                # num_labels=1表示回归任务, 即输出一个实数作为奖励值
                reward_fns[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        self.reward_fns = reward_fns

        # self.reward_tokenizers
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(self.reward_fns)
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        else:
            if len(reward_tokenizers) != len(self.reward_fns):
                raise ValueError("Length of reward_tokenizers must match length of reward_fns")

        for i, (reward_tokenizer, reward_fn) in enumerate(zip(reward_tokenizers, reward_fns)):
            if isinstance(reward_fn, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_fn.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token

                reward_fn.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers

        # others
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # buffer for gradient accumulation
        self.input_buffer = [None] * self.args.gradient_accumulation_steps

        self.update_steps = 0

    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        return tokenizer

    # 生成样本, 以组为单位
    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)

        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        max_len = self.args.max_generate_len + self.args.max_prompt_len

        for prompt, answer in zip(prompts, answers):
            input_text = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ], add_generation_prompt=True, tokenize = False
            )
            inputs = self.tokenizer([input_text] * self.args.num_generations,
                                    padding='max_len',  # pad to max length
                                    max_len=self.args.max_prompt_len,
                                    truncation=True,
                                    return_tensors='pt'
                                    )
            prompt_ids = inputs['input_ids']
            with torch.no_grad():
                prompt_response_ids = self.model.generate(**inputs.to(self.args.device),
                                                          max_new_tokens=self.args.max_generate_len,
                                                          temperature=0.9,
                                                          top_p=1,  # nucleus sampling
                                                          top_k=50,
                                                          )
            if prompt_response_ids.size(1) > max_len:
                prompt_response_ids = prompt_response_ids[:, -max_len:]
            else:
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full(
                         (prompt_response_ids.size(0), max_len - prompt_response_ids.size(1)),
                            fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)],dim=1)

            attn_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (response_ids.ne(self.tokenizer.pad_token_id)
                           & response_ids.ne(self.tokenizer.eos_token_id)).to(dtype=torch.long)

            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt=prompt,
                answer=answer,
                attn_mask=attn_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),    # all tokens
                response_length=action_mask.float().sum(dim=-1) # valid tokens
            )
            samples_list.append(samples)

        return samples_list

    # 生成经验
    def generate_experiences(self, inputs):
        self.model.eval()
        samples_list = self.generate_samples(inputs)

        batch_prompt_response_ids = []
        batch_attn_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []

        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # [num_generations, seq_len]
            response_ids = samples.response_ids
            answer = samples.answer
            attn_mask = samples.attn_mask
            action_mask = samples.action_mask
            num_actions = samples.num_actions
            prompt = samples.prompt

            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attn_mask.append(attn_mask)
            batch_action_mask.append(action_mask)

            with torch.no_grad():
                old_action_log_probs = self.get_action_log_probs(
                    self.model, prompt_response_ids, attn_mask, action_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)

                if self.ref_model:
                    ref_action_log_probs = self.get_action_log_probs(
                        self.ref_model, prompt_response_ids, attn_mask, action_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)

                rewards_per_fn = torch.zeros(len(self.reward_fns), self.args.num_generations, device=self.args.device)

                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)    # [num_generations]
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [p + r for p, r in zip(prompt_texts, response_texts)]

                for i, (reward_fn, reward_tokenizer) in enumerate(
                    zip(self.reward_fns, self.reward_tokenizers)
                ):
                    if isinstance(reward_fn, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(
                                prompt_response_texts,
                                padding=True, return_tensors='pt')
                            # logits: [batch_size, num_labels -> squeeze to [batch_size]
                            # e.g., [ [0.5], [1.2], [0.3] ] -> [0.5, 1.2, 0.3]
                            rewards_per_fn = reward_fn(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    else:
                        answers = [answer] * len(response_texts)
                        output_reward_fn = reward_fn(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_fn = [reward if reward is not None else torch.nan for reward in output_reward_fn]
                        rewards_per_fn[i] = torch.tensor(output_reward_fn, dtype=torch.float32, device=self.args.device)

                # rewards_per_fn:  [num_reward_fns, num_generations]
                if not self.args.reward_weights:
                    self.reward_weights = [1.0] * len(self.reward_fns)
                if len(self.args.reward_weights) != len(self.reward_fns):
                    raise ValueError("Length of reward_weights must match length of reward_fns")

                rewards = rewards_per_fn * torch.tensor(
                    self.args.reward_weights,
                    dtype=torch.float32, device=rewards_per_fn.device).unsqueeze(1) # [num_reward_fns, 1]
                rewards = rewards.sum(dim=0)  # [num_generations]
                print(f"Rewards: {rewards}")

                g_mean = rewards.mean()
                g_std = rewards.std()
                advantages = (rewards - g_mean) / (g_std + 1e-8)
                batch_advantages.append(advantages)

        # B * [num_generations, seq_len] -> (B * num_generations, seq_len)
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attn_mask": torch.cat(batch_attn_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0)
        }

    def get_action_log_probs(self, model, input_ids, attn_mask, num_actions):
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))   # [b, s-1, 1]
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]  # [b, num_actions]
        return action_log_probs

    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs['prompt_response_ids']
        attn_mask = inputs['attn_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attn_mask, num_actions)

        if self.args.beta != 0.0:
            # KL( ref | pi ) = E[log(ref) - log(pi)] = E[log_ratio]
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs
            log_ratio = log_ratio * action_mask
            # k3是KL散度的二阶泰勒展开近似, 即 exp(x) ≈ 1 + x + x^2/2, 所以 exp(x) - 1 - x ≈ x^2/2
            k3 = log_ratio.exp() - 1 - log_ratio    # [b*num_generations, num_actions]

        advantages = inputs['advantages']   # [b*g]
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iters > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        coef_2 = torch.clamp(coef_1, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # [b*g, num_actions]
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3

        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)   # [b*g]
        loss = loss.mean()

        return loss

    def train_step(self, model, inputs, optimizer, step):
        model.train()
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"Step {self.update_steps}/{self.global_steps}, GRPO_Loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    def train(self):
        self.global_steps = self.args.num_iters * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                inputs = self.generate_experiences(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    for step, inputs in enumerate(self.input_buffer):
                        self.train_step(self.model, inputs, self.optimizer, step)
                    self.update_steps += 1
                    if self.update_steps % self.args.save_steps == 0:
                        self.model.save_pretrained(self.args.output_dir + 'f/checkpoint_{self.update_steps}')
                        self.tokenizer.save_pretrained(self.args.output_dir + 'f/checkpoint_{self.update_steps}')
                del inputs

    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    SYSTEM_PROMPT = """
    按照如下格式回答问题：
    <think>
    你的思考过程
    </think>
    <answer>
    你的回答
    </answer>
    """

    args = GRPOArguments()
    writer = SummaryWriter('./runs')

    # policy model
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # reward/critic model
    # reward_model_name = ''
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     reward_model_name, num_labels=1)
    # reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

    # dataset
    data_path = ''
    prompts_dataset = GSM8KDataset(data_path, tokenizer)
    trainer = GRPOTrainer(
        model=model,
        reward_fns=[reward_correctness, reward_digit, reward_hard_format, reward_mark],
        args=args,
        train_dataset=prompts_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()
