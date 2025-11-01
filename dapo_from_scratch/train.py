from modelscope.msdatasets.dataset_cls.custom_datasets.audio.kws_nearfield_processor import padding
from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os


class GSM8KDataset(Dataset):
    def __init__(self, data_pth, tokenizer):
        self.tokenizer = tokenizer
        data = load_dataset(data_pth)
        self.data = data['train']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        answer = item['answer_only']
        prompt = item['question_zh-cn']
        return {
            'prompt': prompt,
            'answer': answer
        }

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

class DAPOArguments:
    output_dir: str = "./output"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-6
    save_steps = 100
    epoch = 3
    num_generations = 4
    max_prompt_length = 256
    max_generate_length = 256
    reward_weights : List[float] = None
    beta = 0.0
    clip_eps_high = 0.28
    clip_eps_low = 0.2
    gradient_accumulation_steps = 2
    num_iterations = 1
    batch_size = 1

'''
Decoupled Clip and Dynamic sAmpling Policy Optimization, DAPO
1. Clip-Higher      
2. Dynamic Sampling
3. Token-Level Loss
软超长惩罚, 移除KL散度, 基于规则的奖励建模, etc.
详细参考:https://zhuanlan.zhihu.com/p/31157035727
'''
class DAPOTrainer:
    def __init__(self,
                 model = None,
                 reward_funcs: Union[List[str], List[Callable]] = None,
                 args = None,
                 train_dataset: Optional[Union[Dataset]] = None,
                 eval_dataset: Optional[Union[Dataset]] = None,
                 tokenizer = None,
                 reward_tokenizers = None):
        self.args = args
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)

        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(self.model).to(self.args.device)
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = self.get_tokenizer(tokenizer)

        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(reward_func, num_labels=1).to(self.args.device)
        self.reward_funcs = reward_funcs

        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(self.reward_funcs)
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        else:
            if len(reward_tokenizers) != len(self.reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to length of reward_funcs")

        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer

        self.reward_tokenizers = reward_tokenizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        self.update_steps = 0

    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        return tokenizer

    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        max_length = self.args.max_prompt_length + self.args.max_generate_length

        for prompt, answer in zip(prompts, answers):
            input_text = self.tokenzier.apply_chat_template(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ], add_generation_prompt=True, tokenize=False
            )
            inputs = self.tokenizer(
                [input_text] * self.args.num_generations, padding='max_length',
                max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt'
            )
            prompt_ids = inputs['input_ids'].to(self.args.device)
            with torch.no_grad():
                prompt_response_ids = self.model.generate(
                    **inputs.to(self.args.device),
                    max_new_tokens=self.args.max_generate_length,
                    temperature=0.9,
                    top_p=1,
                    top_k=50
                )
            if prompt_response_ids.size(1) > max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat(
                    [
                        prompt_response_ids,
                        torch.full(
                            (prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)),
                            fill_value=self.tokenizer.pad_token_id,
                            device=prompt_response_ids.device
                        )
                    ], dim=1
                )

            attn_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (prompt_response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)

            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt=prompt,
                answer=answer,
                attn_mask=attn_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list

    def generate_experiences(self, inputs):
        self.model.eval()
        samples_list = self.generate_samples(inputs)

        batch_prompt_response_ids = []
        batch_attn_mask = []
        batch_action_mask = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        batch_advantages = []

        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids
            response_ids = samples.response_ids
            prompt = samples.prompt
            answer = samples.answer
            attn_mask = samples.attn_mask
            action_mask = samples.action_mask
            num_actions = samples.num_actions

            with torch.no_grad():
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * self.args.num_generations
                prompt_response_texts = [p + r for p, r in zip(prompt_texts, response_texts)]
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, padding=True, return_tensors='pt')
                            # logits.squeeze(-1): [B, L, V] -> [B, L]
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    else:
                        answers = [answer] * self.args.num_generations
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)

                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("Length of reward_weights must be equal to length of reward_funcs")
                # [num_funcs, num_generations]
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, device=rewards_per_func.device).unsqueeze(1)
                rewards = rewards.sum(dim=0)  # [num_generations]

                mean_g_reward = rewards.mean()
                std_g_reward = rewards.std()
                advantages = (rewards - mean_g_reward) / (std_g_reward + 1e-8)
                # dapo动态采样
                non_zero_num = advantages.count_nonzero().item()
                if non_zero_num == 0:
                    continue
                batch_advantages.append(advantages)

                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attn_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)

                if self.ref_model:
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attn_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)

                batch_prompt_response_ids.append(prompt_response_ids)
                batch_attn_mask.append(attn_mask)
                batch_action_mask.append(action_mask)

        return {
            'prompt_response_ids': batch_prompt_response_ids,
            'attn_mask': batch_attn_mask,
            'action_mask': batch_action_mask,
            'old_action_log_probs': batch_old_action_log_probs,
            'ref_action_log_probs': batch_ref_action_log_probs,
            'advantages': batch_advantages
        }

    def compute_loss(self, model, inputs):
        prompt_respose_ids = inputs['prompt_response_ids']
        attn_mask = inputs['attn_mask']
        action_mask = inputs['action_mask']
        num_actions = inputs['num_actions']
        action_log_probs = self.get_action_log_probs(model, prompt_respose_ids, attn_mask, num_actions)

        if self.args.beta != 0.0:
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs
            log_ratio = log_ratio * action_mask
            k3 = log_ratio.exp() - log_ratio - 1

        advantages = inputs['advantages']
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        coef_2 = torch.clamp(coef_1, 1-self.args.clip_eps_low, 1+self.args.clip_eps_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)   # [B*num_generations, num_actions]
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3

        # # grpo loss
        # loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)   # [B*num_generations]
        # loss = loss.mean()

        # dapo loss
        per_token_loss = per_token_loss.view(-1, self.args.num_generations, num_actions)    # [B, num_generations, num_actions]
        action_mask = action_mask.view(-1, self.args.num_generations, num_actions)
        loss = per_token_loss.sum(-1).sum(-1) / action_mask.sum(-1).sum(-1)   # [B]
        loss = loss.mean()

        return loss


    def get_action_log_probs(self, model, input_ids, attn_mask, num_actions):
        # input_ids: [B, L]
        output = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = output.logits  # [B, L, V]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]  # [B, num_actions]
        return action_log_probs

    def train_step(self, model, inputs, optimizer, step):
        model.train()
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('dapo_loss', loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}, loss: {loss.item()}")

    def train(self):
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            buffer = {
                'prompt_response_ids': [],
                'attn_mask': [],
                'action_mask': [],
                'old_action_log_probs': [],
                'ref_action_log_probs': [],
                'advantages': []
            }
            idx = 0
            for batch in dataloader:
                inputs = self.generate_experiences(batch)
                buffer['prompt_response_ids'].append(inputs.prompt_response_ids)
                buffer['attn_mask'].append(inputs.attn_mask)
                buffer['action_mask'].append(inputs.action_mask)
                buffer['old_action_log_probs'].append(inputs.old_action_log_probs)
                if self.ref_model is not None:
                    buffer['ref_action_log_probs'].append(inputs.ref_action_log_probs)
                else:
                    buffer['ref_action_log_probs'].append(inputs.old_action_log_probs)
                buffer['advantages'].append(inputs.advantages)

                # dapo动态采样导致样本舍弃
                if len(buffer['prompt_response_ids']) < self.args.batch_size:
                    continue

                # 从缓冲区中取出一个batch进行训练，并将剩余的数据保留在缓冲区中
                if self.ref_model is not None:
                    inputs = {k: v[:self.args.batch_size] for k, v in buffer.items()}
                    inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}
                    buffer = {k: v[self.args.batch_size:] for k, v in buffer.items()}
                else:
                    inputs = {k: v[:self.args.batch_size] for k, v in buffer.items() if k != 'ref_action_log_probs'}
                    inputs = {k: torch.cat(v, dim=0) for k, v in inputs.items()}
                    inputs['ref_action_log_probs'] = None
                    buffer = {k: v[self.args.batch_size:] for k, v in buffer.items() if k != 'ref_action_log_probs'}
                    buffer['ref_action_log_probs'] = None
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs

                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                    self.update_steps += 1
                    if self.update_steps % self.args.save_steps == 0:
                        self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                        self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')

                idx += 1
                del inputs

    def save_model(self):
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

if __name__ == "__main__":
    SYSTEM_PROMPT = """
    按照如下格式回答问题：
    <think>
    你的思考过程
    </think>
    <answer>
    你的回答
    </answer>
    """

    args = DAPOArguments()

    writer = SummaryWriter('./runs')

    # policy model
    tokenizer = AutoTokenizer.from_pretrained()
    model = AutoModelForCausalLM.from_pretrained()
    # reward model
    #

    prompts_dataset = GSM8KDataset("gsm8k", tokenizer)
    trainer = DAPOTrainer(
        model=model,
        reward_funcs = [reward_correctness, reward_digit, reward_hard_format, reward_mark],
        args=args,
        train_dataset=prompts_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()