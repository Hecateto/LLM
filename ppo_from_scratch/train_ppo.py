import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from accelerate import optimizer
from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.final_prompts = []
        for prompt in self.prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.final_prompts[idx]


class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()  # 冻结基础模型参数
        # 价值头的作用是将基础模型的隐藏状态映射到一个标量值，表示该状态的价值估计
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attn_mask, num_actions):
        # 获取模型最后一层的隐藏状态: last_hidden_state [batch_size, seq_length, hidden_size]
        hidden_states = self.base_model(input_ids, attention_mask=attn_mask).last_hidden_state
        value_model_output = self.value_head(hidden_states)
        # squeeze(-1): [batch_size, seq_length]
        # 只取动作对应的价值估计, 具体来说是取序列末尾num_actions个位置的价值估计（即新生成的token）
        values = value_model_output.squeeze(-1)[:, -num_actions:]
        return values


def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    # 计算PPO的策略损失
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2)
    if action_mask is None:
        # loss: [batch_size, seq_length] -> [batch_size] -> scalar
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps=0.2):
    if clip_eps is not None:
        # 限制价值函数的更新幅度
        clipped_values = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
        # loss1: 未裁剪的价值函数损失
        # values是当前价值估计,由critic_model计算得到
        # returns是实际回报, 由reward_model计算得到
        loss1 = (values - returns) ** 2
        # loss2: 裁剪后的价值函数损失
        loss2 = (clipped_values - returns) ** 2
        # 惩罚较大的误差
        loss = torch.max(loss1, loss2)
    else:
        loss = (values - returns) ** 2
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


@dataclass
class Samples:
    # 表示一次经验采样的结果
    seqs: torch.Tensor  # [batch_size, seq_length]
    attn_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]  # 表示每个序列的实际长度（不包括填充部分）
    response_length: torch.Tensor
    total_length: torch.Tensor


@dataclass
class Experience:
    # 和Samples的区别在于增加了与PPO算法相关的信息
    # 具体来说, Experience类包含了动作的对数概率,价值估计,回报,优势等信息, 在PPO算法中用于计算损失函数和更新策略
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attn_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None  # 策略与参考模型之间的KL散度


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []

    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
            "seqs",
            "action_log_probs",
            "values",
            "returns",
            "advantages",
            "attn_mask",
            "action_mask",
            "num_actions"
        )

        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
        # extend和append的区别在于extend是将一个可迭代对象中的元素逐一添加到列表中
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[-self.limit:]

    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


@dataclass
class BufferItem:
    # 用于将多个Experience对象合并成一个批处理对象
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attn_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]
    num_actions: Union[int, torch.Tensor]


def collate_fn(batch):
    # 1. 提取所有数据
    seqs = [x['seqs'] for x in batch]
    action_log_probs = [x['action_log_probs'] for x in batch]
    values = [x['values'] for x in batch]
    returns = [x['returns'] for x in batch]
    advantages = [x['advantages'] for x in batch]
    attn_masks = [x['attn_mask'] for x in batch]
    action_masks = [x['action_mask'] for x in batch]

    # 2. 确定批次中的最大响应长度
    # 奖励、价值等张量维度为 (batch_size, response_length)
    max_response_len = max(t.size(1) for t in action_log_probs)

    # 3. 对所有 (batch_size, response_length) 的张量进行填充
    padded_action_log_probs = []
    padded_values = []
    padded_returns = []
    padded_advantages = []
    padded_action_masks = []

    for log_probs, val, ret, adv, mask in zip(action_log_probs, values, returns, advantages, action_masks):
        current_len = log_probs.size(1)
        padding_len = max_response_len - current_len

        # 定义填充值：对数概率、价值、回报、优势等填充为0
        # 注意：mask需要填充为0 (False)

        # 填充动作对数概率 (log_probs)
        padded_log_probs = F.pad(log_probs, (0, padding_len), 'constant', 0)
        padded_action_log_probs.append(padded_log_probs)

        # 填充价值、回报、优势
        padded_values.append(F.pad(val, (0, padding_len), 'constant', 0))
        padded_returns.append(F.pad(ret, (0, padding_len), 'constant', 0))
        padded_advantages.append(F.pad(adv, (0, padding_len), 'constant', 0))

        # 填充动作掩码 (mask)，注意类型
        # action_mask 原本是 BoolTensor，但你在前面代码中转成了 LongTensor，所以这里用0填充
        padded_mask = F.pad(mask.float(), (0, padding_len), 'constant', 0).long()
        padded_action_masks.append(padded_mask)

    # 4. 拼接所有数据
    seqs = torch.cat(seqs, dim=0)  # [B, T_total]
    attn_masks = torch.cat(attn_masks, dim=0)  # [B, T_total]

    # 拼接填充后的张量 [B, T_response_max]
    final_action_log_probs = torch.cat(padded_action_log_probs, dim=0)
    final_values = torch.cat(padded_values, dim=0)
    final_returns = torch.cat(padded_returns, dim=0)
    final_advantages = torch.cat(padded_advantages, dim=0)
    final_action_masks = torch.cat(padded_action_masks, dim=0)

    # 这里的 num_actions 应该是 batch 中最大的响应长度
    num_actions = max_response_len

    # 5. 返回 BufferItem
    return BufferItem(
        seqs,
        final_action_log_probs,
        final_values,
        final_returns,
        final_advantages,
        attn_masks,
        final_action_masks,
        num_actions
    )


def train_step(experience, steps):
    actor_model.train()
    actor_optimizer.zero_grad()

    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attn_mask = experience.attn_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns

    # 计算当前策略下的动作对数概率
    logits = actor_model(sequences, attention_mask=attn_mask).logits  # [batch_size, seq_length, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    # gather: 根据sequences[:,1:]中的token ID, 从log_probs中提取对应的对数概率
    # sequences.unsqueeze(-1): [batch_size, seq_length] -> [batch_size, seq_length, 1]
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    # 只取动作对应的对数概率
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

    # 策略损失指的是PPO算法中的策略损失函数, 用于衡量当前策略与旧策略之间的差异
    # 策略损失用于更新策略模型的参数, 以最大化预期回报
    policy_loss = compute_policy_loss(
        action_log_probs,
        old_action_log_probs,
        advantages
    )
    policy_loss.backward()
    actor_optimizer.step()
    writer.add_scalar('policy_loss', policy_loss.item(), steps)

    # 价值损失指的是PPO算法中的价值损失函数, 用于衡量当前价值估计与实际回报之间的差异
    # 价值损失用于更新价值模型的参数, 以提高价值估计的准确性
    critic_model.train()
    critic_optimizer.zero_grad()
    values = critic_model.forward(sequences, attn_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    critic_optimizer.step()
    writer.add_scalar('value_loss', value_loss.item(), steps)
    print(f"step: {steps}, policy_loss: {policy_loss.item():.4f}, value_loss: {value_loss.item():.4f}")


def compute_approx_kl(
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
):
    log_ratio = log_probs - ref_log_probs
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio


def get_advantanges_and_returns(
        values: torch.Tensor,  # [batch_size, response_length]
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float
):
    # 计算GAE优势和回报
    lastGAElam = 0
    advantages_reversed = []
    response_length = rewards.size(1)

    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    # A(t) = R(t) + γV(t+1) - V(t)
    # R(t)表示在时间步t获得的即时奖励, V(t)表示在时间步t的状态价值估计, V(t+1)表示在时间步t+1的状态价值估计
    # GAE优势计算公式: A(t) = δ(t) + γλA(t+1)
    # 其中, δ(t) = R(t) + γV(t+1) - V(t)
    # γ (gamma): 折扣因子, 用于衡量未来奖励的重要性. 取值范围为0到1. 较高的γ值表示未来奖励更重要, 而较低的γ值表示当前奖励更重要.
    # λ (lambda): GAE的平滑参数, 取值范围为0到1. 较高的λ值会导致优势估计更加平滑, 而较低的λ值会使优势估计更加依赖于即时奖励.
    # 通过调整这两个参数, 可以在偏差和方差之间进行权衡, 从而影响策略的学习效果.
    # 具体来说, 较高的γ和λ值通常会导致更稳定的学习过程, 但可能会增加计算复杂性.
    # 较低的γ和λ值可能会加快学习速度, 但可能会导致不稳定的策略更新.

    for t in reversed(range(response_length)):
        # A(t+1) = 0, V(t+1) = 0, A(t) = δ(t) = R(t) - V(t)         (·)
        # A(t-1) = δ(t-1) + γλA(t) = R(t-1) + γV(t) - V(t-1) + γλA(t)
        # 已知A(t), 可计算A(t-1), 依此类推.
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastGAElam = delta + gamma * lambd * lastGAElam
        advantages_reversed.append(lastGAElam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # returns表示从当前时间步t开始到未来所有时间步的累计奖励
    # returns(t) = A(t) + V(t) = R(t) + γV(t+1) - V(t) + V(t) = R(t) + γV(t+1) + γλA(t+1) = R(t) + γ*returns(t+1)
    # 这里returns(t)和A(t)在最大t时的表达相同.                          (·)
    returns = advantages + values
    return advantages.detach(), returns


def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    # 根据提示词生成样本
    samples_list = []
    model.eval()
    all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], [])
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        # micro_rollout_batch_size: 每次取多少条经验生成经验
        prompts = all_prompts[i:i + micro_rollout_batch_size]
        inputs = actor_tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        # **将字典解包为关键字参数传递给函数
        seqs = model.generate(**inputs.to(device),
                              max_new_tokens=max_new_tokens,
                              eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        # seqs: [batch_size, seq_length]
        if seqs.size(1) > max_length + max_new_tokens:
            seqs = seqs[:, :max_length + max_new_tokens]
        else:
            seqs = torch.cat([seqs,
                              torch.full(
                                  (seqs.size(0), max_length + max_new_tokens - seqs.size(1)),
                                  fill_value=pad_token_id,
                                  device=seqs.device)], dim=1)

        attn_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        ans = seqs[:, input_ids.size(1):]
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long)

        samples = Samples(
            seqs=seqs,
            attn_mask=attn_mask,
            action_mask=action_mask,
            num_actions=ans.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attn_mask.float().sum(dim=-1)
        )
        samples_list.append(samples)
    return samples_list


# 计算R(t) = - KL散度 + 外部奖励值
def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    # kl_ctl: 控制KL散度的系数, r: 奖励值, kl: KL散度 [batch_size, seq_length]
    kl_divergence_estimate = -kl_ctl * kl
    rewards = kl_divergence_estimate
    # .sum(1): [batch_size], +1 定位到最后一个有效动作的位置
    ends = action_mask.sum(1) + 1
    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)
    reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)  # [batch_size, 1]

    batch_size = r.size(0)
    # 把外部奖励值加到每个序列的最后一个有效动作位置上
    for j in range(batch_size):
        rewards[j, :ends[j]][-1] += reward_clip[j, 0]
    return rewards


def generate_experiences(samples_list):
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []
    for samples in samples_list:
        seqs = samples.seqs  # [batch_size, seq_length]
        attn_mask = samples.attn_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        with torch.no_grad():
            # 计算当前策略下的动作对数概率
            logits = actor_model(seqs, attention_mask=attn_mask).logits  # [batch_size, seq_length, vocab_size]
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

            # 计算参考模型下的动作对数概率
            ref_logits = ref_model(samples.seqs, attention_mask=attn_mask).logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=samples.seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -samples.num_actions:]

            # 计算价值估计
            value = critic_model.forward(seqs, attn_mask, num_actions).to(device)
            # 计算外部奖励r, 对生成的完整文本（seq_texts）打分
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)
            r = reward_model(**reward_model_inputs.to(device)).logits  # [batch_size, num_labels]
            # 计算KL散度(action-ref)
            kl = compute_approx_kl(
                action_log_probs,
                ref_action_log_probs,
                action_mask=action_mask
            ).to(device)
            # 计算实际奖励: -kl_ctl * kl + r
            rewards = compute_rewards(
                kl,
                r,
                action_mask,
                kl_ctl=0.1,
                clip_reward_value=0.2
            )

            # 计算优势和回报
            advantages, returns = get_advantanges_and_returns(
                value,
                rewards,
                samples.action_mask,
                gamma=0.1,
                lambd=0.2
            )

        experience = Experience(
            seqs=seqs,
            action_log_probs=action_log_probs.detach(),
            values=value.detach(),
            returns=returns.detach(),
            advantages=advantages.detach(),
            attn_mask=attn_mask,
            action_mask=action_mask,
            reward=r.detach(),
            response_length=samples.response_length,
            total_length=samples.total_length,
            num_actions=num_actions,
            kl=kl.detach()
        )
        experiences.append(experience)
    return experiences


def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompt_dataloader:
            samples = generate_samples(
                rand_prompts,
                actor_model,
                max_length,
                max_new_tokens,
                n_samples_per_prompt,
                micro_rollout_batch_size
            )
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            buffer.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    episodes = 3
    max_epochs = 5
    rollout_batch_size = 8  # 每次从提示词数据集中采样的数量用于生成经验
    micro_rollout_batch_size = 2  # 每次取多少条经验生成经验
    n_samples_per_prompt = 2  # 每个提示词生成多少条经验
    max_new_tokens = 50
    max_length = 256
    micro_train_batch_size = 2  # 每次训练取多少条经验
    writer = SummaryWriter('./runs')  # TensorBoard日志记录器

    actor_path = "Qwen/Qwen2-0.5B-Instruct"
    ref_path = "Qwen/Qwen2-0.5B-Instruct"
    reward_path = "OpenAssistant/reward-model-deberta-v3-large-v2"
    # models
    actor_model = AutoModelForCausalLM.from_pretrained(actor_path).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_path).to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_path).to(device)
    # tokenizers
    actor_tokenizer = AutoTokenizer.from_pretrained(actor_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path)

    critic_model = Critic(actor_model.base_model).to(device)
    # optimizers
    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=1e-5)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=1e-5)
    # left padding
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompt_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompt_dataloader = DataLoader(prompt_dataset, batch_size=rollout_batch_size, shuffle=True)

    train()