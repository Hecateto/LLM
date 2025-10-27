from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Dict, Any
import json
import os
from torch.utils.tensorboard import SummaryWriter

'''
MTP: Multi-Token Prediction
1-token -> mutil-token
训练阶段:一次学习多个label, 预测阶段:一次生成多个token, 提升训练效率和推理性能.
DeepSeek-MTP:
训练:相对于之前的方法增加了causal chain的连接关系，同时在embedding层增加了残差链接. Teacher Forcing
推理: Predict-Verify-Accept三阶段. Free Running
原理详解可参考:https://zhuanlan.zhihu.com/p/18056041194
'''

class Config():
    def __init__(self, model = "Qwen/Qwen2-0.5B-Instruct", predict_tokens_num = 5, **kwargs):
        self.model = model
        self.predict_tokens_num = predict_tokens_num
        super().__init__(**kwargs)

class MTPModule(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.linear1 = nn.Linear(2*x, 4*x)
        self.linear2 = nn.Linear(4*x, x)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.main_model = AutoModelForCausalLM.from_pretrained(self.config.model)
        self.mtp_modules = nn.ModuleList(
            [MTPModule(self.main_model.config.hidden_size) for _ in range(self.config.predict_tokens_num - 1)]
        )
        self.output_head = nn.Linear(self.main_model.config.hidden_size, self.main_model.config.vocab_size)

    def forward_main(self, input_ids, attention_mask=None, **kwargs):
        main_hidden_output = self.main_model(input_ids, attention_mask, **kwargs).last_hidden_state
        main_head_output = self.output_head(main_hidden_output)
        return main_hidden_output, main_head_output

    def forward_mtp(self, input_ids, previous_hidden_output, head_index):
        input_embed = self.main_model.get_input_embeddings()(input_ids)
        mtp_input = torch.cat([previous_hidden_output, input_embed], dim=-1)  # B L 2H
        mtp_hidden_output = self.mtp_modules[head_index](mtp_input)
        mtp_head_output = self.output_head(mtp_hidden_output)
        return mtp_hidden_output, mtp_head_output

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = {}
        main_hidden_output, main_head_output = self.forward_main(input_ids, attention_mask, **kwargs)
        previous_hidden_output = main_hidden_output
        outputs['head_main'] = main_head_output
        for i in range(self.config.predict_tokens_num-1):
            previous_hidden_output, mtp_head_output = self.forward_mtp(input_ids, previous_hidden_output, i)
            outputs[f'mtp_head_{i}'] = mtp_head_output
        return outputs

    def generate(self, input_ids, max_length, **kwargs):
        self.eval()
        seq = input_ids.clone() # input_ids: [b, s]
        b, s = seq.size()
        N = self.config.predict_tokens_num
        with torch.no_grad():
            while seq.size(1) < max_length:
                outputs = self.forward(seq)     # running free
                print(seq.shape)
                speculative_tokens_list = []

                # main head
                logits = outputs['head_main']
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.argmax(probs, dim=-1)  # [b, ] keepdim=True则为[b, 1]
                speculative_tokens_list.append(next_token)

                # mtp heads
                for i in range(self.config.predict_tokens_num - 1):
                    logits = outputs[f'mtp_head_{i}']   # [b, s, v]
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_token = torch.argmax(probs, dim=-1)  # [b, ]
                    speculative_tokens_list.append(next_token)
                speculative_tokens = torch.stack(speculative_tokens_list, dim=1)   # [b, N]

                all_tokens = torch.cat([seq, speculative_tokens], dim=1)    # [b, s+N]

                # Verify
                _, all_logits = self.forward_main(all_tokens)
                validation_logits = all_logits[:, -N:, :] # [b, N, v]
                accept_probs = []
                for i in range(N):    # N
                    logits_i = validation_logits[:, i, :] # [b, v]
                    probs_i = torch.softmax(logits_i, dim=-1)   # [b, v]
                    token_i = speculative_tokens[:, i]    # [b, ]
                    token_prob_i = probs_i.gather(1, token_i.unsqueeze(1))    # [b, 1]
                    accept_probs.append(token_prob_i.squeeze(1))

                accept_probs = torch.cat(accept_probs, dim=1)  # [b, N]
                threshold = 1e-6
                accept_mask = (accept_probs > threshold)
                print(f'接受掩码: {accept_mask}')

                # Accept
                accept_counts = torch.full((b,), N, dtype=torch.long, device=seq.device)  # [b, ]
                for i in range(b):
                    batch_mask = accept_mask[i] # [num_rej, ]
                    reject_indices = (~batch_mask).nonzero(as_tuple=True)[0]
                    if reject_indices.numel() > 0:
                        first_reject = reject_indices[0].item()
                        accept_counts[i] = first_reject

                # # arange: [N, ] -> [1, N] -> [b, N], 每一行为0~N-1
                # arange = torch.arange(N, device=seq.device).unsqueeze(0).expand(b, N)
                # # accept_counts: [b, ] -> [b, 1]
                # accept_mask_final = arange < accept_counts.unsqueeze(1)

                new_seq_list = []
                for i in range(b):
                    accepted_num = accept_counts[i].item()
                    if accepted_num > 0:
                        accepted_tokens = speculative_tokens[i, :accepted_num]
                        new_seq = torch.cat([seq[i], accepted_tokens], dim=0)
                    else:
                        logits_fallback = outputs['head_main'][i, -1, :]
                        next_token_fallback = torch.argmax(logits_fallback, dim=-1, keepdim=True)
                        new_seq = torch.cat([seq[i], next_token_fallback], dim=0)
                    new_seq_list.append(new_seq)

                # 取最大长度并 pad, 处理后seq等长 [b, L]
                max_new_len = max(len(ns) for ns in new_seq_list)
                if max_new_len >= max_length:
                    new_seq_list = [ns[:max_length] for ns in new_seq_list]
                    seq = torch.stack(
                        [torch.cat([ns, torch.zeros(max_length-ns.size(0), dtype=ns.dtype, device=ns.device)])
                        if ns.size(0) < max_length else ns for ns in new_seq_list], dim=0)
                    break
                seq = torch.stack(
                    [torch.cat([ns, torch.zeros(max_new_len-ns.size(0), dtype=ns.dtype, device=ns.device)])
                    for ns in new_seq_list], dim=0)
                if seq.size(1) >= max_length:
                    seq = seq[:, :max_length]
                    break

        return seq

def train(config, model, dataloader, optimizer, writer, device, epochs, print_step, save_step, save_pth):
    steps = 0
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            main_hidden_output, main_head_output = model.forward_main(input_ids)    # teacher forcing

            loss_main = F.cross_entropy(
                main_head_output[:, :-1].reshape(-1, model.main_model.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )

            loss_total = loss_main

            previous_hidden_output = main_hidden_output
            for i in range(config.predict_tokens_num - 1):
                shift = i+2
                previous_hidden_output, mtp_head_output = model.forward_mtp(input_ids, previous_hidden_output, i)
                # [b*(s-shift), v]
                mtp_head_output = mtp_head_output[:, :-shift, :].reshape(-1, model.main_model.config.vocab_size)
                target = labels[:, shift:]
                target = target.contiguous().view(-1) # [b*(s-shift)]
                loss_mtp = F.cross_entropy(mtp_head_output, target, ignore_index=-100)
                loss_mtp.backward(retain_graph=True)

            loss_total.backward()
            optimizer.step()

            if (steps + 1) % print_step == 0:
                writer.add_scalar('loss_main', loss_main.item(), steps)
                writer.add_scalar('loss_mto', loss_mtp.item(), steps)
                print(
                    f"Epoch {epoch + 1}], Step {steps + 1}, loss_main: {loss_main.item():.4f}, loss_mtp: {loss_mtp.item():.4f}")

            if (steps + 1) % save_step == 0:
                torch.save(model.state_dict(), f"{save_pth}/model_{steps}.pth")

            steps += 1

class MyDataset(Dataset):
    def __init__(self, data_pth, tokenizer):
        super().__init__()
        self.data_pth = data_pth
        self.tokenizer = tokenizer
        with open(self.data_pth, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx].strip()
        sample = json.loads(sample)
        conversations = sample['conversations']
        user = conversations[0]['content']
        assistant = conversations[1]['content']
        q = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True
        )
        a = assistant + self.tokenizer.eos_token
        q_ids = self.tokenizer(q)['input_ids']
        a_ids = self.tokenizer(a)['input_ids']
        input_ids = q_ids + a_ids
        labels = [-100] * len(q_ids) + a_ids    # 只计算回答部分的loss
        return {
            'input_ids': input_ids,
            'labels': labels
        }

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f['input_ids']) for f in features)
        input_ids = []
        labels = []
        for f in features:
            input_ids.append(f['input_ids'] + self.tokenizer.pad_token_id * (max_len - len(f['input_ids'])))
            labels.append(f['labels'] + [-100] * (max_len - len(f['labels'])))
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

if __name__ == "__main__":
    writer = SummaryWriter(log_dir='./runs')
    config = Config()
    model = MTP(config)
    model.cuda()
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    data_pth = ''
    dataset = MyDataset(data_pth, tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=MyDataCollator(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_pth = './mtp'
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    train(config, model, dataloader, optimizer, writer,
          device='cuda', epochs=3, print_step=10, save_step=500, save_pth='mtp')


