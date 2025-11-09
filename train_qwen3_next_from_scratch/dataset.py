import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, \
    DataCollatorForTokenClassification, AutoConfig


class LLMDataset(Dataset):
    def __init__(self, data_pth, tokenizer, max_seq_len):
        super().__init__()
        self.data_pth = data_pth
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_pth, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = json.loads(line)
        text = '<s>' + line['input'].replace('<|endoftext|>', '') + line['output'] + '</s>'
        input_ids = self.tokenizer.encode(text)
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            text_len = self.max_seq_len
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
        attention_mask = [1] * text_len + [0] * (self.max_seq - text_len)
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)
        X = np.array(input_ids[:-1]).astype(np.int64)
        attention_mask = np.array(attention_mask[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        return {
            'input_ids': torch.from_numpy(X),
            'attention_mask': torch.from_numpy(attention_mask),
            'labels': torch.from_numpy(Y)
        }


class SFTDataset(Dataset):
    def __init__(self, data_pth, tokenizer, max_seq_len):
        super().__init__()
        self.data_pth = data_pth
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_pth, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        line = json.loads(line)

        instruction_text = line['instruction']
        input_text = line['input']
        output_text = line['output']
        history = line['history']

        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token
        messages = []
        if history:
            for h in history:
                messages.append({'role': 'user', 'content': h[0]})
                messages.append({'role': 'assistant', 'content': h[1]})
        messages.append({'role': 'user', 'content': query})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        input_ids = prompt_input_ids + answer_input_ids
        labels = [0] * len(prompt_input_ids) + answer_input_ids
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            text_len = self.max_seq_len
            attention_mask = [1] * text_len
        else:
            attention_mask = [1] * text_len + [0] * (self.max_seq_len - text_len)
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
            labels = labels + [-100] * (self.max_seq_len - text_len)
        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        labels = labels[:-1]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }