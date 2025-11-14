from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any

'''
在多模态模型中, 图像通常被分割为多个patches, 每个patch通过视觉编码器(如SigLIP)转换为一个视觉token
image_pad_num控制了最终的视觉token数量, 影响模型的计算效率和表示能力。
224×224图像通常被分割为14×14=196个patches(每个16×16像素)
压缩为7×7=49个patches, 每个代表4个原始patches(196/49=4)。
'''
class VLMConfig(PretrainedConfig):
    model_type = "vlm"

    def __init__(self, lm_name, vm_name,
                 freeze_vm=True, image_pad_num=49, **kwargs):
        super().__init__(**kwargs)
        self.vm_name = vm_name
        self.lm_name = lm_name
        self.freeze_vm = freeze_vm
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)


class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vm = AutoModel.from_pretrained(config.vm_name)
        self.lm = AutoModelForCausalLM.from_pretrained(config.lm_name)
        self.processor = AutoProcessor.from_pretrained(config.vm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_name)
        self.linear1 = nn.Linear(self.vm.config.vision_config.hidden_size*4,
                                    self.lm.config.hidden_size)
        self.linear2 = nn.Linear(self.lm.config.hidden_size,
                                    self.lm.config.hidden_size)

        # 冻结预训练模型的参数, 只训练投影层
        for param in self.lm.parameters():
            param.requires_grad = False
        if config.freeze_vm:
            for param in self.vm.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        # pixel_values: [batch_size, num_channels, height, width]
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        img_embeds = self.vm.vision_model(pixel_values).last_hidden_state
        b, s, d = img_embeds.shape
        img_embeds = img_embeds.view(b, -1, d*4)    # [b, 196, d] -> [b, 49, d*4]
        img_features = self.linear2(F.silu(self.linear1(img_embeds)))
        text_embeds = text_embeds.to(img_features.dtype)
        inputs_embeds = self.merge_input_ids_with_img_features(img_features, text_embeds, input_ids)
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device))
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_input_ids_with_img_features(self, img_features, inputs_embeds, input_ids):
        num_imgs, num_img_patches, embed_dim = img_features.shape
        # 找到input_ids中<|image_pad|>的位置, 用img_features替换对应的embeddings
        batch_indices, img_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        inputs_embeds[batch_indices, img_indices] = img_features.view(-1, embed_dim)
        return inputs_embeds

def find_assistant_tokens(tokenizer, target):
    res = []
    start_idx, end_idx = 0, 0
    while start_idx <= len(target) - 1:
        if target[start_idx] != tokenizer('assistant')['input_ids'][0]:
            start_idx += 1
            end_idx += 1
        else:
            end_idx += 1
            if target[end_idx] == tokenizer('<|im_end|>')['input_ids'][0]:
                res.append((start_idx+1, end_idx+1))
                start_idx = end_idx + 1
    return res

class MyDataset(Dataset):
    def __init__(self, img_pth, data_pth, tokenizer, processor, config):
        super().__init__()
        self.img_pth = img_pth
        self.data_pth = data_pth
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(data_pth, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            img_name = sample['image']
            conversations = sample['conversations']

            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
            for conv in conversations:
                if conv['from'] == 'human':
                    messages.append({'role': 'user', 'content': conv['value']})
                else:
                    messages.append({'role': 'assistant', 'content': conv['value']})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False).replace('<image>',
                                                                                   '<|image_pad|>' * self.config.image_pad_num)
            input_ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
            indices = find_assistant_tokens(tokenizer, input_ids)

            labels = len(input_ids) * [self.tokenizer.pad_token_id]
            for idx in indices:
                labels[idx[0]:idx[1]] = input_ids[idx[0]:idx[1]]
            input_ids = input_ids[:-1]
            labels = labels[1:]
            image = Image.open(os.path.join(self.img_pth, img_name)).convert('RGB')
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            default_img = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_img, return_tensors='pt')['pixel_values'][0]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is shown in the image? <image>"},
                {"role": "assistant", "content": "The image is empty."}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
            input_ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
            indices = find_assistant_tokens(tokenizer, input_ids)
            labels = len(input_ids) * [self.tokenizer.pad_token_id]
            for idx in indices:
                labels[idx[0]:idx[1]] = input_ids[idx[0]:idx[1]]
            input_ids = input_ids[:-1]
            labels = labels[1:]
            
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f['input_ids']) for f in features)
        input_ids = []
        labels = []
        pixel_values = []
        for f in features:
            input_ids.append(f['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(f['input_ids'])))
            labels.append(f['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(f['labels'])))
            pixel_values.append(f['pixel_values'])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0)  # [1->batch_size, 3, 224, 224]
        }

if __name__ == "__main__":
    config = VLMConfig(vm_name='', lm_name='', freeze_vm=True, image_pad_num=49)
    model = VLM(config).cuda()
    img_pth = ''
    data_pth = ''
    tokenizer = AutoTokenizer.from_pretrained(config.lm_name)
    processor = AutoProcessor.from_pretrained(config.vm_name)
    output_dir = './vlm_model'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(img_pth, data_pth, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    trainer.save_state()