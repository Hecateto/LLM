from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from train_llm_from_scratch.dataset import DPODataset, DPODataCollator
from train import LLM, Config

class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        # 参考模型不进行梯度更新
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels=labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)
        ref_probs = mask_logits(ref_probs, labels)
        # 策略模型
        logits = model(input_ids=input_ids, labels=labels).logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)
        loss = dpo_loss(ref_probs, probs, 0.1)
        return loss

def logits_to_probs(logits, labels):
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    # probs: [batch_size, seq_len]
    log_probs = F.log_softmax(logits, dim=2)    # 将logits转换为log概率 [batch_size, seq_len, vocab_size]
    # 通过标签索引获取对应的log概率
    # gather的作用是根据索引从log_probs中提取对应位置的值
    # labels.unsqueeze(2).squeeze(1): : [batch_size, seq_len] -> [batch_size, seq_len, 1] -> [batch_size, seq_len]
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs

def mask_logits(probs, labels):
    new_logits = []
    for logit, label in zip(probs, labels):
        # logit: [seq_len]
        # 将非标签位置的log概率设为0，只保留标签位置的log概率, 并对这些log概率求和
        # unsqueeze(0)将结果的形状调整为[1]
        new_logits.append(logit[label!=0].sum().unsqueeze(0))
    return new_logits

def dpo_loss(ref_probs, probs, beta):
    # data按照chosen和rejected交替排列
    def split_probs(probs):
        len_chosen = int(len(probs) / 2)
        chosen_data = probs[:len_chosen]
        rejected_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(rejected_data)

    ref_chosen_probs, ref_rejected_probs = split_probs(ref_probs)
    chosen_probs, rejected_probs = split_probs(probs)

    pi_logratios = chosen_probs - rejected_probs
    ref_logratios = ref_chosen_probs - ref_rejected_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(logits * beta)
    # batch avg loss
    return loss.mean()


if __name__ == "__main__":
    AutoConfig.register("small_model", Config)
    AutoModelForCausalLM.register(Config, LLM)

    model_sft_path = ''
    model = AutoModelForCausalLM.from_pretrained(model_sft_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_sft_path).eval().to('cuda')

    tokenizer = AutoTokenizer.from_pretrained('./tokenizer', use_fast=True)
    data_collator = DPODataCollator(tokenizer=tokenizer, max_seq_len=512)
    args = TrainingArguments(
        output_dir='./dpo_output',
        num_train_epochs=1,
        do_train=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        # max_steps=15000,
        logging_steps=50,
        report_to='tensorboard',
        save_total_limit=3,
        bf16=True,
        learning_rate=1e-5,
        lr_scheduler_type='cosine',
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        save_safetensors=False,
        save_steps=100)

    data_dpo_path = ''
    dataset = DPODataset(data_path=data_dpo_path, tokenizer=tokenizer)
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset,
                         data_collator=data_collator, tokenizer=tokenizer)

    trainer.train()
    trainer.save_model('')
    trainer.save_state()