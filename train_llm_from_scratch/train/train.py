import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator
from train_llm_from_scratch.dataset import LLMDataset

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.float()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)  # (1, seq_len, 1, dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q2 = (q * cos) + (rotate_half(q) * sin)
    k2 = (k * cos) + (rotate_half(k) * sin)
    return q2, k2

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) # (dim/2,)
        t = torch.arange(max_seq_len).float().unsqueeze(1)   # (max_seq_len, 1)
        freqs = t @ inv_freq.unsqueeze(0)   # (max_seq_len, dim/2)
        freqs = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)

        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)  # (1, seq_len, dim)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)

def repeat_kv(x, n_rep):
    batch, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # (batch, slen, num_key_value_heads, n_rep, head_dim)
    x = x[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return x.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.k_cache, self.v_cache = None, None
        self.is_causal = True
        self.flash_attn = self.config.flash_attn

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, use_kv_cache=False):
        b, s = x.shape[:2]
        if use_kv_cache and self.eval():
            if self.k_cache is None or self.k_cache.shape[1] != s-1:
                q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            else:
                token = x[:, -1:, :]    # (b, 1, dim)
                q = torch.cat((torch.zeros_like(x[:, :-1, :]), self.q_proj(token)), dim=1)
                k = torch.cat((self.k_cache, self.k_proj(token)), dim=1)
                v = torch.cat((self.v_cache, self.v_proj(token)), dim=1)
            self.k_cache, self.v_cache = k, v
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_key_value_heads, self.head_dim)
        v = v.view(b, s, self.num_key_value_heads, self.head_dim)

        q, k = self.rotary_emb(q, k)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q = q.transpose(1, 2)   # (b, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                    dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=self.is_causal)
        else:
            mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1) # causal mask
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + mask[:, :, :s, :s]
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, v)

        output = output.transpose(1, 2).contiguous().view(b, s, -1)
        output = self.o_proj(output)
        output = self.residual_dropout(output)
        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(self.hidden_size)
        self.post_attention_layernorm = RMSNorm(self.hidden_size)
        self.layer_idx = layer_idx

    def forward(self, x, use_kv_cache):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, use_kv_cache=use_kv_cache)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x

class Config(PretrainedConfig):
    model_type = 'small_model'
    def __init__(self,
                 hidden_size=512,
                 num_attention_heads=16,
                 num_key_value_heads=8,
                 flash_attn=True,
                 attention_bias=False,
                 max_seq_len=512,
                 intermediate_size=2048,
                 mlp_bias=False,
                 vocab_size=6400,
                 n_layers=8,
                 dropout=0.0,
                 **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        super().__init__(**kwargs)

class LLM(PreTrainedModel):
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.n_layers = self.config.n_layers

        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)

        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(self.n_layers)])

        self.norm = RMSNorm(self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.loss = None

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels, use_kv_cache=False):
        x = self.token_embeddings(input_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, use_kv_cache=use_kv_cache)
        x = self.norm(x)
        if labels is not None:
            logits = self.output(x)
            # logits: (b, s, vocab_size) -> (b*s, vocab_size)
            # labels: (b, s) -> (b*s,)
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        else:
            logits = self.output(x[:,[-1],:])
            self.loss = None

        return CausalLMOutputWithPast(
            self.loss, logits
        )

    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens,
                 temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                 use_kv_cache=True):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        s = input_ids.shape[1]
        while input_ids.shape[1] < max_new_tokens - 1:
            inference_res = self(input_ids, labels, use_kv_cache=use_kv_cache)
            logits = inference_res.logits
            logits = logits[:, -1, :]

            for token in set(input_ids.tolist()[0]):
                logits[:, token] /= repetition_penalty

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)

            if idx_next == eos:
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if stream:
                yield input_ids[:, s:]

        if not stream:
            yield input_ids[:, s:]


if __name__ == '__main__':

    config = Config(attention_bias=True, mlp_bias=True)
    model = LLM(config)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(count_parameters(model))

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("../tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./results2048',
                            num_train_epochs=10,
                            do_train=True,
                            per_device_train_batch_size=32,
                            gradient_accumulation_steps=32,
                            # max_steps=15000,
                            logging_steps=100,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True,
                            save_safetensors=False)
    dataset = LLMDataset('./dataset/pretrain_hq.jsonl', tokenizer=tokenizer, max_seq_len=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./save/model')
    trainer.save_state()