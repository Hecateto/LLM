from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM
import torch
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoProcessor.from_pretrained('')
tokenizer = AutoTokenizer.from_pretrained('')
AutoConfig.register("vlm", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)
model = AutoModelForCausalLM.from_pretrained('').to(device)

q_text = tokenizer.apply_chat_template(
    [
        {'role': 'system', 'content': 'You are a helpful assistant that helps people find information.'},
        {'role': 'user', 'content': 'What is shown in the image?\n<image>'}
        # mult_images: {'role': 'user', 'content': 'What is shown in the images?\n<image>\n<image>'}
    ], tokenize=False, add_generation_tokens=True).replace('<image>', '<|image_pad|>'*49)
input_ids = tokenizer(q_text, return_tensors='pt')['input_ids'].to(device)
img = Image.open('').convert('RGB')
pixel_values = processor(text=None, images=img).pixel_values.to(device)    # [1, 3, 224, 224]

model.eval()
max_new_tokens = 100
temperature = 0.0
eos = tokenizer.eos_token_id
top_k = None
s = input_ids.shape[1]
penalty_rate = 1.2

while input_ids.shape[1] < s + max_new_tokens:
    inferece_res = model(input_ids=input_ids, pixel_values=pixel_values)
    logits = inferece_res.logits[:, -1, :]  # [1, vocab_size], last token logits
    for token in set(input_ids.tolist()[0]):    # apply repetition penalty
        logits[:, token] /= penalty_rate
    if temperature == 0.0:
        _, idx_next = torch.topk(logits, k=1, dim=-1)
    else:
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1, generator=None)
    if idx_next.item() == eos:
        break
    input_ids = torch.cat([input_ids, idx_next], dim=-1)

print(tokenizer.decode(input_ids[:, s:])[0])