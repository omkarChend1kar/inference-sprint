from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained('/workspace/models/Qwen3-0.6B')
model = AutoModelForCausalLM.from_pretrained('/workspace/models/Qwen3-0.6B', dtype=torch.bfloat16).cuda()

ids = tok('The capital of France is', return_tensors='pt').input_ids.cuda()
out = model.generate(ids, max_new_tokens=10, do_sample=False)
print(tok.decode(out[0]))
