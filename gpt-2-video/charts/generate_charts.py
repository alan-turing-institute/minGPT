#!python3
# vim: noet:ts=2:sts=2:sw=2

# SPDX-License-Identifier: MIT
# Copyright © 2024 David Llewellyn-Jones

from transformers import GPT2LMHeadModel
import matplotlib.pyplot as plt

# See https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)

sd_hf["transformer.wpe.weight"].view(-1)[:20]

plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")
plt.savefig("out-01.png")
plt.clf()

plt.plot(sd_hf["transformer.wpe.weight"][:, 150])
plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
plt.plot(sd_hf["transformer.wpe.weight"][:, 250])
plt.savefig("out-02.png")
plt.clf()

plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300,:300], cmap="gray")
plt.savefig("out-03.png")
plt.clf()

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
result = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
for line in result:
    print(line["generated_text"])
