import os
import torch
import gzip
import json

# a = torch.randn(5, 2, 3)
# b = torch.ones(2, 3)
# print(b[None].shape)
# c = a+b
# print(c.size())

root_dir = 'glide_text2im/tokenizer'
with gzip.open(os.path.join(root_dir, "encoder.json.gz"), "r") as f:
    encoder = json.load(f)
print(len(encoder.keys()))
# print(encoder)

with gzip.open(os.path.join(root_dir, "vocab.bpe.gz"), "r") as f:
    # 读取gzip压缩的字节数据，解码为UTF-8字符串
    bpe_data = str(f.read(), "utf-8")
# print(bpe_data)

bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
# print(bpe_ranks)

from glide_text2im.tokenizer.bpe import bytes_to_unicode, get_encoder
tokenizer = get_encoder()
text = 'hello world'
print(text.encode("utf-8"))
token = "".join(bytes_to_unicode()[b] for b in text.encode("utf-8"))
print(token)
print(tokenizer.encode(text))
bpe_tokens = tokenizer.encode(text)
print(tokenizer.decode(bpe_tokens))