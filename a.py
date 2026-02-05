import os
import torch
import gzip
import json

# a = torch.randn(5, 2, 3)
# b = torch.ones(2, 3)
# print(b[None].shape)
# c = a+b
# print(c.size())

## 分词器测试
# root_dir = 'glide_text2im/tokenizer'
# with gzip.open(os.path.join(root_dir, "encoder.json.gz"), "r") as f:
#     encoder = json.load(f)
# print(len(encoder.keys()))
# # print(encoder)
#
# with gzip.open(os.path.join(root_dir, "vocab.bpe.gz"), "r") as f:
#     # 读取gzip压缩的字节数据，解码为UTF-8字符串
#     bpe_data = str(f.read(), "utf-8")
# # print(bpe_data)
#
# bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
# bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
# # print(bpe_ranks)
#
# from glide_text2im.tokenizer.bpe import bytes_to_unicode, get_encoder
# tokenizer = get_encoder()
# # text = 'hello world'
# # text = '你好'
# text = "hello\nworld\t结束"
# print(text.encode("utf-8"))
# token = "".join(bytes_to_unicode()[b] for b in text.encode("utf-8"))
# print(token)
# print(tokenizer.encode(text))
# bpe_tokens = tokenizer.encode(text)
# print(tokenizer.decode(bpe_tokens))

## 理解字符、utf-8编码字节和Unicode字符之间的关系
# def understand_string_identity():
#     """理解Python字符串的恒等性"""
#
#     # 创建字符串的不同方式
#     str_from_literal = "e"
#     str_from_byte = bytes([101]).decode('utf-8')  # 字节101 -> 'e'
#     str_from_ord = chr(101)  # Unicode码点101 -> 'e'
#     str_from_file = "e"  # 模拟从文件读取
#
#     print("不同方式创建的字符串:")
#     print(f"1. 字面量: '{str_from_literal}' (id: {id(str_from_literal)})")
#     print(f"2. 字节解码: '{str_from_byte}' (id: {id(str_from_byte)})")
#     print(f"3. chr(101): '{str_from_ord}' (id: {id(str_from_ord)})")
#     print(f"4. 文件读取: '{str_from_file}' (id: {id(str_from_file)})")
#
#     print("\n相等性比较:")
#     print(f"1 == 2? {str_from_literal == str_from_byte}")
#     print(f"1 == 3? {str_from_literal == str_from_ord}")
#     print(f"1 == 4? {str_from_literal == str_from_file}")
#
#     print("\n同一性比较（是否同一个对象）:")
#     print(f"1 is 2? {str_from_literal is str_from_byte}")
#     print(f"1 is 3? {str_from_literal is str_from_ord}")
#     print(f"1 is 4? {str_from_literal is str_from_file}")
#
#     # 关键：对于字典索引，只关心相等性，不关心同一性
#     test_dict = {("e", "n"): 0}
#
#     print("\n字典索引测试:")
#     key1 = ("e", "n")  # 从文件读取的规则
#     key2 = (str_from_byte, "n")  # 从字节解码的字符
#
#     print(f"key1: {key1}")
#     print(f"key2: {key2}")
#     print(f"key1 == key2? {key1 == key2}")
#     print(f"key1 in test_dict? {key1 in test_dict}")
#     print(f"key2 in test_dict? {key2 in test_dict}")
#
# understand_string_identity()

# # 验证bpe中bytes_to_unicode对中文的处理
# bs = (
#         # 1. 基本拉丁字母和常见符号：! 到 ~
#         # 包括：数字、大写字母、小写字母、基本标点
#         list(range(ord("!"), ord("~") + 1))
#         # 2. 拉丁字母补充1：¡ 到 ¬
#         # 包括：倒置标点、货币符号、音调字母
#         + list(range(ord("¡"), ord("¬") + 1))
#         # 3. 拉丁字母补充2：® 到 ÿ
#         # 包括：注册商标、版权符号、带分音符号的字母
#         + list(range(ord("®"), ord("ÿ") + 1))
#     )
# print(list(b in bs for b in '你好'.encode('utf-8')))

text = "hello\nworld\t结束"
print(f"文本: {repr(text)}")