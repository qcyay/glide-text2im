"""
Byte pair encoding utilities adapted from:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

import gzip
import json
import os
from functools import lru_cache
from typing import List, Tuple

import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    创建UTF-8字节到Unicode字符的可逆映射表。

    这个函数的主要目的是为了解决BPE（字节对编码）在Unicode字符串上工作的问题。
    我们需要为每个可能的字节（0-255）分配一个可打印的Unicode字符，以便：
    1. 避免处理UTF-8编码时的解码错误
    2. 为BPE算法提供干净的处理单位
    3. 避免映射到空白符或控制字符（这些字符会让BPE算法出错）

    背景知识：
    - 在大型数据集（如10B token）上，需要约5K个Unicode字符才能获得良好的覆盖率
    - 这对于一个32K的BPE词汇表来说是一个显著比例
    - 为避免这种情况，我们在UTF-8字节和Unicode字符串之间建立查找表

    Returns:
        dict: 字节值（0-255）到Unicode字符的映射字典

    示例:
        {33: '!', 34: '"', ..., 97: 'a', ..., 256: 'Ā', 257: 'ā', ...}
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    # 第一部分：可打印字符的直接映射
    # 创建三个可打印字符范围的列表
    bs = (
        # 1. 基本拉丁字母和常见符号：! 到 ~
        # 包括：数字、大写字母、小写字母、基本标点
        list(range(ord("!"), ord("~") + 1))
        # 2. 拉丁字母补充1：¡ 到 ¬
        # 包括：倒置标点、货币符号、音调字母
        + list(range(ord("¡"), ord("¬") + 1))
        # 3. 拉丁字母补充2：® 到 ÿ
        # 包括：注册商标、版权符号、带分音符号的字母
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    # 此时 bs 包含了所有"天然可打印"的字符编码

    # cs 初始化为 bs 的副本，稍后会进行扩展
    cs = bs[:]
    # 第二部分：处理不可打印/保留字节
    n = 0 # 计数器，用于生成扩展的Unicode编码
    # 遍历所有256个可能的字节值（0-255）
    # TODO:这个处理方式是什么意思
    for b in range(2 ** 8): # 2**8 = 256
        # 如果字节值不在可打印字符范围内
        # 这包括：
        # - 控制字符 (0-31, 127)
        # - 空白符 (如空格 32)
        # - 扩展ASCII字符 (128-255中未包含的部分)
        if b not in bs:
            bs.append(b) # 将字节值添加到映射键列表
            # 为这个字节分配一个"安全"的Unicode字符
            # 从 256 (0x100) 开始，这是基本多文种平面的开始
            # 这是为了确保不会映射到可能干扰BPE算法的字符
            cs.append(2 ** 8 + n) # 256 + n
            # 递增计数器
            n += 1
    # 第三部分：将Unicode编码转换为实际的Unicode字符
    cs = [chr(n) for n in cs] # 将编码值转换为Unicode字符
    # breakpoint()
    # 第四部分：创建并返回映射字典
    # 返回格式：{字节值: Unicode字符, ...}
    # 例如：{0: 'Ā', 1: 'ā', 2: 'Ă', ..., 33: '!', 34: '"', ...}
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    获取单词中所有相邻符号对（symbol pairs）的集合。

    这个函数是BPE（字节对编码）算法的核心辅助函数，用于：
    1. 找出单词中所有可以合并的相邻字符/子词
    2. 为BPE算法的迭代合并步骤提供候选对

    Args:
        word: tuple, 单词的符号元组，符号可以是可变长度的字符串
              BPE算法处理中，单词通常被表示为字符或子词的元组
              例如: ("h", "e", "l", "l", "o") 或 ("he", "ll", "o")

    Returns:
        set: 相邻符号对的集合，每个元素是一个元组 (prev_char, char)

    示例:
        >>> get_pairs(("h", "e", "l", "l", "o"))
        {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')}

        >>> get_pairs(("hello",))
        set()  # 只有一个元素，没有相邻对

    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 初始化一个空集合来存储符号对
    pairs = set()
    # 获取第一个符号作为前一个字符
    prev_char = word[0]
    # 遍历单词中剩余的符号（从第二个开始）
    for char in word[1:]:
        # 将当前符号对添加到集合中
        # 集合会自动去重，避免重复的符号对
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，为下一次迭代准备
        prev_char = char
    return pairs


class Encoder:
    """
   BPE（字节对编码）分词器实现，用于文本的token化和反token化。

   支持GPT-2/CLIP风格的分词，将文本转换为token ID序列，或反向解码。
   核心功能包括：BPE合并、子词分割、填充处理等。
   """
    def __init__(self, encoder, bpe_merges, errors="replace"):
        """
        初始化BPE编码器

        Args:
            encoder: dict, token到id的映射字典，如 {"hello": 123, "world": 456}
            bpe_merges: list, BPE合并规则列表，如 [("e", "n"), ("t", "h")]
            errors: str, 解码错误处理方式 ("replace"/"ignore"/"strict")
        """
        # 1. 基础映射字典
        self.encoder = encoder # token -> id 映射
        self.decoder = {v: k for k, v in self.encoder.items()} # id -> token 反向映射
        # 2. 错误处理和编码配置
        self.errors = errors  # how to handle errors in decoding # 解码错误处理策略
        # 3. 字节级编码器（处理Unicode字符）
        self.byte_encoder = bytes_to_unicode() # 字节到Unicode字符映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()} # 反向映射
        # 4. BPE合并规则排名
        # 将合并规则转换为排名字典：("e", "n") -> 0, ("t", "h") -> 1, ...
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 5. 缓存机制（提高重复token的处理速度）
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        # 6. 正则表达式模式（用于文本预分词）
        # 匹配：缩写、字母、数字、标点、空白符等
        # 它会把文本切成类似：
        # 英语缩写 's 't 等（单独成 token）
        # （可带一个前导空格的）字母串、数字串、符号串
        # 以及各种空白（包括末尾空白）
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @property
    def n_vocab(self) -> int:
        """返回词汇表大小"""
        return len(self.encoder)

    @property
    def end_token(self) -> int:
        """返回结束token的ID（通常是词汇表最后一个token）"""
        return self.n_vocab - 1

    def padded_tokens_and_mask(
        self, tokens: List[int], text_ctx: int
    ) -> Tuple[List[int], List[bool]]:
        """
        对token序列进行填充，生成固定长度的序列和注意力掩码

        Args:
            tokens: token ID列表，如 [123, 456, 789]
            text_ctx: 目标序列长度（上下文窗口大小）

        Returns:
            tuple: (填充后的tokens, 注意力掩码)

        示例:
            tokens = [1, 2, 3], text_ctx = 5
            返回: ([1, 2, 3, end_token, end_token], [True, True, True, False, False])
        """
        # 1. 截断超长序列
        tokens = tokens[:text_ctx]
        # 2. 计算需要填充的长度
        padding = text_ctx - len(tokens)
        # 3. 用结束token进行填充
        padded_tokens = tokens + [self.end_token] * padding
        # 4. 生成注意力掩码（True表示真实token，False表示填充部分）
        mask = [True] * len(tokens) + [False] * padding
        return padded_tokens, mask

    def bpe(self, token):
        """
        对单个token应用BPE算法，进行子词分割

        BPE原理：迭代合并最常见的字节对，直到无法合并为止

        Args:
            token: str, 输入token（如"hello"）

        Returns:
            str: BPE分割后的结果（如"he ll o"）

        示例:
            "hello" -> "he ll o"（如果"he"和"ll"在合并规则中）
        """
        # 1. 缓存检查（避免重复计算）
        if token in self.cache:
            return self.cache[token]
        # 2. 将token转换为字符元组
        word = tuple(token) # "hello" -> ('h', 'e', 'l', 'l', 'o')
        # 3. 获取所有相邻字符对
        pairs = get_pairs(word) # 如[('h','e'), ('e','l'), ('l','l'), ('l','o')]

        # 4. 如果没有字符对（单字符token），直接返回
        if not pairs:
            return token

        # 5. 迭代合并过程
        while True:
            # 找到优先级最高的字符对（在bpe_ranks中排名最小）
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果字符对不在合并规则中，停止合并
            if bigram not in self.bpe_ranks:
                break
            # 开始合并：如合并('e','l')为'el'
            first, second = bigram
            new_word = []
            i = 0
            # 遍历当前word，寻找并合并目标字符对
            while i < len(word):
                try:
                    # 查找第一个字符的位置
                    j = word.index(first, i)
                    new_word.extend(word[i:j]) # 添加之前的部分
                    i = j
                except:  # pylint: disable=bare-except
                    new_word.extend(word[i:]) # 添加剩余部分
                    break

                # 如果找到可合并的字符对
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second) # 合并字符
                    i += 2
                else:
                    new_word.append(word[i]) # 无法合并，保留原字符
                    i += 1
            # 更新word，继续下一轮合并
            new_word = tuple(new_word)
            word = new_word
            # breakpoint()
            # 如果只剩一个子词，停止合并
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word) # 重新计算字符对
        # 6. 将元组转换为字符串，用空格分隔
        word = " ".join(word) # ('he','ll','o') -> "he ll o"
        # 7. 缓存结果
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        将文本编码为token ID序列

        Args:
            text: str, 输入文本（如"Hello world!"）

        Returns:
            list: token ID列表（如[123, 456, 789]）

        处理流程：
            1. 文本规范化（小写化）
            2. 正则预分词
            3. 字节编码
            4. BPE分割
            5. 映射到ID
        """
        # 1. 文本规范化（转为小写）
        text = text.lower()
        # 2. 存储最终的token IDs
        bpe_tokens = []
        # 3. 使用正则表达式进行预分词
        for token in re.findall(self.pat, text):
            # 4. 将token的每个字节映射为Unicode字符（避免编码问题）
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            # breakpoint()
            # 5. 应用BPE算法分割token(bpe_segmented = self.bpe(token))
            # 6. 将BPE子词映射为ID并添加到结果列表
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")) # 如"hello" -> "he ll o"
        return bpe_tokens

    def decode(self, tokens):
        """
        将token ID序列解码为文本

        Args:
            tokens: list, token ID列表

        Returns:
            str: 解码后的文本

        处理流程：
            1. ID到token的映射
            2. 拼接token
            3. 字节解码
            4. UTF-8解码
        """
        # 1. 将ID映射回token并拼接
        text = "".join([self.decoder[token] for token in tokens])
        # 2. 字节级解码：Unicode字符 -> 原始字节
        # 3. UTF-8解码为字符串，处理可能的解码错误
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text


def get_encoder():
    """
    加载并初始化CLIP/GPT-2风格的BPE分词器编码器。

    从gzip压缩的配置文件中加载编码器映射和BPE合并规则，
    构造Encoder对象用于文本token化。

    此函数通常用于加载预训练的文本编码器，如：
    - CLIP模型的文本编码器
    - GPT-2风格的分词器
    - Stable Diffusion的文本编码器

    Returns:
        Encoder: 初始化好的编码器对象，可用于文本token化

    Raises:
        FileNotFoundError: 如果配置文件不存在
        json.JSONDecodeError: 如果encoder.json文件格式错误
        UnicodeDecodeError: 如果bpe文件编码错误
    """
    # 1. 获取当前文件的绝对路径所在的目录
    # __file__: 当前.py文件的路径
    # os.path.abspath: 获取绝对路径
    # os.path.dirname: 获取父目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 加载编码器映射文件 (encoder.json.gz)
    # 这个文件包含token到id的映射关系
    # 例如: {"<|startoftext|>": 49406, "<|endoftext|>": 49407, ...}
    with gzip.open(os.path.join(root_dir, "encoder.json.gz"), "r") as f:
        # 解析JSON文件，加载token到id的映射字典
        encoder = json.load(f)
    # 3. 加载BPE合并规则文件 (vocab.bpe.gz)
    # BPE (Byte Pair Encoding) 是一种子词分词算法
    # 文件格式示例:
    #   #version: 0.2
    #   e n
    #   i t
    #   t h
    with gzip.open(os.path.join(root_dir, "vocab.bpe.gz"), "r") as f:
        # 读取gzip压缩的字节数据，解码为UTF-8字符串
        bpe_data = str(f.read(), "utf-8")
    # 4. 解析BPE合并规则
    # 原始格式: 每行是一个合并对，用空格分隔
    # 例如: "e n" 表示可以合并"e"和"n"为"en"
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    # 5. 创建并返回编码器实例
    return Encoder(
        encoder=encoder, # token到id的映射字典
        bpe_merges=bpe_merges, # BPE合并规则列表
    )
