"""
Simple byte-level tokenizer. Every UTF-8 byte maps to an ID 0..255.
It supports Arabic and code without a pretrained tokenizer.
"""
class ByteTokenizer:
    vocab_size = 256
    def encode(self, text):
        return list(text.encode("utf-8", errors="replace"))
    def decode(self, ids):
        return bytes([int(i) % 256 for i in ids]).decode("utf-8", errors="replace")
