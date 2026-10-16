"""
Haroon-2.5-HK ByteTokenizer.

Each UTF-8 byte maps to one token ID from 0 to 255.
Supports encoding and decoding UTF-8 text.
"""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be a string")

        return list(
            text.encode("utf-8", errors="replace")
        )

    def decode(self, ids):
        values = []

        for token_id in ids:
            token_id = int(token_id)

            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"Invalid token ID: {token_id}"
                )

            values.append(token_id)

        return bytes(values).decode(
            "utf-8",
            errors="replace",
        )
