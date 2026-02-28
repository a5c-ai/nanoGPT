"""
ReasoningTokenizer: tiktoken GPT-2 wrapper with special reasoning tokens.

Special tokens use GPT-2's padding range (50257-50260):
  50257: <think>    (think_start)
  50258: </think>   (think_end)
  50259: <answer>   (answer_start)
  50260: </answer>  (answer_end)
"""

import tiktoken


# Special token definitions
THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

SPECIAL_TOKENS = {
    THINK_START: 50257,
    THINK_END: 50258,
    ANSWER_START: 50259,
    ANSWER_END: 50260,
}

SPECIAL_TOKEN_IDS = {v: k for k, v in SPECIAL_TOKENS.items()}


class ReasoningTokenizer:
    """Wraps tiktoken's GPT-2 encoder with 4 special reasoning tokens."""

    def __init__(self):
        self.base = tiktoken.get_encoding("gpt2")
        self.special_tokens = SPECIAL_TOKENS
        self.special_token_ids = SPECIAL_TOKEN_IDS
        self.n_vocab = 50261  # 50257 base + 4 special

        # Token ID accessors
        self.think_start_id = SPECIAL_TOKENS[THINK_START]
        self.think_end_id = SPECIAL_TOKENS[THINK_END]
        self.answer_start_id = SPECIAL_TOKENS[ANSWER_START]
        self.answer_end_id = SPECIAL_TOKENS[ANSWER_END]
        self.eot_id = 50256

    def encode(self, text: str) -> list[int]:
        """Encode text, recognizing special tokens as single token IDs."""
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for tok_str, tok_id in self.special_tokens.items():
                if text[i:].startswith(tok_str):
                    tokens.append(tok_id)
                    i += len(tok_str)
                    matched = True
                    break
            if not matched:
                # Find the next special token or end of string
                next_special = len(text)
                for tok_str in self.special_tokens:
                    pos = text.find(tok_str, i)
                    if pos != -1 and pos < next_special:
                        next_special = pos
                tokens.extend(self.base.encode(text[i:next_special]))
                i = next_special
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text, handling special tokens."""
        result = []
        buf = []
        for tid in token_ids:
            if tid in self.special_token_ids:
                if buf:
                    result.append(self.base.decode(buf))
                    buf = []
                result.append(self.special_token_ids[tid])
            else:
                buf.append(tid)
        if buf:
            result.append(self.base.decode(buf))
        return "".join(result)

    def encode_reasoning_example(
        self, prompt: str, thinking: str, answer: str
    ) -> list[int]:
        """Encode a full reasoning example: prompt <think>thinking</think><answer>answer</answer>"""
        tokens = self.base.encode(prompt)
        tokens.append(self.think_start_id)
        tokens.extend(self.base.encode(thinking))
        tokens.append(self.think_end_id)
        tokens.append(self.answer_start_id)
        tokens.extend(self.base.encode(answer))
        tokens.append(self.answer_end_id)
        return tokens


def get_tokenizer() -> ReasoningTokenizer:
    """Factory function returning a ReasoningTokenizer instance."""
    return ReasoningTokenizer()
