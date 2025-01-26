from typing import Tuple
from transformers import AutoTokenizer

def check_special_tokens_wrap(tokenizer: AutoTokenizer) -> Tuple[bool,bool]:
    """
    Check if a sequence is being wrapped with special tokens.

    Args:
        tokenizer: Tokenizer used for encoding

    Returns:
        Tuple of two booleans indicating if the sequence is wrapped with special tokens
    """
    start_token = False
    end_token = False

    inputs = tokenizer("a", return_tensors="pt", return_special_tokens_mask=True)

    if inputs.special_tokens_mask[0, 0] == 1:
        start_token = True
    if inputs.special_tokens_mask[0, -1] == 1:
        end_token = True

    return start_token, end_token