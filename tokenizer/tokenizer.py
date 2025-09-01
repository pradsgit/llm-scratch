# BPE tokenizer class file
import regex as re
from train import train_bpe

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize_text(text, special_tokens: list[str], split_pattern=GPT2_SPLIT_PATTERN):
    if not special_tokens:
        # No special tokens, just apply normal pattern
        compiled_pattern = re.compile(split_pattern)
        return [match.group() for match in re.finditer(compiled_pattern, text)]
    
    escaped_tokens = [re.escape(tok) for tok in special_tokens]
    # Create pattern with capturing groups to preserve special tokens
    delimiter_pattern = f"({'|'.join(escaped_tokens)})"
    
    # Split while preserving delimiters
    segments = re.split(delimiter_pattern, text)
    
    pre_tokens = []
    compiled_pattern = re.compile(split_pattern)
    special_tokens_set = set(special_tokens)
    
    for segment in segments:
        if not segment:  # Skip empty segments
            continue
            
        if segment in special_tokens_set:
            # Special token - add as-is
            pre_tokens.append(segment)
        else:
            # Regular text - apply tokenization pattern
            for match in re.finditer(compiled_pattern, segment):
                pre_tokens.append(match.group())
    
    return pre_tokens

def get_pairs(ids: list[int]) -> list[tuple[int, int]]:
    res = []
    for pair in zip(ids[:-1], ids[1:]):
        res.append(pair)

    return res

def merge_pair(ids, pair, new_token):
    new_ids = []
    i = 0
    while i < len(ids):
        if (i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]):
            # pair found in ids, push new_token
            new_ids.append(new_token)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None # user-provided special tokens
    ):
        self.vocab = vocab.copy()
        self.merges = merges
        self.new_special_tokens = special_tokens or [] 

        # get existing special tokens
        self.existing_spec_tokens = self._get_existing_spec_tokens()
        # combine new and existing speical tokens
        self.special_tokens = self.existing_spec_tokens + self.new_special_tokens
        # add any user-defined special tokens to vocab
        self._add_special_tokens()
        self.inverse_vocab = {token: idx for idx, token in self.vocab.items()}
        self.merges_dict = self._build_merges_dict()


    def _get_existing_spec_tokens(self):
        # Find tokens that look like special tokens (e.g., surrounded by <| |>)
        special_pattern = r'<\|.*?\|>'
        res = []

        for token_bytes in self.vocab.values():
            try:
                token_str = token_bytes.decode('utf8')
                if re.match(special_pattern, token_str):
                    res.append(token_str)
            except UnicodeDecodeError:
                continue
        return res
    
    def _add_special_tokens(self):
        if not self.special_tokens:
            return
            
        # Check for existing tokens first
        existing_tokens = set(self.vocab.values())
        
        for special_token in self.new_special_tokens:
            token_bytes = special_token.encode('utf-8')
            if token_bytes in existing_tokens:
                raise ValueError(f"Special token '{special_token}' already exists in vocabulary")
        
        # Add new tokens
        max_id = max(self.vocab.keys()) if self.vocab else -1
        for i, special_token in enumerate(self.new_special_tokens):
            token_id = max_id + 1 + i
            token_bytes = special_token.encode('utf-8')
            self.vocab[token_id] = token_bytes

    def _build_merges_dict(self):
        # (int, int) -> int
        merges_dict = {}
        for merge in self.merges:
            merges_dict[(self.inverse_vocab[merge[0]], self.inverse_vocab[merge[1]])] = self.inverse_vocab[b''.join(list(merge))]
        return merges_dict

    def decode(self, ids: list[int]) -> str:
        """
        given list of integers, return decoded text
        """
        tokens = b''.join([self.vocab[idx] for idx in ids])
        text = tokens.decode('utf8', errors='replace')
        return text

    def encode(self, text: str) -> list[int]:
        """
        given text, encode and return list of integers
        """
        # step1: pre-tokenize the text
        pre_tokens = pre_tokenize_text(text, self.special_tokens, GPT2_SPLIT_PATTERN)
        all_tokens = []

        for token in pre_tokens:
            # for each pre-token, apply merge rules
            # check if token is a special token
            if token in self.special_tokens:
                all_tokens.extend([self.inverse_vocab[token.encode('utf8')]])
                continue
            ids = list(token.encode('utf8')) # ids are utf8 codepoints

            # apply merge rules to represent utf8 codepoints with vocab ids
            while len(ids) >= 2:
                pairs = get_pairs(ids)
                pair_to_merge = min(pairs, key=lambda p: self.merges_dict.get(p, float('inf')))
                if pair_to_merge not in self.merges_dict:
                    break
                # merge pair in ids
                idx = self.merges_dict[pair_to_merge]
                ids = merge_pair(ids, pair_to_merge, idx)
                
            all_tokens.extend(ids)

        return all_tokens
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        Load tokenizer from vocabulary and merges files
        
        Args:
            vocab_filepath: Path to JSON file mapping GPT-2 unicode strings to IDs
            merges_filepath: Path to text file with merge rules (one per line)
            special_tokens: Optional list of special tokens to add
        
        Returns:
            Tokenizer instance
        """
        import json
        from train import gpt2_bytes_to_unicode
        
        # GPT-2 unicode to byte decoder (reverse of byte_encoder in save function)
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        
        # Load vocabulary file
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Convert vocab from {gpt2_unicode_str: id} to {id: token_bytes}
        vocab = {}
        for token_str, token_id in vocab_data.items():
            # Check if it's a special token (no byte decoding needed)
            if token_str in (special_tokens or []):
                token_bytes = token_str.encode('utf-8')
            else:
                # Convert GPT-2 unicode string back to bytes
                token_bytes = bytes([byte_decoder[c] for c in token_str])
            vocab[token_id] = token_bytes
        
        # Load merges file
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                
                # Split line into two tokens
                parts = line.split(' ', 1)  # Split on first space only
                if len(parts) == 2:
                    # Convert GPT-2 unicode strings back to bytes
                    token1_bytes = bytes([byte_decoder[c] for c in parts[0]])
                    token2_bytes = bytes([byte_decoder[c] for c in parts[1]])
                    merges.append((token1_bytes, token2_bytes))
        
        return cls(vocab, merges, special_tokens)
        


if __name__ == "__main__":
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    vocab_path = '../artifacts/tokenizer/vocab.json'
    merges_path = '../artifacts/tokenizer/merges.txt'

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path)
    print(tokenizer.vocab)

    # vocab, merges = train_bpe(
    #     '../data/tinystories-valid.txt',
    #     vocab_size,
    #     special_tokens,
    # )
    # print(vocab)

    # merges = [(vocab[int(pair[0])], vocab[int(pair[1])]) for pair, token in merges.items()]

    text = """"Sleep well, my loves. I'll wake you up when we get home," mom said. But the man did not stop. He pulled harder and harder. At last, the bow broke. The man was not happy. The town was sad. They lost their best bow.
<|endoftext|>"""

    ids = tokenizer.encode(text)
    # print(ids)
    decoded = tokenizer.decode(ids)
    # print(decoded)
    assert text == decoded, "Failed"
