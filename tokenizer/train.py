# BPE trainer file
import os
from typing import BinaryIO
import multiprocessing as mp
import regex as re
from collections import defaultdict

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_worker_results(all_results):
    """Efficiently merge frequency dicts from all worker processes"""
    combined_frequencies = defaultdict(int)
    
    for worker_result in all_results:
        for token, count in worker_result.items():
            combined_frequencies[token] += count
    
    return dict(combined_frequencies)

def find_chunk_boundaries(
    file: BinaryIO,
    num_chunks: int, 
    split_special_token: bytes,
):
    """
    given a text file handler, return chunk boundaries byte positions as list of ints
    where each consecutive pair represent chunk byte positions that are meaningfully split at split_special_token value
    """
    # get the file size
    file.seek(0, os.SEEK_END) # go to end of file
    file_size = file.tell() # read the byte pos
    file.seek(0)

    print(f'file size is {file_size // (1024 * 1024)} MB')

    chunk_size = file_size // num_chunks

    # initial chunk boundaries spaced evenly
    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 # 4KB

    for bi in range(1, len(chunk_boundaries) - 1): # first and last byte remain unchanged
        byte_pos = chunk_boundaries[bi]
        file.seek(byte_pos)

        while True:
            # read a mini chunk at this byte pos
            mini_chunk = file.read(mini_chunk_size)

            # if EOF, this byte_pos must be at the end of file
            if mini_chunk == b'':
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # special token is found in this mini_chunk, move the byte_pos to here
                chunk_boundaries[bi] = byte_pos + found_at
                break
            byte_pos += mini_chunk_size
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize_chunk(
    chunk_text: str, 
    split_pattern: str, 
    special_tokens: list[str]
):
    """break down the docs at special tokens and pre-tokenize based on split pattern"""
    
    if not special_tokens:
        # No special tokens - process entire chunk as one document
        cleaned_docs = [chunk_text]
    else:
        # Split at special tokens
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        delimiter_pattern = "|".join(escaped_tokens)
        docs = re.split(delimiter_pattern, chunk_text)
        cleaned_docs = [item for item in docs if item.strip()]

    # Always return dict of token frequencies
    pre_tokens = {}
    compiled_pattern = re.compile(split_pattern)
    for doc in cleaned_docs:
        res = re.finditer(compiled_pattern, doc)
        for item in res:
            pre_token = item.group()
            pre_tokens[pre_token] = pre_tokens.get(pre_token, 0) + 1
    
    return pre_tokens

def worker_func(
    file_path: str,
    start_byte: int,
    end_byte: int,
    special_tokens: list[str],
    split_pattern: str=GPT2_SPLIT_PATTERN
):
    with open(file_path, 'r', encoding='utf8') as f:
        f.seek(start_byte)
        # what if the chunk size is in GB? how do we stream chunks?
        chunk = f.read(end_byte-start_byte)
        pre_tokens = pre_tokenize_chunk(chunk, split_pattern, special_tokens)

    return pre_tokens
    

def merge(word_tuple, pair, new_token):
    tokens = list(word_tuple)
    new_tokens = []
    i = 0
    while i < len(tokens):
        if (i < len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]):
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    return tuple(new_tokens)

def get_most_frequent_pair(pair_freqs, vocab):
    if not pair_freqs:
        return None
    
    # find max frequency
    max_freq = max(pair_freqs.values())
    pairs_with_same_val = [pair for pair, freq in pair_freqs.items() if freq == max_freq]

    def pair_to_byte_sequences(pair):
        # get the actual byte sequences these tokens represent
        token1_bytes = vocab[int(pair[0])]
        token2_bytes = vocab[int(pair[1])]
        return (token1_bytes, token2_bytes)

    max_pair = max(pairs_with_same_val, key=pair_to_byte_sequences)
    return int(max_pair[0]), int(max_pair[1])

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    # read the file and chunk it down to pre-tokenize before starting BPE training
    # use multi-processing for fast implementation
    num_processes = os.cpu_count() - 2

    with open(input_path, 'rb') as f:
        chunk_boundaries = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')

    combined_args = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        combined_args.append((input_path, start, end, special_tokens))

    with mp.Pool(num_processes) as pool:
        all_results = pool.starmap(worker_func, combined_args)

    pre_token_freqs = merge_worker_results(all_results)

    token_byte_freqs = {}
    # got all pre_tokens and their freqs
    for pre_token, freq in pre_token_freqs.items():
        word_tuple = tuple(list(pre_token.encode('utf8')))
        token_byte_freqs[word_tuple] = token_byte_freqs.get(word_tuple, 0) + freq
    
    # build pair freqs of all the pre-tokens?
    pair_freqs = {}
    for word_tuple, freq in token_byte_freqs.items():
        for pair in zip(word_tuple[:-1], word_tuple[1:]):
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

    base_vocab_size = 256
    max_merges = vocab_size - base_vocab_size - len(special_tokens)
    # initial vocab
    vocab = {i: bytes([i]) for i in range(base_vocab_size)} # int -> bytes
    merges = {} # (int, int) -> int

    # training loop
    for i in range(max_merges):
        # find the highest freq pair in pair_freqs
        # todo: update this to take into account tie breaking
        candidate_pair = get_most_frequent_pair(pair_freqs, vocab)
        # mint new vocab id
        new_token = base_vocab_size + i
        merges[candidate_pair] = new_token
        vocab[new_token] = vocab[candidate_pair[0]] + vocab[candidate_pair[1]]

        new_token_byte_freqs = {}
        for word_tuple, freq in token_byte_freqs.items():
            # check if pair exists
            if candidate_pair[0] in word_tuple and candidate_pair[1] in word_tuple:
                # Check if they're actually consecutive
                contains_pair = False
                for k in range(len(word_tuple) - 1):
                    if word_tuple[k] == candidate_pair[0] and word_tuple[k + 1] == candidate_pair[1]:
                        contains_pair = True
                        break

                if contains_pair:
                    merged_word = merge(word_tuple, candidate_pair, new_token)
                    new_token_byte_freqs[merged_word] = new_token_byte_freqs.get(merged_word, 0) + freq

                    # Incrementally update pair_freqs if token changed
                    if merged_word != word_tuple:
                        # remove old pairs
                        # splits = word[0].split()
                        for j in range(len(word_tuple) - 1):
                            old_pair = (word_tuple[j], word_tuple[j + 1])
                            pair_freqs[old_pair] -= freq
                            if pair_freqs[old_pair] == 0:
                                del pair_freqs[old_pair]

                        # add new pairs
                        # splits = merged_word.split()
                        for j in range(len(merged_word) - 1):
                            new_pair = (merged_word[j], merged_word[j + 1])
                            pair_freqs[new_pair] = pair_freqs.get(new_pair, 0) + freq
                
                else:
                    # Token unchanged
                    new_token_byte_freqs[word_tuple] = freq
            else:
                new_token_byte_freqs[word_tuple] = freq


        token_byte_freqs = new_token_byte_freqs

    # add special tokens
    for token in special_tokens:
        curr_vocab_size = len(vocab.values())
        vocab[curr_vocab_size] = token.encode('utf8')
    
    return vocab, merges


if __name__ == "__main__":
    vocab, _ = train_bpe(
        './data/tinystories-valid.txt',
        500,
        ["<|endoftext|>"]
    )
    print(vocab)