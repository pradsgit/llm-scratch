# BPE trainer file
import os
from typing import BinaryIO
import multiprocessing as mp
import regex as re
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
import time
import json

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
    """
    break down the docs at special tokens and pre-tokenize each doc 
    based on split pattern that splits each doc at word boundaries 
    """
    
    if not special_tokens:
        # No special tokens - process entire chunk as one document
        cleaned_docs = [chunk_text]
    else:
        # Split at special tokens
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        delimiter_pattern = "|".join(escaped_tokens)
        docs = re.split(delimiter_pattern, chunk_text)
        cleaned_docs = [item for item in docs if item.strip()]

    
    # process each doc after splitting at special tokens
    pre_tokens = {} # Always return dict of token frequencies
    compiled_pattern = re.compile(split_pattern)

    for doc in cleaned_docs:
        res = re.finditer(compiled_pattern, doc)
        for item in res:
            pre_token = item.group()
            pre_tokens[pre_token] = pre_tokens.get(pre_token, 0) + 1
    
    return pre_tokens


def worker_func_streaming_boundaries(
    file_path: str,
    start_byte: int,
    end_byte: int,
    special_tokens: list[str],
    split_pattern: str = GPT2_SPLIT_PATTERN,
    target_chunk_size: int = 100 * 1024 * 1024 # 100MB chunk streaming 
):
    """
    each process executes this func. 
    handles large chunk size in range start_byte and end_byte by streaming chunks of target_chunk_size size sequentailly
    """  
    all_token_freqs = defaultdict(int)
    total_bytes = end_byte - start_byte
    process_id = os.getpid()

    chunk_info = []  # Track chunk boundaries for validation
    
    pbar = tqdm(
        total=total_bytes,
        desc=f"Proc-{str(process_id)[-4:]}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024
    )
    
    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        current_pos = start_byte
        
        while current_pos < end_byte:
            boundary_pos = find_streaming_boundary(
                f, current_pos, end_byte, target_chunk_size
            )

            chunk_size = boundary_pos - current_pos
            chunk_info.append({
                'start': current_pos,
                'end': boundary_pos,
                'size_mb': chunk_size / (1024*1024)
            })
            
            f.seek(current_pos)

            chunk_bytes = f.read(chunk_size)
            chunk_text = chunk_bytes.decode('utf8')

            if current_pos + chunk_size < end_byte:  # Not the last chunk
                if not chunk_text.rstrip().endswith('<|endoftext|>'):
                    print(f"WARNING: Chunk doesn't end with <|endoftext|>: ...{chunk_text[-50:]}")
                
            # Get frequency dict from pre_tokenize_chunk
            token_freqs = pre_tokenize_chunk(chunk_text, split_pattern, special_tokens)
            
            for token, count in token_freqs.items():
                all_token_freqs[token] += count
            
            pbar.update(chunk_size)
            current_pos = boundary_pos
    
    pbar.close()
    total_tokens = sum(all_token_freqs.values())
    print(f"Process {process_id}: Generated {total_tokens:,} total tokens, {len(all_token_freqs):,} unique")

    print(f"Process {process_id} chunk summary:")
    for i, info in enumerate(chunk_info):
        print(f"  Chunk {i}: {info['start']:,} to {info['end']:,} ({info['size_mb']:.1f}MB)")
    
    return dict(all_token_freqs)

def find_streaming_boundary(file, start_pos, max_end_pos, target_size):
    """Find next <|endoftext|> around target_size from start_pos"""
    target_pos = min(start_pos + target_size, max_end_pos)
    split_token = b'<|endoftext|>'
    
    # If we're at the very end, just return end
    if target_pos >= max_end_pos:
        return max_end_pos
    
    # Search window: target_pos ± 50MB
    search_start = max(start_pos, target_pos - 50*1024*1024)  
    search_end = min(max_end_pos, target_pos + 50*1024*1024)
    
    # Read the search window
    file.seek(search_start)
    search_buffer = file.read(search_end - search_start)
    
    # Find all <|endoftext|> positions in this buffer
    token_positions = []
    search_pos = 0
    while True:
        found = search_buffer.find(split_token, search_pos)
        if found == -1:
            break
        token_positions.append(search_start + found + len(split_token))
        search_pos = found + len(split_token)
    
    if not token_positions:
        # No tokens found, fallback to target position
        return target_pos
    
    # Find closest token to our target
    target_absolute = target_pos
    closest_token = min(token_positions, key=lambda x: abs(x - target_absolute))
    
    return closest_token

def worker_func_simple(file_path, start_byte, end_byte, special_tokens, split_pattern=GPT2_SPLIT_PATTERN):
    # Calculate total bytes this process will handle
    total_bytes = end_byte - start_byte
    process_id = os.getpid()
    
    # Create progress bar for this specific process
    pbar = tqdm(
        total=total_bytes,
        desc=f"Process {process_id}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        position=None,  # tqdm will auto-assign position for multiple processes
        leave=True
    )
    stream_size = 100 * 1024 * 1024  # Fixed 100MB chunks
    
    with open(file_path, 'r', encoding='utf8') as f:
        f.seek(start_byte)
        current_pos = start_byte
        all_tokens = []
        
        while current_pos < end_byte:
            remaining = end_byte - current_pos
            chunk_size = min(stream_size, remaining)
            
            chunk = f.read(chunk_size)
            tokens = pre_tokenize_chunk(chunk, split_pattern, special_tokens)
            all_tokens.extend(tokens)
            
            current_pos += len(chunk.encode('utf8'))
            pbar.update(len(chunk.encode('utf8')))
        
        return all_tokens


def worker_func(
    file_path: str,
    start_byte: int,
    end_byte: int,
    special_tokens: list[str],
    split_pattern: str=GPT2_SPLIT_PATTERN
):
    """
    note: this breaks if the chunk size range is in GBs
    """
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
    # handles tie breaking condition where multiple pairs having same frequency
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

    print(f'got chunk bounds {chunk_boundaries}')

    # def format_bytes(bytes_size):
    #     """Convert bytes to human readable format"""
    #     for unit in ['B', 'KB', 'MB', 'GB']:
    #         if bytes_size < 1024.0:
    #             return f"{bytes_size:.2f} {unit}"
    #         bytes_size /= 1024.0
    #     return f"{bytes_size:.2f} TB"

    # print("Chunk sizes per process:")
    # for i in range(len(chunk_boundaries) - 1):
    #     start = chunk_boundaries[i]
    #     end = chunk_boundaries[i + 1]
    #     size_bytes = end - start
    #     print(f"Process {i}: {start:,} to {end:,} = {format_bytes(size_bytes)} ({size_bytes:,} bytes)")


    combined_args = [(input_path, start, end, special_tokens) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]

    with mp.Pool(num_processes) as pool:
        all_results = pool.starmap(worker_func_streaming_boundaries, combined_args)

    print(f'got all the processes results')

    pre_token_freqs = merge_worker_results(all_results)

    # print(f'pre-token-freqs: {pre_token_freqs}')
    # import sys; sys.exit(0)

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

    print(f'doing {max_merges} number of merges')

    # training loop
    train_start = time.time()
    for i in range(max_merges):
        merge_start = time.time()
        print(f'merge {i+1}==')
        # find the highest freq pair in pair_freqs
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

        merge_end = time.time()
        print(f'merge took about {merge_end-merge_start:.4f} secs')
        token_byte_freqs = new_token_byte_freqs

    train_end = time.time()

    print(f'training took about {train_end-train_start:.4f} secs')

    # add special tokens
    for token in special_tokens:
        curr_vocab_size = len(vocab.values())
        vocab[curr_vocab_size] = token.encode('utf8')
    
    return vocab, merges


def download_openwebtext_hf(output_path="data/openwebtext_hf.txt"):
    dataset = load_dataset("openwebtext")
    print(f"Processing {len(dataset['train']):,} documents...")
    
    with open(output_path, "w", encoding="utf8") as f:
        for doc in tqdm(dataset["train"]):
            f.write(doc["text"])
            f.write("\n<|endoftext|>\n")
    
    return output_path

def gpt2_bytes_to_unicode():
    """
    GPT-2's byte to unicode mapping.
    Ensures all 0–255 byte values are mapped to unique, printable Unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~")+1))
        + list(range(ord("¡"), ord("¬")+1))
        + list(range(ord("®"), ord("ÿ")+1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_bpe_tokenizer(
    vocab,
    merges,
    vocab_size,
    special_tokens,
    save_dir="./tokenizer_gpt2style/"
):
    os.makedirs(save_dir, exist_ok=True)

    # GPT-2 byte to unicode maps
    byte_encoder = gpt2_bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    # ---- Save vocab.json ----
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        # Convert each byte of the token into GPT-2 unicode mapping
        token_str = "".join(byte_encoder[b] for b in token_bytes)
        vocab_json[token_str] = token_id

    # Add special tokens at the end
    for token in special_tokens:
        if token not in vocab_json:
            vocab_json[token] = len(vocab_json)

    with open(f"{save_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # ---- Save merges.txt ----
    with open(f"{save_dir}/merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for (t1, t2), new_token in merges.items():
            t1_str = "".join(byte_encoder[b] for b in vocab[t1])
            t2_str = "".join(byte_encoder[b] for b in vocab[t2])
            f.write(f"{t1_str} {t2_str}\n")

    # ---- Save tokenizer_config.json ----
    config = {
        "model_type": "BPE",
        "vocab_size": vocab_size,
        "merges_file": "merges.txt",
        "vocab_file": "vocab.json",
        "special_tokens": special_tokens,
        "byte_level": True
    }
    with open(f"{save_dir}/tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Tokenizer saved to {save_dir}")



if __name__ == "__main__":

    # filename = './data/openwebtext_hf.txt'
    filename = '../../data/tinystories-valid.txt'
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    start = time.time()
    vocab, merges = train_bpe(
        filename,
        vocab_size,
        special_tokens,
    )
    end = time.time()

    print(f'train_bpe took {end - start:.4f} secs')

    print(list(vocab.items())[:20])

    save_bpe_tokenizer(vocab, merges, vocab_size, special_tokens, '../../artifacts/tokenizer')



    # with open('./data/data/openwebtext_train.txt', 'r', encoding='utf8') as f:
    #     sample = f.read(1000)
    #     print(sample)

    # dataset = load_dataset("openwebtext", trust_remote_code=True)
    # train_dataset = dataset['train']

    # filename = './data/openwebtext_hf.txt'
    # with open(filename, 'w', encoding='utf8') as f:
    #     for i, doc in enumerate(tqdm(train_dataset, total=500_001, desc="Writing docs")):
    #         f.write(doc["text"])
    #         f.write("\n<|endoftext|>\n")
    #         if i == 1_000_000:
    #             break

    # file_size = os.path.getsize(filename)
    # print(f"File size: {file_size} bytes")
    # print(f"File size: {file_size / (1024**2):.2f} MB")
    # print(f"File size: {file_size / (1024**3):.2f} GB")


    # print(f"Processing {len(dataset['train']):,} documents...")