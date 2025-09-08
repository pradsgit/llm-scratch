from collections import defaultdict
from itertools import pairwise, islice
import os
import regex as re
from multiprocessing import Pool
from typing import BinaryIO
import base64
import json
from typing import Dict, List, Tuple
from collections.abc import Iterable, Iterator
from heapq import heappush, heappop
import numpy as np
import numpy.typing as npt
# import torch

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
    
    # pbar = tqdm(
    #     total=total_bytes,
    #     desc=f"Proc-{str(process_id)[-4:]}",
    #     unit="B",
    #     unit_scale=True,
    #     unit_divisor=1024
    # )
    
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
            
            # pbar.update(chunk_size)
            current_pos = boundary_pos
    
    # pbar.close()
    total_tokens = sum(all_token_freqs.values())
    print(f"Process {process_id}: Generated {total_tokens:,} total tokens, {len(all_token_freqs):,} unique")

    print(f"Process {process_id} chunk summary:")
    for i, info in enumerate(chunk_info):
        print(f"  Chunk {i}: {info['start']:,} to {info['end']:,} ({info['size_mb']:.1f}MB)")
    
    return dict(all_token_freqs)


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

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


def process_chunk(path: str, start: int, end: int, splitter_pattern: re.Pattern, special_tokens: list[str]) -> defaultdict[tuple, int]:
    local_word_counts = defaultdict(int)
    
    with open(path, "rb") as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors='replace')

        text_parts = [chunk_text]
        for token in special_tokens:
            # For each special token, split every existing part further
            new_parts = []
            for part in text_parts:
                new_parts.extend(part.split(token))
            text_parts = new_parts
        
        # Now text_parts contains only the text BETWEEN special tokens
        for sub_chunk in text_parts:
            if not sub_chunk:
                continue
            p_iter = splitter_pattern.finditer(sub_chunk)
            for match in p_iter:
                word_tuple = tuple(match.group().encode("utf-8"))
                local_word_counts[word_tuple] += 1
            
    return local_word_counts

class BPETrainer:

    def __init__(self):
        self.pair_to_words = defaultdict(list)
        self.pair_counts = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.vocab = {i : bytes([i]) for i in range(256)}
        self.special_tokens = []
        self.merges = []
        self.splitter = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.splitter_pattern = re.compile(self.splitter)
        self.heap = []
        self._inv_cache: dict[bytes, tuple[int, ...]] = {}
        self._inv_table = bytes.maketrans(bytes(range(256)), bytes(range(255, -1, -1)))


    def _lexinvert(self, tok: bytes) -> tuple[int, ...]:
        inv = tok.translate(self._inv_table)
        return (-len(tok), *inv)

    def save_vocab(self, file_path: str | os.PathLike) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("#\n")
            for idx in sorted(self.vocab):
                tok_b64 = base64.b64encode(self.vocab[idx]).decode("ascii")
                f.write(f"{idx}\t{tok_b64}\n")

    def save_merges(self, file_path: str | os.PathLike) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("#\n")
            for b1, b2 in self.merges:
                t1 = base64.b64encode(b1).decode("ascii")
                t2 = base64.b64encode(b2).decode("ascii")
                f.write(f"{t1} {t2}\n")

    def _pretokenize_parallel(self, data_path: str):
        num_processes = os.cpu_count()
        splitter_pattern = self.splitter_pattern

        with open(data_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

        jobs = [(data_path, start, end, self.special_tokens, splitter_pattern) for start, end in zip(boundaries[:-1], boundaries[1:])]

        with Pool(processes=num_processes) as pool:
            list_of_dicts = pool.starmap(worker_func_streaming_boundaries, jobs)

        word_counts = defaultdict(int)
        for local_dict in list_of_dicts:
            for word, count in local_dict.items():
                word_counts[word] += count

        token_byte_freqs = defaultdict(int)
        # got all pre_tokens and their freqs
        for pre_token, freq in word_counts.items():
            word_tuple = tuple(list(pre_token.encode('utf8')))
            token_byte_freqs[word_tuple] = token_byte_freqs.get(word_tuple, 0) + freq

        return token_byte_freqs



    def _lexinvert(self, tok: bytes) -> tuple[int, ...]:
        cached = self._inv_cache.get(tok)
        if cached is None:
            inv = tok.translate(self._inv_table)
            cached = self._inv_cache[tok] = (*inv, 255 - len(tok))
        return cached

    def _heap_push_pair(self, cnt: int, p0: int, p1: int) -> None:
        tok1, tok2 = self.vocab[p0], self.vocab[p1]
        heappush(
            self.heap,
            (
                -cnt,                 
                self._lexinvert(tok1),
                self._lexinvert(tok2),
                p0,                   
                p1,
            ),
        )


    def _heap_best_pair(self) -> tuple[int, int] | None:
        while self.heap:
            neg_cnt   = self.heap[0][0]     # first field
            p0, p1    = self.heap[0][-2:]   # last two fields
            real_cnt  = self.pair_counts.get((p0, p1), 0)
            if real_cnt and -neg_cnt == real_cnt:   # fresh
                return p0, p1
            heappop(self.heap)                       # stale → drop
        return None


    def _add_or_inc(self, b0: int, b1: int, delta: int = 1):
        pair = (b0, b1)
        new_cnt = self.pair_counts.get(pair, 0) + delta
        self._heap_push_pair(new_cnt, b0, b1)


    def _initialize_stats(self, input_path):
        # pretokenize
        word_counts = self._pretokenize_parallel(input_path)
        self.word_counts = word_counts
        print(len(word_counts))

        for word, count in self.word_counts.items():
            for pair in pairwise(word):
                self.pair_counts[pair] += count
                self.pair_to_words[pair].append(word)

        for (p1, p2), count in self.pair_counts.items():
            self._add_or_inc(p1, p2, 0)


    def _update_stats(self, old_word, new_word, count):
        self.word_counts[new_word] += count
        self.word_counts[old_word] -= count
        if self.word_counts[old_word] == 0:
            del self.word_counts[old_word]


        for p1, p2 in pairwise(old_word):
            pair = (p1, p2)
            self._add_or_inc(p1, p2, -1*count)
            self.pair_counts[pair] -= count


        for p1, p2 in pairwise(new_word):
            pair = (p1, p2)
            self._add_or_inc(p1, p2, count)
            self.pair_counts[pair] += count
            self.pair_to_words[pair].append(new_word)


    def train(self, input_path: str, vocab_size: int = 259, special_tokens: list[str] = [], verbose: bool = False):

        self.special_tokens = special_tokens
        for token_str in special_tokens:
            if token_str.encode("utf-8") not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_str.encode("utf-8")
        self._initialize_stats(input_path)

        # print(f'word_counts: {self.word_counts}')
        # print(f'pair_counts: {self.pair_counts}')
        # import sys; sys.exit(0)



        while len(self.vocab) < vocab_size:
            merge_start = time.time()
            if verbose:
                print(f"Vocab size: {len(self.vocab)}")
    
            if not self.pair_counts:
                break

            best_pair = self._heap_best_pair()


            if self.pair_counts[best_pair] == 0:
                break    

            p1, p2 = self.vocab[best_pair[0]], self.vocab[best_pair[1]]

            self.merges.append((p1, p2))
            new_idx = len(self.vocab)
            self.vocab[new_idx] = p1 + p2
            affected_words = self.pair_to_words[best_pair].copy()

            for word in affected_words:
                if word not in self.word_counts:
                    continue
    
                count = self.word_counts[word]

                j = 0
                new_word = []
                n = len(word)
                while j < n:
                    if j < n - 1 and (word[j], word[j+1]) == best_pair:
                        new_word.append(new_idx)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
    
                self._update_stats(word, tuple(new_word), count)

            merge_end = time.time() - merge_start

            print(f'merge took {merge_end:.3f} secs')

        return self.vocab, self.merges
    


if __name__ == "__main__":
    import time

    trainer = BPETrainer()
    # filename = '../data/tinystories-valid.txt'
    filename = '../data/openwebtext_hf.txt'
    vocab_size = 4000
    special_tokens = ["<|endoftext|>"]
    start = time.time()
    trainer.train(
        input_path=filename,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        verbose=True
    )
    end = time.time() - start
    print(f'trainer took {end:.2f} secs')