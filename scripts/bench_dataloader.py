"""
Benchmark dataloader packing overhead on synthetic documents.

This avoids the real parquet/tokenizer cache so it can run quickly on any
checkout. It compares the current sorted-buffer make_dataloader implementation
against the previous linear-scan best-fit strategy.

Usage:
    uv run python scripts/bench_dataloader.py
"""

import argparse
from pathlib import Path
import random
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare


class SyntheticTokenizer:
    def __init__(self, bos_token_id=999):
        self.bos_token_id = bos_token_id

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        del num_threads
        if isinstance(text, list) and (not text or not isinstance(text[0], int)):
            return [self.encode(row, prepend=prepend) for row in text]
        row = list(text)
        if prepend is not None:
            row.insert(0, prepend)
        return row


def make_documents(count, min_len, max_len, seed):
    rng = random.Random(seed)
    docs = []
    for doc_id in range(count):
        length = rng.randint(min_len, max_len)
        docs.append([doc_id % 8192] * length)
    return docs


def document_batches(docs, tokenizer_batch_size):
    epoch = 1
    while True:
        for index in range(0, len(docs), tokenizer_batch_size):
            yield docs[index : index + tokenizer_batch_size], epoch
        epoch += 1


def make_legacy_dataloader(tokenizer, docs, batch_size, seq_len, buffer_size, tokenizer_batch_size):
    row_capacity = seq_len + 1
    batches = document_batches(docs, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    while True:
        all_rows = []
        for _ in range(batch_size):
            row = []
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for index, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = index
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda index: len(doc_buffer[index]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
                    pos += remaining

            all_rows.append(row[:row_capacity])

        row_array = mx.array(all_rows, dtype=mx.int32)
        inputs = row_array[:, :-1]
        targets = row_array[:, 1:]
        yield inputs, targets, epoch


def time_loader(name, loader, steps, batch_size, seq_len):
    t0 = time.perf_counter()
    for _ in range(steps):
        inputs, targets, _ = next(loader)
        mx.eval(inputs, targets)
    elapsed = time.perf_counter() - t0
    tokens = steps * batch_size * seq_len
    print(f"{name:>10}: {elapsed:.3f}s | {tokens / elapsed:,.0f} tokens/s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark synthetic dataloader packing.")
    parser.add_argument("--documents", type=int, default=20_000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = make_documents(args.documents, args.min_len, args.max_len, args.seed)
    tokenizer = SyntheticTokenizer()

    legacy_loader = make_legacy_dataloader(
        tokenizer,
        docs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        buffer_size=args.buffer_size,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )

    original_document_batches = prepare._document_batches
    try:
        prepare._document_batches = lambda split: document_batches(docs, args.tokenizer_batch_size)
        optimized_loader = prepare.make_dataloader(
            tokenizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            split="train",
            buffer_size=args.buffer_size,
        )

        legacy_elapsed = time_loader("legacy", legacy_loader, args.steps, args.batch_size, args.seq_len)
        optimized_elapsed = time_loader("optimized", optimized_loader, args.steps, args.batch_size, args.seq_len)
    finally:
        prepare._document_batches = original_document_batches

    if optimized_elapsed > 0:
        print(f"speedup:   {legacy_elapsed / optimized_elapsed:.2f}x")


if __name__ == "__main__":
    main()
