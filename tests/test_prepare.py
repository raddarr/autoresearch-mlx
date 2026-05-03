import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare


class FakeTokenizer:
    bos_token_id = 999

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        del num_threads
        if isinstance(text, list) and (not text or not isinstance(text[0], int)):
            rows = [self.encode(row, prepend=prepend) for row in text]
            return rows
        row = list(text)
        if prepend is not None:
            row.insert(0, prepend)
        return row


def document_batches(docs, tokenizer_batch_size=3):
    epoch = 1
    while True:
        for index in range(0, len(docs), tokenizer_batch_size):
            yield docs[index : index + tokenizer_batch_size], epoch
        epoch += 1


def legacy_packed_rows(docs, batch_size, seq_len, buffer_size, rows_needed):
    tokenizer = FakeTokenizer()
    batches = document_batches(docs)
    row_capacity = seq_len + 1
    doc_buffer = []
    rows = []

    def refill_buffer():
        doc_batch, _ = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=tokenizer.get_bos_token_id())
        doc_buffer.extend(token_lists)

    while len(rows) < rows_needed:
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

        rows.append(row[:row_capacity])

    return rows


class MakeDataloaderTests(unittest.TestCase):
    def test_is_valid_parquet_rejects_missing_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing.parquet")
            invalid_path = os.path.join(tmpdir, "invalid.parquet")
            valid_path = os.path.join(tmpdir, "valid.parquet")

            with open(invalid_path, "wb") as handle:
                handle.write(b"not parquet")
            prepare.pq.write_table(prepare.pa.table({"text": ["hello"]}), valid_path)

            self.assertFalse(prepare.is_valid_parquet(missing_path))
            self.assertFalse(prepare.is_valid_parquet(invalid_path))
            self.assertTrue(prepare.is_valid_parquet(valid_path))

    def test_sorted_buffer_matches_legacy_best_fit_packing(self):
        docs = [
            [10, 11],
            [20, 21],
            [30, 31, 32, 33],
            [40],
            [50, 51, 52],
            [60, 61, 62, 63, 64, 65],
            [70],
            [80, 81, 82],
            [90, 91],
            [100, 101, 102, 103],
        ]
        batch_size = 3
        seq_len = 5
        buffer_size = 4
        expected_rows = legacy_packed_rows(
            docs,
            batch_size=batch_size,
            seq_len=seq_len,
            buffer_size=buffer_size,
            rows_needed=batch_size * 3,
        )

        with patch.object(prepare, "_document_batches", lambda split: document_batches(docs)):
            loader = prepare.make_dataloader(
                FakeTokenizer(),
                batch_size=batch_size,
                seq_len=seq_len,
                split="train",
                buffer_size=buffer_size,
            )
            actual_rows = []
            for _ in range(3):
                inputs, targets, _ = next(loader)
                inputs = np.array(inputs)
                targets = np.array(targets)
                self.assertEqual(inputs.shape, (batch_size, seq_len))
                self.assertEqual(targets.shape, (batch_size, seq_len))
                np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])
                actual_rows.extend(np.concatenate([inputs, targets[:, -1:]], axis=1).tolist())

        self.assertEqual(actual_rows, expected_rows)

    def test_long_documents_are_cropped_to_full_rows(self):
        docs = [
            [10, 11, 12, 13, 14, 15, 16],
            [20, 21, 22, 23, 24, 25],
            [30, 31, 32, 33, 34, 35, 36, 37],
        ]

        with patch.object(prepare, "_document_batches", lambda split: document_batches(docs, 1)):
            loader = prepare.make_dataloader(
                FakeTokenizer(),
                batch_size=2,
                seq_len=4,
                split="val",
                buffer_size=3,
            )
            inputs, targets, epoch = next(loader)
            inputs = np.array(inputs)
            targets = np.array(targets)

        self.assertGreaterEqual(epoch, 1)
        self.assertEqual(inputs.shape, (2, 4))
        self.assertEqual(targets.shape, (2, 4))
        np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])


if __name__ == "__main__":
    unittest.main()
