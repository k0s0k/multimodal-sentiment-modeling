"""Pinned FP32 BERT encoding; no half-precision storage roundtrip."""

import numpy as np
from sentiment.cache import FrozenBertEncoder


def set_numeric_profile():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)


class FrozenEncoder:

    def __init__(self, bert_path, device, batch_size=64):
        set_numeric_profile()
        self._encoder = FrozenBertEncoder(
            str(bert_path), device=device, local_files_only=True
        )
        self.model = self._encoder.model
        self.metadata = dict(
            self._encoder.metadata,
            numeric_profile="q3_fp32_v1",
            storage_dtype="float32",
        )
        self.batch_size = int(batch_size)

    def encode(self, tokens):
        import torch

        set_numeric_profile()
        self.model.eval()
        output = np.empty((len(tokens), 50, 768), np.float32)
        with torch.autocast(device_type="cuda", enabled=False):
            for start in range(0, len(tokens), self.batch_size):
                output[start : start + self.batch_size] = self._encoder(
                    tokens[start : start + self.batch_size]
                )
        return output
