import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LlamaEmbedConfig:
    llama_bin: str
    model_path: str
    pooling: str = "mean"
    normalize: int = 2
    threads: int | None = None
    batch_size: int = 2048
    ubatch_size: int = 512


class LlamaEmbedder:
    def __init__(self, config: LlamaEmbedConfig) -> None:
        self.config = config

    def embed(self, texts: Iterable[str], chunk_size: int | None = None) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        if chunk_size is None or chunk_size <= 0:
            chunk_size = len(items)

        all_emb = []
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
                for t in chunk:
                    tmp.write(str(t).replace("\n", " ").strip() + "\n")
                tmp_path = tmp.name

            try:
                cmd = [
                    self.config.llama_bin,
                    "--model",
                    self.config.model_path,
                    "--file",
                    tmp_path,
                    "--pooling",
                    self.config.pooling,
                    "--embd-normalize",
                    str(self.config.normalize),
                    "--embd-output-format",
                    "json",
                    "--batch-size",
                    str(self.config.batch_size),
                    "--ubatch-size",
                    str(self.config.ubatch_size),
                ]
                if self.config.threads is not None:
                    cmd.extend(["--threads", str(self.config.threads)])

                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                raw = result.stdout.strip()
                if not raw:
                    raise RuntimeError("llama-embedding returned empty output")
                data = json.loads(raw)
                if isinstance(data, dict) and "data" in data:
                    emb = [row["embedding"] for row in data["data"]]
                elif isinstance(data, list):
                    emb = data
                else:
                    raise RuntimeError("Unexpected embedding JSON format")
                all_emb.append(np.asarray(emb, dtype=np.float32))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return np.vstack(all_emb)
