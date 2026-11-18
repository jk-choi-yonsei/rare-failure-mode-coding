"""Fetch the open-weight arms' runtime and weights, pinned.

llama.cpp is the runtime because it is the one harness that fits all three
vendors' models in the RTX 3090's 24 GB on Windows: gpt-oss ships MXFP4,
Gemma 4 ships a quantization-aware-trained q4_0, and transformers would need
40-60 GB of bf16 weights for either. Everything is pinned -- the llama.cpp
build by tag and GitHub's SHA-256 digest, each GGUF by repo revision and LFS
SHA-256 -- and lands outside OneDrive.

Usage: python openweight_fetch.py [runtime] [model_key ...]   (default: all)
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import common  # noqa: F401  (injects truststore for corporate TLS)
from huggingface_hub import hf_hub_download

LLAMA_BUILD = "b11435"  # 2026-10-06, commit 43fe9c64
LLAMA_ASSETS = {
    f"llama-{LLAMA_BUILD}-bin-win-cuda-12.4-x64.zip":
        "18ef0727457adf843fe0e6dc17da296c019c821e56b7bc5fc88db734538fbfb7",
    "cudart-llama-bin-win-cuda-12.4-x64.zip":
        "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
}
LLAMA_DIR = Path.home() / "tools" / "llama.cpp" / LLAMA_BUILD
SERVER_EXE = LLAMA_DIR / "llama-server.exe"
MODEL_DIR = Path.home() / "models" / "gguf"

# key -> (repo, revision, file, LFS sha256)
MODELS = {
    "gptoss20b": ("ggml-org/gpt-oss-20b-GGUF",
                  "ef9b12f2ff56c69cf32153a02784e7a3c88bf524",
                  "gpt-oss-20b-MXFP4.gguf",
                  "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901"),
    "qwen38_27b": ("unsloth/Qwen3.8-27B-GGUF",
                   "4ca720788d1e01f1bff70c033e0d0028fd02e502",
                   "Qwen3.8-27B-UD-Q4_K_M.gguf",
                   "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"),
    "gemma4_31b": ("google/gemma-4-31B-it-qat-q4_0-gguf",
                   "59dde24573e7e61570dba08b18a2e1fe246955ed",
                   "gemma-4-31B_q4_0-it.gguf",
                   "179cfb99212709597eae5929112cfca677e1bbf566178b479ae1da0c4772874b"),
    # Size control for the LoRA arm (notes/prereg-maude-lora-2026-10-09.md): the
    # post-trained model of the same release as the fine-tuned Qwen3-4B-Base.
    "qwen3_4b": ("Qwen/Qwen3-4B-GGUF",
                 "bc640142c66e1fdd12af0bd68f40445458f3869b",
                 "Qwen3-4B-Q8_0.gguf",
                 "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300"),
}


def model_path(key):
    repo, _, fname, _ = MODELS[key]
    return MODEL_DIR / repo.replace("/", "__") / fname


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def fetch_runtime():
    if SERVER_EXE.exists():
        print(f"runtime: {SERVER_EXE} present", flush=True)
        return
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/"
    for name, digest in LLAMA_ASSETS.items():
        dest = LLAMA_DIR / name
        if not dest.exists():
            print(f"runtime: downloading {name}", flush=True)
            urllib.request.urlretrieve(base + name, dest)
        if sha256(dest) != digest:
            raise SystemExit(f"{name}: SHA-256 mismatch, refusing to extract")
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(LLAMA_DIR)
        dest.unlink()
    print(f"runtime: {SERVER_EXE} ready", flush=True)


def fetch_model(key):
    repo, rev, fname, digest = MODELS[key]
    dest = model_path(key)
    if dest.exists() and sha256(dest) == digest:
        print(f"{key}: present and verified", flush=True)
        return
    print(f"{key}: downloading {repo}/{fname}@{rev[:8]}", flush=True)
    hf_hub_download(repo, fname, revision=rev, local_dir=dest.parent)
    if sha256(dest) != digest:
        raise SystemExit(f"{key}: SHA-256 mismatch")
    print(f"{key}: verified ({dest.stat().st_size / 2**30:.2f} GiB)", flush=True)


def main():
    targets = sys.argv[1:] or ["runtime", *MODELS]
    for t in targets:
        fetch_runtime() if t == "runtime" else fetch_model(t)


if __name__ == "__main__":
    main()
