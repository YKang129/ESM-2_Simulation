"""ESM-2 蛋白语言模型的可复用工具库。

本模块提供「计算生物学 + 人工智能药学」作业所需的核心函数：
  1. 读取 FASTA 蛋白序列与元数据 CSV；
  2. 加载 ESM-2 模型与分词器（自动配置国内下载镜像 + 复用本地缓存）；
  3. 对蛋白序列生成固定维度的嵌入向量（长序列自动切窗平均）；
  4. 计算蛋白两两之间的余弦相似度；
  5. 保存 / 加载嵌入结果，保证结果可复现。

既可以被 analysis.ipynb 导入，也可以直接在命令行运行：

    python src/embed_targets.py --fasta data/drug_targets.fasta \
        --model facebook/esm2_t6_8M_UR50D --out results/embeddings.npz
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# --------------------------------------------------------------------------- #
# 环境配置
# --------------------------------------------------------------------------- #
def setup_huggingface_env() -> None:
    """配置 HuggingFace 的国内镜像与本地缓存目录。

    1. 设置 HF_ENDPOINT 指向 hf-mirror.com，国内无需科学上网即可下载模型；
    2. 若本机 E:/hf_cache 已缓存过模型，自动复用，避免重复下载。
    对已经能直连 huggingface.co 的环境，第一条不影响正常使用。
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if "HF_HOME" not in os.environ:
        local_cache = Path("E:/hf_cache")
        if local_cache.is_dir():
            os.environ["HF_HOME"] = str(local_cache)


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def read_fasta(path: str | Path) -> Dict[str, str]:
    """读取 FASTA 文件，返回 {序列名: 序列字符串}（已去掉换行与空白）。

    FASTA 格式示例::

        >EGFR
        MRPSGTAGAAL...
        >KRAS
        MTEYKLVVVG...
    """
    records: Dict[str, str] = {}
    name: str | None = None
    chunks: List[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks)
                # 取 ">基因名" 之后的第一个空白前字段作为名称
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if name is not None:
        records[name] = "".join(chunks)
    return records


def read_metadata(path: str | Path) -> List[dict]:
    """读取 CSV 元数据，返回 list[dict]（每行一条蛋白靶点信息）。"""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def load_model(
    model_name: str, device: str = "auto"
) -> Tuple[AutoModel, AutoTokenizer, str]:
    """加载 ESM-2 模型与分词器。

    参数
    ----
    model_name : 模型标识，例如 "facebook/esm2_t6_8M_UR50D"
                 或更大的 "facebook/esm2_t33_650M_UR50D"（质量更高但更占内存）。
    device     : "auto"（有 GPU 用 GPU，否则 CPU）/ "cpu" / "cuda"。

    返回
    ----
    (model, tokenizer, device) ：model 已置于 eval 模式。
    """
    setup_huggingface_env()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, tokenizer, device


# --------------------------------------------------------------------------- #
# 嵌入计算
# --------------------------------------------------------------------------- #
def embed_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    sequences: Sequence[str],
    device: str,
    max_length: int,
) -> np.ndarray:
    """对一批序列做前向传播，并对残基级隐向量做平均池化。

    返回形状为 [批大小, 隐层维度] 的蛋白级嵌入。
    平均池化时排除 <cls> / <eos> / <pad> 三个特殊 token，只对真实氨基酸平均。
    """
    enc = tokenizer(
        list(sequences),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        hidden = model(**enc).last_hidden_state  # [B, L, D]

    input_ids = enc["input_ids"]
    special = {tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id}
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for tid in special:
        if tid is not None:
            mask &= input_ids != tid

    mask = mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, D]
    return pooled.cpu().numpy()


def embed_proteins(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    sequences: Sequence[str],
    device: str,
) -> np.ndarray:
    """把多条蛋白序列逐条嵌入为向量。

    对长度超过模型窗口的序列（例如 1620 个氨基酸的 ALK），
    切成多段分别嵌入后再取平均，避免直接截断丢掉 C 端信息。
    返回形状为 [序列数, 隐层维度] 的 NumPy 数组。
    """
    # 模型能处理的最大 token 数（含 <cls> 与 <eos>），真实残基要减 2
    max_length = int(model.config.max_position_embeddings)
    max_residues = max_length - 2

    results: List[np.ndarray] = []
    for seq in sequences:
        windows = [seq[i : i + max_residues] for i in range(0, len(seq), max_residues)]
        emb = embed_batch(model, tokenizer, windows, device, max_length=max_length)
        results.append(emb.mean(axis=0))  # 多窗口取平均 -> [D]

    return np.vstack(results)


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """计算嵌入矩阵行向量两两之间的余弦相似度，返回 [N, N] 矩阵。

    先对每行做 L2 归一化，再求内积，等价于 cos(θ)。
    """
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    return normed @ normed.T


# --------------------------------------------------------------------------- #
# 结果存取
# --------------------------------------------------------------------------- #
def save_embeddings(path: str | Path, embeddings: np.ndarray, names: Sequence[str]) -> None:
    """把嵌入与序列名一起保存为 .npz（压缩的 NumPy 格式）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, names=np.array(names))


def load_embeddings(path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """读取 .npz 中的嵌入与序列名。"""
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], list(data["names"])


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="ESM-2 蛋白嵌入生成脚本")
    parser.add_argument("--fasta", default="data/drug_targets.fasta",
                        help="输入 FASTA 路径")
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D",
                        help="ESM-2 模型标识")
    parser.add_argument("--out", default="results/embeddings.npz",
                        help="输出 .npz 路径")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    sequences = read_fasta(args.fasta)
    names = list(sequences.keys())

    model, tokenizer, device = load_model(args.model, args.device)
    print(f"[INFO] 设备={device} | 蛋白数={len(names)}")

    embeddings = embed_proteins(model, tokenizer, [sequences[n] for n in names], device)
    save_embeddings(args.out, embeddings, names)
    print(f"[INFO] 已保存 {embeddings.shape} -> {args.out}")


if __name__ == "__main__":
    main()
