#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pathlib
from typing import List

import faiss
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from data_utils import initialize_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=pathlib.Path, required=True)
    p.add_argument("--image_base_path", required=True)
    p.add_argument("--split_value", default="1")
    p.add_argument("--split_column", default="split")
    p.add_argument("--report_column", default="impression")
    p.add_argument("--id_column", default="dicom_id")
    p.add_argument("--image_path_column", default="path")
    p.add_argument("--suffix", default=".jpg")
    p.add_argument("--target", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--hf_model", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_index", type=pathlib.Path)
    p.add_argument("--load_index", type=pathlib.Path)
    return p.parse_args()


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    model.to(device)
    model.eval()
    vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.cpu().numpy())
    return np.concatenate(vecs).astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embeddings)
    return idx


def save_index(index: faiss.Index, path: pathlib.Path):
    faiss.write_index(index, str(path))


def load_index(path: pathlib.Path) -> faiss.Index:
    return faiss.read_index(str(path))


def knn(index: faiss.Index, query: np.ndarray, corpus: List[str], ids: List[str], k: int):
    scores, idx = index.search(query, k)
    return [(ids[i], float(s), corpus[i]) for i, s in zip(idx[0], scores[0])]


def main():
    args = parse_args()
    data = initialize_data(
        args.csv_path,
        args.image_base_path,
        args.split_value,
        args.split_column,
        args.report_column,
        args.id_column,
        args.image_path_column,
        args.suffix,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModel.from_pretrained(args.hf_model)


    if args.load_index and args.load_index.exists():
        index = load_index(args.load_index)
    else:
        corpus_vecs = embed_texts(
            data["reports"],
            tokenizer,
            model,
            args.max_length,
            args.batch_size,
            device,
        )
        index = build_index(corpus_vecs)
        if args.save_index:
            save_index(index, args.save_index)
    query_vec = embed_texts([args.target], tokenizer, model, args.max_length, args.batch_size, device)
    results = knn(index, query_vec, data["reports"], data["ids"], args.k)
    for r, (i, s, t) in enumerate(results, 1):
        print(f"{r}. id={i}\tscore={s:.4f}\n{t}\n")


if __name__ == "__main__":
    main()
