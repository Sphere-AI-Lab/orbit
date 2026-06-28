"""Local Search-R1 retriever backed by FAISS and tar/plain JSONL corpus.

The upstream Search-R1 retrieval server loads the corpus through
``datasets.load_dataset``. On this cluster that can materialize a large Arrow
cache and hit quota limits, so this server random-accesses JSONL records by
byte offset instead.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


class JsonlOffsetCorpus:
    def __init__(self, corpus_path: str):
        self.path = Path(corpus_path)
        self._local = threading.local()
        self.base_offset, self.data_size = self._resolve_payload()
        self.offsets = self._build_offsets()

    def _resolve_payload(self) -> tuple[int, int]:
        if not tarfile.is_tarfile(self.path):
            return 0, self.path.stat().st_size

        with tarfile.open(self.path, "r") as tar:
            members = [member for member in tar.getmembers() if member.isfile()]
            if len(members) != 1:
                raise ValueError(f"Expected one JSONL member in {self.path}, got {len(members)}")
            member = members[0]
            return member.offset_data, member.size

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        end = self.base_offset + self.data_size
        with self.path.open("rb") as handle:
            handle.seek(self.base_offset)
            with tqdm(desc="Indexing corpus lines", unit="lines") as progress:
                while handle.tell() < end:
                    offset = handle.tell() - self.base_offset
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(offset)
                        progress.update(1)
        return offsets

    def _handle(self):
        handle = getattr(self._local, "handle", None)
        if handle is None:
            handle = self.path.open("rb")
            self._local.handle = handle
        return handle

    def __getitem__(self, idx: int) -> dict[str, Any]:
        handle = self._handle()
        handle.seek(self.base_offset + self.offsets[int(idx)])
        return json.loads(handle.readline())

    def __len__(self) -> int:
        return len(self.offsets)


def load_docs(corpus: JsonlOffsetCorpus, doc_idxs) -> list[dict[str, Any]]:
    return [corpus[int(idx)] for idx in doc_idxs]


def load_model(model_path: str, use_fp16: bool = False):
    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval().cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method: str = "mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError(f"Unknown pooling method: {pooling_method}")


class Encoder:
    def __init__(self, model_name: str, model_path: str, pooling_method: str, max_length: int, use_fp16: bool):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.no_grad()
    def encode(self, query_list: list[str] | str, is_query: bool = True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            query_list = [f"{prefix}{query}" for query in query_list]
        elif is_query and "bge" in self.model_name.lower():
            query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method)
        if "dpr" not in self.model_name.lower():
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        result = query_emb.detach().cpu().numpy().astype(np.float32, order="C")
        del inputs, output, query_emb
        torch.cuda.empty_cache()
        return result


class DenseRetriever:
    def __init__(self, config):
        print(f"Loading FAISS index: {config.index_path}", flush=True)
        self.index = faiss.read_index(config.index_path)
        if config.faiss_gpu:
            if hasattr(faiss, "GpuMultipleClonerOptions"):
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            else:
                print("FAISS GPU bindings are unavailable; using CPU index.", flush=True)

        print(f"Indexing corpus offsets: {config.corpus_path}", flush=True)
        self.corpus = JsonlOffsetCorpus(config.corpus_path)
        print(f"Indexed {len(self.corpus)} corpus records.", flush=True)

        self.search_lock = threading.Lock()
        self.encoder = Encoder(
            model_name=config.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def batch_search(self, query_list: list[str] | str, num: int | None = None, return_score: bool = False):
        with self.search_lock:
            return self._batch_search_locked(query_list, num=num, return_score=return_score)

    def _batch_search_locked(self, query_list: list[str] | str, num: int | None = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        return results


class Config:
    def __init__(
        self,
        retrieval_method: str,
        retrieval_topk: int,
        index_path: str,
        corpus_path: str,
        faiss_gpu: bool,
        retrieval_model_path: str,
        retrieval_pooling_method: str,
        retrieval_query_max_length: int,
        retrieval_use_fp16: bool,
        retrieval_batch_size: int,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


app = FastAPI()
retriever: DenseRetriever
config: Config


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or config.retrieval_topk
    scores = None
    if request.return_scores:
        results, scores = retriever.batch_search(request.queries, num=topk, return_score=True)
    else:
        results = retriever.batch_search(request.queries, num=topk, return_score=False)

    response = []
    for idx, single_result in enumerate(results):
        if request.return_scores:
            assert scores is not None
            response.append(
                [
                    {"document": doc, "score": score}
                    for doc, score in zip(single_result, scores[idx], strict=False)
                ]
            )
        else:
            response.append(single_result)
    return {"result": response}


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a local Search-R1 FAISS retriever.")
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", type=str, default="e5")
    parser.add_argument("--retriever_model", required=True)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--retrieval_batch_size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.retrieval_batch_size,
    )
    retriever = DenseRetriever(config)
    uvicorn.run(app, host=args.host, port=args.port)
