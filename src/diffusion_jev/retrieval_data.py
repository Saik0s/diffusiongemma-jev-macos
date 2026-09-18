"""Pinned public CodeSearchNet inputs, cached outside the working tree."""

import gzip
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

REPO_REVISION = "cd9a35b22aeb4187334f7018a0ee1960a7470586"
DATASET_REVISION = "68e8f0731a656fa4bd5b7c81936d95ad48a39bfe"
REPO = "https://raw.githubusercontent.com/anessbelbati/jev-rerank-bench/"
DATASET = "https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval/resolve/"
DEFAULT_CACHE = Path.home() / ".cache/diffusion-jev/retrieval-v1"


class MissingBenchmarkDependency(ValueError):
    """The optional public-data reader has not been installed."""


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str


SOURCES = (
    Source("candidates.jsonl", f"{REPO}{REPO_REVISION}/candidates/csn-python.jsonl",
           "0ea910e26a0d3289eb3c6e1ffde0d210dc42eaf59712bf379c6c9c37c544fdd8"),
    Source("corpus.parquet", f"{DATASET}{DATASET_REVISION}/"
           "python-corpus/test-00000-of-00001.parquet",
           "c3ffed7d172b9ef900d0c41a5081a42ff364f0e93ae156154bf314609709c962"),
    Source("reference.jsonl.gz", f"{REPO}{REPO_REVISION}/"
           "cache/jev-noul-batch/csn-python.present.jsonl.gz",
           "cb7837a8d3c159a8a4b42ac5c195e81e52e808734997ab4078da6e42b67073e0"),
)


class SourceRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class Candidate(SourceRecord):
    did: str
    bm25: float


class Query(SourceRecord):
    qid: str
    query: str
    relevant: dict[str, int]
    present: list[Candidate] = Field(min_length=30, max_length=30)
    n_rel_top30: int


class Document(SourceRecord):
    id: str
    text: str
    title: str


class ArchivedPrediction(SourceRecord):
    qid: str
    variant: str
    model: str
    dids: list[str]
    scores: list[float] = Field(min_length=30, max_length=30)
    ok: bool


@dataclass(frozen=True)
class RetrievalData:
    queries: list[Query]
    documents: dict[str, str]
    archived: dict[str, ArchivedPrediction]


def fetch_source(source: Source, cache: Path, *, persist: bool = True) -> bytes:
    path = cache / source.name
    try:
        if persist and path.is_file():
            data = path.read_bytes()
        else:
            with httpx.Client(follow_redirects=True, timeout=90.0) as client:
                response = client.get(source.url)
                response.raise_for_status()
                data = response.content
        if hashlib.sha256(data).hexdigest() != source.sha256:
            raise ValueError("Public benchmark source checksum mismatch")
        if persist and not path.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".download-", dir=cache) as temporary:
                pending = Path(temporary) / source.name
                pending.write_bytes(data)
                os.replace(pending, path)
        return data
    except httpx.HTTPError:
        raise ValueError("Public benchmark source download failed") from None


def validate_join(data: RetrievalData) -> None:
    seen: set[str] = set()
    for query in data.queries:
        if query.qid in seen:
            raise ValueError("Duplicate public query ID")
        seen.add(query.qid)
        ids = [candidate.did for candidate in query.present]
        if len(set(ids)) != 30 or any(did not in data.documents for did in ids):
            raise ValueError("Public candidate IDs are duplicated or missing from corpus")
        if not query.relevant or any(grade <= 0 for grade in query.relevant.values()):
            raise ValueError("Invalid relevance grades")
        hits = sum(did in query.relevant for did in ids)
        if hits != query.n_rel_top30:
            raise ValueError("Public candidate coverage disagrees with relevance labels")
        reference = data.archived.get(query.qid)
        if reference is None or not reference.ok:
            raise ValueError("Missing successful archived Jev reference")
        if reference.dids != ids or reference.variant != "present":
            raise ValueError("Archived Jev candidate identity or order differs")
        if reference.model != "jev-noul-batch":
            raise ValueError("Unexpected archived Jev protocol")
        if any(not 0 <= score <= 1 for score in reference.scores):
            raise ValueError("Invalid archived probabilities")


def load_data(limit: int = 50, cache: Path = DEFAULT_CACHE) -> RetrievalData:
    if not 1 <= limit <= 300:
        raise ValueError("Query limit must be between 1 and 300")
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        raise MissingBenchmarkDependency(
            "Install the optional data reader with: uv sync --extra benchmark"
        ) from None
    candidates = fetch_source(SOURCES[0], cache)
    corpus = fetch_source(SOURCES[1], cache)
    # Raw third-party responses are decoded in memory, never written to task artifacts.
    archive = fetch_source(SOURCES[2], cache, persist=False)
    try:
        queries = [Query.model_validate_json(line) for line in candidates.splitlines()][:limit]
        docs = TypeAdapter(list[Document]).validate_python(
            parquet.read_table(io.BytesIO(corpus)).to_pylist()
        )
        predictions = [ArchivedPrediction.model_validate_json(line)
                       for line in gzip.decompress(archive).splitlines()]
    except (ValidationError, OSError):
        raise ValueError("Public benchmark data failed schema validation") from None
    if len(queries) != limit or len({doc.id for doc in docs}) != len(docs):
        raise ValueError("Public benchmark row count or corpus identity mismatch")
    if len({item.qid for item in predictions}) != len(predictions):
        raise ValueError("Duplicate archived query ID")
    data = RetrievalData(
        queries=queries,
        documents={doc.id: (f"{doc.title}\n{doc.text}".strip() if doc.title else doc.text.strip())
                   for doc in docs},
        archived={item.qid: item for item in predictions},
    )
    validate_join(data)
    return data
