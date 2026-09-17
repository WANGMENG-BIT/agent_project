"""Create and load a persistent local Chroma vector store."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
DEFAULT_COLLECTION_NAME = "local_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def _resolve_vector_db_dir(path: str | Path | None = None) -> Path:
    """将环境变量或参数中的相对路径解析到项目根目录。"""
    raw_path = path or os.getenv("VECTOR_DB_DIR") or DEFAULT_VECTOR_DB_DIR
    resolved = Path(raw_path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def get_embedding_model() -> HuggingFaceEmbeddings:
    """创建本地 Embedding 模型；模型首次运行时会下载到本机。"""
    load_dotenv(override=False)
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    device = os.getenv("EMBEDDING_DEVICE", "cpu")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_store(
    persist_directory: str | Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """连接已持久化的 Chroma；目录不存在时会创建一个空集合。"""
    load_dotenv(override=False)
    db_dir = _resolve_vector_db_dir(persist_directory)
    db_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=(
            collection_name
            or os.getenv("CHROMA_COLLECTION_NAME")
            or DEFAULT_COLLECTION_NAME
        ),
        embedding_function=get_embedding_model(),
        persist_directory=str(db_dir),
    )


def _stable_document_id(document: Document) -> str:
    """为 Chunk 生成稳定 ID，避免重复运行时重复写入相同内容。"""
    identity = "|".join(
        [
            str(document.metadata.get("source", "")),
            str(document.metadata.get("page", "")),
            str(document.metadata.get("start_index", "")),
            document.page_content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def create_vector_store(
    chunks: list[Document],
    persist_directory: str | Path | None = None,
    collection_name: str | None = None,
    reset: bool = False,
) -> Chroma:
    """向量化 Chunk 并写入本地 Chroma，返回可直接检索的向量库。"""
    if not chunks:
        raise ValueError("chunks 不能为空，请先加载并分割文档。")

    vector_store = load_vector_store(persist_directory, collection_name)

    if reset:
        vector_store.delete_collection()
        vector_store = load_vector_store(persist_directory, collection_name)

    vector_store.add_documents(
        documents=chunks,
        ids=[_stable_document_id(chunk) for chunk in chunks],
    )
    return vector_store


def main() -> None:
    """运行完整的“加载 + 分块 + 建库”流程。"""
    import argparse

    from rag.document_loader import (
        DEFAULT_KNOWLEDGE_BASE_DIR,
        load_documents,
    )
    from rag.text_splitter import split_documents

    parser = argparse.ArgumentParser(description="构建本地 Chroma 知识库索引")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_KNOWLEDGE_BASE_DIR))
    parser.add_argument(
        "--reset",
        action="store_true",
        help="先清空当前集合，再重新写入全部 Chunk",
    )
    args = parser.parse_args()

    documents = load_documents(args.path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks, reset=args.reset)

    print(f"建库完成：{len(chunks)} 个 Chunk 已写入。")
    print(f"向量库位置：{_resolve_vector_db_dir()}")
    print(f"集合中的记录数：{vector_store._collection.count()}")


if __name__ == "__main__":
    main()

