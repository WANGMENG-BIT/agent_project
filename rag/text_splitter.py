"""Split loaded documents into retrieval-sized chunks."""

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """创建适合中英文普通文档的递归字符分块器。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0。")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size。")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """把加载后的 Document 切成更小的 Chunk，并补充块编号元数据。"""
    if not documents:
        raise ValueError("documents 不能为空，请先加载知识库文件。")

    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    source_counters: defaultdict[tuple[str, object], int] = defaultdict(int)

    for global_index, chunk in enumerate(chunks):
        source_key = (
            str(chunk.metadata.get("source", "unknown")),
            chunk.metadata.get("page", "document"),
        )
        chunk.metadata["chunk_index"] = source_counters[source_key]
        chunk.metadata["global_chunk_index"] = global_index
        source_counters[source_key] += 1

    return chunks


def main() -> None:
    """单独运行本文件时，完成“加载 + 分块”检查。"""
    import argparse

    from rag.document_loader import (
        DEFAULT_KNOWLEDGE_BASE_DIR,
        load_documents,
    )

    parser = argparse.ArgumentParser(description="加载并分割知识库文档")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_KNOWLEDGE_BASE_DIR))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    documents = load_documents(args.path)
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)

    print(f"原始 Document：{len(documents)} 个")
    print(f"切分后 Chunk：{len(chunks)} 个")
    print(f"第一个 Chunk：\n{chunks[0].page_content[:300]}")


if __name__ == "__main__":
    main()

