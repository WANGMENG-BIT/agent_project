"""Build a retriever and query relevant chunks from the vector store."""

import os

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma

from rag.vector_store import load_vector_store


DEFAULT_TOP_K = 4


def get_retriever(
    vector_store: Chroma | None = None,
    k: int | None = None,
) -> BaseRetriever:
    """把向量库转换为 Retriever，每次默认返回最相关的 4 个 Chunk。"""
    top_k = k if k is not None else int(os.getenv("RETRIEVER_TOP_K", DEFAULT_TOP_K))
    if top_k <= 0:
        raise ValueError("k 必须大于 0。")

    store = vector_store or load_vector_store()
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def retrieve_documents(
    query: str,
    retriever: BaseRetriever | None = None,
) -> list[Document]:
    """根据用户问题检索相关 Chunk。"""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query 不能为空。")

    active_retriever = retriever or get_retriever()
    return active_retriever.invoke(clean_query)


def format_documents(documents: list[Document]) -> str:
    """把检索结果格式化为便于观察的文本。"""
    if not documents:
        return "没有检索到相关内容。"

    sections = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source_name", "未知来源")
        page = document.metadata.get("page")
        page_text = f"，第 {int(page) + 1} 页" if isinstance(page, int) else ""
        sections.append(
            f"[结果 {index}｜来源：{source}{page_text}]\n{document.page_content}"
        )
    return "\n\n".join(sections)


def main() -> None:
    """单独运行本文件，验证向量检索是否正常。"""
    import argparse

    parser = argparse.ArgumentParser(description="检索本地知识库")
    parser.add_argument("query", help="需要检索的问题")
    parser.add_argument("-k", type=int, default=None, help="返回的 Chunk 数量")
    args = parser.parse_args()

    documents = retrieve_documents(args.query, get_retriever(k=args.k))
    print(format_documents(documents))


if __name__ == "__main__":
    main()

