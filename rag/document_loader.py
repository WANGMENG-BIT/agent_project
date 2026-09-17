"""Load local knowledge-base files into LangChain ``Document`` objects."""

from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}


def _create_loader(file_path: Path):
    """根据文件扩展名创建对应的 LangChain Loader。"""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(file_path))
    if suffix in {".txt", ".md"}:
        return TextLoader(
            str(file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )
    if suffix == ".docx":
        return Docx2txtLoader(str(file_path))

    raise ValueError(
        f"暂不支持 {suffix or '无扩展名'} 文件：{file_path.name}。"
        f"当前支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def _find_supported_files(path: Path) -> list[Path]:
    """返回单个文件，或递归查找目录内所有受支持文件。"""
    if not path.exists():
        raise FileNotFoundError(f"路径不存在：{path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件类型：{path.suffix or '无扩展名'}")
        return [path]

    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file()
        and not file_path.name.startswith("~$")
        and file_path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_documents(path: str | Path = DEFAULT_KNOWLEDGE_BASE_DIR) -> list[Document]:
    """加载一个文件或目录，返回尚未分块的 ``Document`` 列表。"""
    input_path = Path(path).expanduser().resolve()
    file_paths = _find_supported_files(input_path)

    if not file_paths:
        raise ValueError(
            f"{input_path} 中没有可加载的知识库文件。"
            f"请放入：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    documents: list[Document] = []

    for file_path in file_paths:
        loader = _create_loader(file_path)
        loaded_documents = loader.load()

        for document in loaded_documents:
            document.metadata.update(
                {
                    "source": str(file_path),
                    "source_name": file_path.name,
                    "file_type": file_path.suffix.lower(),
                }
            )

        documents.extend(loaded_documents)

    return documents


def main() -> None:
    """单独运行本文件时，检查文档能否正常加载。"""
    import argparse

    parser = argparse.ArgumentParser(description="加载本地知识库文档")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_KNOWLEDGE_BASE_DIR),
        help="单个文件或知识库目录",
    )
    args = parser.parse_args()

    documents = load_documents(args.path)
    source_names = sorted({doc.metadata["source_name"] for doc in documents})

    print(f"加载成功：{len(source_names)} 个文件，{len(documents)} 个 Document。")
    print("文件：", "、".join(source_names))


if __name__ == "__main__":
    main()

