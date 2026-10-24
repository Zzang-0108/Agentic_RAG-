from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdf_documents(
    doc_dir: str | Path,
    base_dir: str | Path,
) -> list[Document]:
    """
    递归读取目录下的 PDF，返回包含文本和来源信息的 Document 列表。

    doc_dir: PDF 所在目录。
    base_dir: 项目根目录，用于生成相对来源路径。
    """
    doc_dir = Path(doc_dir).resolve()
    base_dir = Path(base_dir).resolve()

    if not doc_dir.is_dir():
        raise NotADirectoryError(f"目录不存在或不是文件夹：{doc_dir}")

    pdf_files = sorted(
        p for p in doc_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise FileNotFoundError(f"目录中没有 PDF：{doc_dir}")

    print(f"发现 {len(pdf_files)} 个 PDF")

    documents: list[Document] = []

    for pdf_path in pdf_files:
        print(f"正在读取：{pdf_path.name}")
        pages = PyMuPDFLoader(str(pdf_path)).load()

        kept_pages = 0

        for page_index, doc in enumerate(pages):
            if not doc.page_content.strip():
                continue

            doc.metadata["book_name"] = pdf_path.stem
            doc.metadata["source"] = (
                pdf_path.relative_to(base_dir).as_posix()
            )
            doc.metadata["page_number"] = (
                doc.metadata.get("page", page_index) + 1
            )

            documents.append(doc)
            kept_pages += 1

        print(f"  共 {len(pages)} 页，保留 {kept_pages} 个有文本的页面")

        if kept_pages == 0:
            raise ValueError(
                f"{pdf_path.name} 未提取到文本，请检查是否为扫描版 PDF"
            )

    return documents