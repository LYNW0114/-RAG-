"""基于streamlit的网页端文件上传"""
from io import BytesIO

import streamlit as st
from langchain_community.embeddings import DashScopeEmbeddings
from openpyxl import load_workbook
from PyPDF2 import PdfReader

from knowledge_base import KnowledgeBaseService

# DashScope text-embedding-v4 单次最多 10 条，当前 langchain 版本会整批提交，这里补一个分批调用
_original_embed_documents = DashScopeEmbeddings.embed_documents


def _embed_documents_in_batches(self, texts):
    return [
        vector
        for start in range(0, len(texts), 10)
        for vector in _original_embed_documents(self, texts[start:start + 10])
    ]


DashScopeEmbeddings.embed_documents = _embed_documents_in_batches


def parse_uploaded_file(file_name, raw_bytes):
    """按扩展名解析上传文件，返回纯文本"""
    suffix = file_name.rsplit(".", 1)[-1].lower()

    if suffix == "pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(raw_bytes)).pages)

    if suffix == "xlsx":
        workbook = load_workbook(BytesIO(raw_bytes), read_only=True, data_only=True)
        lines = []
        for sheet in workbook.worksheets:
            lines.append(f"工作表：{sheet.title}")
            lines += [
                ",".join(str(cell) for cell in row if cell is not None)
                for row in sheet.iter_rows(values_only=True)
                if any(cell is not None for cell in row)
            ]
        workbook.close()
        return "\n".join(lines)

    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw_bytes.decode("gbk", errors="ignore")


st.title("文件上传")

uploader_file = st.file_uploader(
    "请选择上传的文件",
    type=['txt', 'csv', 'xlsx', 'pdf'],
    accept_multiple_files=False,
)

if "server" not in st.session_state:
    st.session_state["server"] = KnowledgeBaseService()

if uploader_file is not None:
    file_name = uploader_file.name
    st.write(f"文件名：{file_name}")
    st.write(f"文件大小：{uploader_file.size / 1024:.2f}KB | 文件类型：{uploader_file.type}")

    try:
        text = parse_uploaded_file(file_name, uploader_file.getvalue()).strip()
    except Exception as exc:
        st.error(f"文件解析失败：{exc}")
        st.stop()

    if not text:
        st.warning("未从文件中提取到文本内容，请检查文件是否为扫描版或空文件")
        st.stop()

    with st.spinner("正在写入知识库..."):
        result = st.session_state["server"].upload_by_str(text, file_name)
    st.write(result)
