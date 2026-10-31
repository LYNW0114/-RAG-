import os
import config_data as config
import hashlib
from datetime import datetime
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def check_md5(md5_str):
    if not os.path.exists(config.md5_file_path):
        open(config.md5_file_path, "w").close()
        return False
    return any(line.strip() == md5_str for line in open(config.md5_file_path, encoding="utf-8"))


def save_md5(md5_str):
    """保存md5"""
    with open(config.md5_file_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str):
    """将输入字符串进行md5"""
    return hashlib.md5(input_str.encode("utf-8")).hexdigest()


class KnowledgeBaseService():
    def __init__(self):
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self, data, file_name):
        """把私有数据推送到向量库"""
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]数据已存在"

        knowledge_chunks = self.spliter.split_text(data) if len(data) > config.max_split_char_number else [data]
        metadata = {
            "source": file_name,
            "creat_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "小梁",
        }
        self.chroma.add_texts(knowledge_chunks, metadatas=[metadata for _ in knowledge_chunks])
        save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"
