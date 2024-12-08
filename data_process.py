from utils import *
import os
from glob import glob  # 遍历文件夹下的文件
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def doc2vec():
    pass


# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

# 读取、分割文件
# 绝对路径
dir_path = os.path.join(os.path.dirname(__file__), 'data\\inputs\\')
documents = []
for file_path in glob(dir_path + '*.*'):  # *通位符，访问所有*.*格式的文件
    loader = None
    if '.csv' in file_path:
        loader = CSVLoader(file_path)
    if '.pdf' in file_path:
        loader = PyMuPDFLoader(file_path)
    if '.txt' in file_path:
        loader = TextLoader(file_path)
    if loader:
        documents += loader.load_and_split(text_splitter)
# print(documents)


# 向量化并存储
if documents:
    db = Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_model(),
        persist_directory=os.path.join(os.path.dirname(__file__), 'data\\db\\')
    )
    db.persist()

# if __name__ == "__main__":
#     doc2vec()
