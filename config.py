import os
from dotenv import load_dotenv

load_dotenv()

# API配置
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 文件路径
PDF_PATH = os.getenv("PDF_PATH", "data/signal_and_systems.pdf")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")

# 检索配置
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
