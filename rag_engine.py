import os
from typing import List, Dict, Optional
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import pickle


class SignalSystemRAG:
    """信号与系统课程RAG系统 - 带页码引用功能"""
    
    def __init__(self, api_key: str, pdf_path: str, persist_dir: str = "./faiss_index"):
        self.api_key = api_key
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir
        self.vector_store = None
        self.llm = None
        
    def load_pdf_with_page_numbers(self) -> List[Document]:
        """加载PDF并保留页码信息"""
        documents = []
        
        try:
            reader = PdfReader(self.pdf_path)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "page": page_num,
                            "source": self.pdf_path
                        }
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            raise RuntimeError(f"PDF加载失败: {e}")
    
    def create_vector_store(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """创建向量数据库"""
        documents = self.load_pdf_with_page_numbers()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
            model="text-embedding-3-small"
        )
        
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.vector_store.save_local(self.persist_dir)
    
    def load_vector_store(self) -> bool:
        """加载已存在的向量数据库"""
        if os.path.exists(self.persist_dir):
            try:
                embeddings = OpenAIEmbeddings(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com/v1",
                    model="text-embedding-3-small"
                )
                self.vector_store = FAISS.load_local(
                    self.persist_dir, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            except Exception as e:
                return False
        return False
    
    def init_llm(self) -> None:
        """初始化大语言模型"""
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            max_tokens=4096,
            temperature=0.1,
        )
    
    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相关知识，返回包含页码的信息"""
        if self.vector_store is None:
            return []
        
        docs = self.vector_store.similarity_search(query, k=top_k)
        
        results = []
        for doc in docs:
            page = doc.metadata.get("page", "未知")
            results.append({
                "page": page,
                "full_content": doc.page_content
            })
        
        return results
    
    def answer_with_references(self, query: str, chat_history: List = None) -> Dict:
        """回答问题并附带页码引用"""
        # 搜索相关知识
        results = self.search_knowledge(query, top_k=5)
        
        # 构建上下文
        context_parts = []
        pages = []
        for i, result in enumerate(results):
            page = result["page"]
            content = result["full_content"]
            context_parts.append(f"【参考资料 {i+1}】（第{page}页）\n{content}\n")
            if page != "未知":
                pages.append(page)
        
        context = "\n---\n".join(context_parts) if context_parts else "未找到相关教材内容。"
        
        # 去重页码
        unique_pages = sorted(list(set([p for p in pages if p != "未知"])))
        
        # 构建提示词
        system_prompt = """你是一位《信号与系统》课程的资深教授。请根据提供的教材内容回答学生的问题。

重要要求：
1. 回答时务必引用教材的具体页码，格式为【第X页】
2. 如果回答涉及多个知识点，请分别标注每个知识点所在的页码
3. 如果教材中没有相关内容，请明确告知
4. 涉及的数学公式请使用LaTeX格式（用$...$或$$...$$包裹）
5. 回答要准确、清晰、有逻辑性

教材参考资料：
{context}
"""
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{query}")
        ])
        
        if self.llm is None:
            self.init_llm()
        
        chain = prompt_template | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "query": query,
            "chat_history": chat_history or [],
            "context": context
        })
        
        return {
            "answer": response,
            "pages": unique_pages,
            "raw_results": results
        }


@st.cache_resource
def get_rag_engine(api_key: str, pdf_path: str) -> SignalSystemRAG:
    """获取RAG引擎实例（带缓存）"""
    engine = SignalSystemRAG(api_key, pdf_path)
    
    if not engine.load_vector_store():
        # 首次运行，创建向量数据库
        with st.spinner("📚 正在构建教材知识库（首次运行需要几分钟）..."):
            engine.create_vector_store()
    
    return engine
