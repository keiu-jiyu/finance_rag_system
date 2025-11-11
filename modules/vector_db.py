import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import numpy as np
from .embedding_model import get_embedding_model

load_dotenv()

class VectorDB:
    def __init__(self, persist_dir: str = None):
        """初始化向量数据库"""
        persist_dir = persist_dir or os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(settings)
        self.embedding_model = get_embedding_model()
        
        # 初始化三个集合
        self._init_collections()
    
    def _init_collections(self):
        """初始化三个知识库集合"""
        self.query_collection = self.client.get_or_create_collection(
            name="query_kb",
            metadata={"hnsw:space": "cosine"}
        )
        self.qa_collection = self.client.get_or_create_collection(
            name="qa_kb",
            metadata={"hnsw:space": "cosine"}
        )
        self.doc_collection = self.client.get_or_create_collection(
            name="doc_kb",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("✅ 三个知识库集合初始化完成")
    
    def add_query_document(self, query: str, answer: str, doc_id: str = None):
        """添加Query类型文档"""
        if doc_id is None:
            doc_id = f"query_{self.query_collection.count() + 1}"
        
        embedding = self.embedding_model.encode([query])[0]
        
        self.query_collection.add(
            documents=[query],
            embeddings=[embedding.tolist()],
            metadatas=[{"type": "query", "answer": answer}],
            ids=[doc_id]
        )
    
    def add_qa_document(self, question: str, answer: str, doc_id: str = None):
        """添加QA类型文档"""
        if doc_id is None:
            doc_id = f"qa_{self.qa_collection.count() + 1}"
        
        # 合并question+answer进行embedding
        combined_text = f"{question} AND {answer}"
        embedding = self.embedding_model.encode([combined_text])[0]
        
        self.qa_collection.add(
            documents=[combined_text],
            embeddings=[embedding.tolist()],
            metadatas={
                "type": "qa",
                "question": question,
                "answer": answer
            },
            ids=[doc_id]
        )
    
    def add_doc_document(self, text: str, doc_id: str = None, source: str = None):
        """添加Doc类型文档"""
        if doc_id is None:
            doc_id = f"doc_{self.doc_collection.count() + 1}"
        
        embedding = self.embedding_model.encode([text])[0]
        
        self.doc_collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas={"type": "docs", "source": source or "unknown"},
            ids=[doc_id]
        )
    
    def search_query(self, query: str, top_k: int = 5, 
                    threshold: float = 0.90) -> List[Dict]:
        """查询Query库（高阈值）"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.query_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return self._parse_results(results, threshold)
    
    def search_qa(self, query: str, top_k: int = 5,
                 threshold: float = 0.75) -> List[Dict]:
        """查询QA库（中等阈值）"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.qa_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return self._parse_results(results, threshold)
    
    def search_docs(self, query: str, top_k: int = 5,
                   threshold: float = 0.70) -> List[Dict]:
        """查询Doc库（中等阈值）"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.doc_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return self._parse_results(results, threshold)
    
    @staticmethod
    def _parse_results(results: Dict, threshold: float) -> List[Dict]:
        """解析查询结果"""
        output = []
        
        if not results['documents'][0]:
            return output
        
        for doc, distance, metadata in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ):
            # 距离转相似度
            similarity = 1 - distance
            
            if similarity >= threshold:
                output.append({
                    'text': doc,
                    'similarity': float(similarity),
                    'metadata': metadata
                })
        
        return sorted(output, key=lambda x: x['similarity'], reverse=True)
    
    def persist(self):
        """持久化数据库"""
        self.client.persist()
        print("✅ 向量库已持久化")


# 全局实例
_vector_db = None

def get_vector_db():
    """单例模式获取向量数据库"""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB()
    return _vector_db