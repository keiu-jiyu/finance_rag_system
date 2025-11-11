from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Tuple
import json
import os

class BM25Retriever:
    def __init__(self):
        """初始化BM25检索器"""
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    def add_documents(self, docs: List[str], doc_ids: List[str] = None):
        """添加文档进行索引"""
        self.documents = docs
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(len(docs))]
        
        # 分词
        tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"✅ BM25已索引{len(docs)}个文档")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25搜索"""
        if self.bm25 is None:
            return []
        
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        
        # 排序
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append({
                    'doc_id': self.doc_ids[idx],
                    'text': self.documents[idx],
                    'score': float(score)
                })
        
        return results


# 全局实例
_bm25_retriever = BM25Retriever()

def get_bm25_retriever():
    return _bm25_retriever