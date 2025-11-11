from typing import Dict, List, Tuple
from .vector_db import get_vector_db
from .bm25_retriever import get_bm25_retriever
from .llm_service import get_llm_service
import os
from dotenv import load_dotenv

load_dotenv()

class RAGRetriever:
    def __init__(self):
        self.vector_db = get_vector_db()
        self.bm25 = get_bm25_retriever()
        self.llm = get_llm_service()
        
        # 阈值配置
        self.query_threshold = float(os.getenv("QUERY_THRESHOLD", 0.90))
        self.qa_threshold = float(os.getenv("QA_THRESHOLD", 0.75))
        self.doc_threshold = float(os.getenv("DOC_THRESHOLD", 0.70))
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """五层级联检索"""
        
        print(f"\n🔍 开始检索: {query}")
        
        # 第1层：Query库检索
        print("📍 【第1层】Query库检索...")
        query_results = self.vector_db.search_query(
            query, top_k, self.query_threshold
        )
        
        if query_results and query_results[0]['similarity'] > self.query_threshold:
            print(f"✅ 【第1层】命中! 相似度: {query_results[0]['similarity']:.4f}")
            return {
                'layer': 1,
                'type': 'query',
                'result': query_results[0]['metadata'].get('answer', ''),
                'source': 'Query库',
                'confidence': query_results[0]['similarity']
            }
        
        # 第2层：QA库检索
        print("📍 【第2层】QA库检索...")
        qa_results = self.vector_db.search_qa(
            query, top_k, self.qa_threshold
        )
        
        if qa_results:
            print(f"✅ 【第2层】命中! 相似度: {qa_results[0]['similarity']:.4f}")
            
            qa_contexts = [
                f"Q: {r['metadata'].get('question', '')}\nA: {r['metadata'].get('answer', '')}"
                for r in qa_results[:top_k]
            ]
            
            answer = self.llm.generate_with_context(query, qa_contexts)
            
            return {
                'layer': 2,
                'type': 'qa',
                'result': answer,
                'source': 'QA库 + LLM',
                'confidence': qa_results[0]['similarity'],
                'contexts': qa_contexts
            }
        
        # 第3层：Doc库检索
        print("📍 【第3层】Doc库检索...")
        doc_results = self.vector_db.search_docs(
            query, top_k, self.doc_threshold
        )
        
        if doc_results:
            print(f"✅ 【第3层】命中! 相似度: {doc_results[0]['similarity']:.4f}")
            
            doc_contexts = [r['text'] for r in doc_results[:top_k]]
            answer = self.llm.generate_with_context(query, doc_contexts)
            
            return {
                'layer': 3,
                'type': 'docs',
                'result': answer,
                'source': 'Doc库 + LLM',
                'confidence': doc_results[0]['similarity'],
                'contexts': doc_contexts
            }
        
        # 第4层：BM25混合检索
        print("📍 【第4层】BM25混合检索...")
        bm25_results = self.bm25.search(query, top_k)
        
        if bm25_results:
            print(f"✅ 【第4层】命中! 得分: {bm25_results[0]['score']:.4f}")
            
            bm25_contexts = [r['text'] for r in bm25_results[:top_k]]
            answer = self.llm.generate_with_context(query, bm25_contexts)
            
            return {
                'layer': 4,
                'type': 'bm25',
                'result': answer,
                'source': 'BM25 + LLM',
                'confidence': min(bm25_results[0]['score'] / 100, 0.9),
                'contexts': bm25_contexts
            }
        
        # 第5层：自由生成
        print("📍 【第5层】自由生成...")
        
        free_prompt = f"""用户问题: {query}

请基于你的知识进行回答。如果你不确定答案，请告诉用户。"""
        
        answer = self.llm.generate(free_prompt)
        
        return {
            'layer': 5,
            'type': 'free',
            'result': answer,
            'source': '自由生成',
            'confidence': 0.5
        }


# 全局实例
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever