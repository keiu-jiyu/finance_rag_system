import os
import json
from typing import List
import PyPDF2
import jieba
from .vector_db import get_vector_db
from .bm25_retriever import get_bm25_retriever

class KnowledgeBuilder:
    def __init__(self):
        self.vector_db = get_vector_db()
        self.bm25 = get_bm25_retriever()
    
    def process_pdf(self, file_path: str, kb_type: str = 'docs') -> int:
        """处理PDF文件"""
        print(f"📄 处理PDF: {file_path}")
        
        count = 0
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    
                    # 分段处理
                    for segment in self._chunk_text(text):
                        if kb_type == 'docs':
                            self.vector_db.add_doc_document(
                                segment,
                                source=os.path.basename(file_path)
                            )
                        
                        count += 1
        
        except Exception as e:
            print(f"❌ PDF处理失败: {str(e)}")
        
        print(f"✅ 已处理{count}个文本段")
        return count
    
    def process_txt(self, file_path: str, kb_type: str = 'docs') -> int:
        """处理TXT文件"""
        print(f"📝 处理TXT: {file_path}")
        
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
                for segment in self._chunk_text(text):
                    if kb_type == 'docs':
                        self.vector_db.add_doc_document(
                            segment,
                            source=os.path.basename(file_path)
                        )
                    
                    count += 1
        
        except Exception as e:
            print(f"❌ TXT处理失败: {str(e)}")
        
        print(f"✅ 已处理{count}个文本段")
        return count
    
    def process_json(self, file_path: str) -> int:
        """处理JSON格式的QA数据"""
        print(f"📋 处理JSON: {file_path}")
        
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 支持两种格式
                if isinstance(data, list):
                    items = data
                else:
                    items = data.get('data', [])
                
                for item in items:
                    if 'question' in item and 'answer' in item:
                        self.vector_db.add_qa_document(
                            item['question'],
                            item['answer']
                        )
                        count += 1
                    elif 'query' in item and 'answer' in item:
                        # 高质量query-answer对
                        self.vector_db.add_query_document(
                            item['query'],
                            item['answer']
                        )
                        count += 1
        
        except Exception as e:
            print(f"❌ JSON处理失败: {str(e)}")
        
        print(f"✅ 已处理{count}个QA对")
        return count
    
    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500,
                   overlap: int = 50) -> List[str]:
        """文本分块"""
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def persist(self):
        """保存知识库"""
        self.vector_db.persist()
        print("✅ 知识库已保存")