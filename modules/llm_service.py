import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

class LLMService:
    def __init__(self):
        """初始化LLM服务"""
        self.llm_type = os.getenv("LLM_TYPE", "dashscope")
        self.model = os.getenv("QWEN_MODEL", "qwen-max")
        
        if self.llm_type == "dashscope":
            self._init_dashscope()
        elif self.llm_type == "local":
            self._init_local()
    
    def _init_dashscope(self):
        """初始化阿里云通义千问"""
        try:
            import dashscope
            from dashscope import Generation
            self.dashscope = dashscope
            self.Generation = Generation
            print("✅ 阿里云通义千问已初始化")
        except ImportError:
            raise ImportError("请安装: pip install dashscope")
    
    def _init_local(self):
        """初始化本地模型（可选）"""
        print("⚠️ 本地模型需要更多配置，建议使用dashscope")
    
    def generate(self, prompt: str, max_tokens: int = 2048,
                temperature: float = 0.7) -> str:
        """生成回答"""
        
        if self.llm_type == "dashscope":
            return self._generate_dashscope(prompt, max_tokens, temperature)
        else:
            return "暂不支持本地模型"
    
    def _generate_dashscope(self, prompt: str, max_tokens: int,
                           temperature: float) -> str:
        """调用阿里云API"""
        try:
            response = self.Generation.call(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                return f"❌ API错误: {response.message}"
        
        except Exception as e:
            return f"❌ 生成失败: {str(e)}"
    
    def generate_with_context(self, query: str, contexts: List[str],
                             max_tokens: int = 2048) -> str:
        """基于上下文生成"""
        
        context_text = "\n".join(contexts)
        
        prompt = f"""请基于以下背景知识回答用户的问题。

【背景知识】
{context_text}

【用户问题】
{query}

【回答】
"""
        
        return self.generate(prompt, max_tokens)


# 全局实例
_llm_service = None

def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service