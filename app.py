from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

from modules.retriever import get_retriever
from modules.knowledge_builder import KnowledgeBuilder
from modules.vector_db import get_vector_db

load_dotenv()

app = Flask(__name__)
CORS(app)

# 创建上传文件夹
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 初始化检索器
retriever = get_retriever()
kb_builder = KnowledgeBuilder()

# ==================== 路由 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('base.html')

@app.route('/upload')
def upload_page():
    """知识库上传页面"""
    return render_template('upload.html')

@app.route('/chat')
def chat_page():
    """聊天页面"""
    return render_template('chat.html')

# ==================== API端点 ====================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件到知识库"""
    try:
        if 'file' not in request.files:
            return jsonify({'code': 400, 'msg': '缺少文件'})
        
        file = request.files['file']
        kb_type = request.form.get('type', 'docs')  # docs, qa
        
        if file.filename == '':
            return jsonify({'code': 400, 'msg': '文件名为空'})
        
        # 保存文件
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理文件
        count = 0
        if filename.endswith('.pdf'):
            count = kb_builder.process_pdf(filepath, kb_type)
        elif filename.endswith('.txt'):
            count = kb_builder.process_txt(filepath, kb_type)
        elif filename.endswith('.json'):
            count = kb_builder.process_json(filepath)
        else:
            return jsonify({'code': 400, 'msg': '不支持的文件格式'})
        
        # 保存知识库
        kb_builder.persist()
        
        return jsonify({
            'code': 200,
            'msg': f'✅ 成功导入{count}条知识',
            'count': count
        })
    
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'错误: {str(e)}'})

@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天接口 - 五层级联检索"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'code': 400, 'msg': '查询内容不能为空'})
        
        # 执行检索
        result = retriever.retrieve(query)
        
        return jsonify({
            'code': 200,
            'msg': '检索成功',
            'data': {
                'answer': result['result'],
                'source': result['source'],
                'layer': result['layer'],
                'confidence': result['confidence'],
                'contexts': result.get('contexts', [])
            }
        })
    
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'错误: {str(e)}'})

@app.route('/api/kb-stats', methods=['GET'])
def kb_stats():
    """获取知识库统计"""
    try:
        vector_db = get_vector_db()
        
        stats = {
            'query_count': vector_db.query_collection.count(),
            'qa_count': vector_db.qa_collection.count(),
            'doc_count': vector_db.doc_collection.count(),
            'total_count': (
                vector_db.query_collection.count() +
                vector_db.qa_collection.count() +
                vector_db.doc_collection.count()
            )
        }
        
        return jsonify({
            'code': 200,
            'data': stats
        })
    
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'错误: {str(e)}'})

# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'code': 404, 'msg': '页面不存在'})

@app.errorhandler(500)
def server_error(e):
    return jsonify({'code': 500, 'msg': '服务器错误'})

if __name__ == '__main__':
    print("🚀 启动金融客服RAG系统...")
    print("📍 访问: http://localhost:5000")
    app.run(debug=True, port=5000)