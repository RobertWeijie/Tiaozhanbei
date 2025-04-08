# 知识库查询工具 - RAG增强Prompt生成器 v1.1（优化版）
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
# 提前运行一次模型保存代码!!（只需执行一次）
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model.save('local_model/')  # 保存到本地文件夹



def load_embeddings(file_path):
    """加载嵌入数据（优化CSV解析速度）"""
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
        # 使用更高效的np.fromstring解析嵌入向量
        df['embedding'] = df['embedding'].apply(
            lambda x: np.fromstring(x[1:-1], sep=', ')
        )
    return df

def generate_query_embedding(query, model):
    """生成查询的词向量"""
    return model.encode([query])[0]

def find_similar_texts(query_vec, df, top_n=7, similarity_threshold=0.5):  # 修改阈值
    embeddings = np.stack(df['embedding'].values)
    similarities = cosine_similarity([query_vec], embeddings)[0]
    
    qualified_indices = np.where(similarities >= similarity_threshold)[0]
    
    # 调试信息
    print(f"\n[DEBUG] 候选段落数：{len(qualified_indices)}")
    print(f"[DEBUG] 相似度范围：{np.min(similarities):.2f}-{np.max(similarities):.2f}")
    
    if qualified_indices.size == 0:
        top_indices = np.argsort(-similarities)[:top_n]
    else:
        sorted_indices = np.argsort(-similarities[qualified_indices])
        top_indices = qualified_indices[sorted_indices][:top_n]
    
    return df.iloc[top_indices]

def create_rag_prompt(query, similar_texts, max_context_length=2000):  # 增加长度限制
    context_parts = []
    current_length = 0
    
    for text in similar_texts['text'].values:
        text_segment = f"相关段落：{text}"
        added_length = len(text_segment) + 2
        
        if current_length + added_length > max_context_length:
            available_length = max_context_length - current_length - 20
            if available_length > 50:
                # 按句子截断保留更多语义
                truncated = text[:available_length].rsplit('.', 1)[0] + "..."
                context_parts.append(f"相关段落：{truncated}")
            break
            
        context_parts.append(text_segment)
        current_length += added_length
    
    context = "\n\n".join(context_parts)
    return f"用户查询：{query}\n\n以下是可供参考的资料，请尽可能根据相关段落提供详细回答：\n{context}"


def main():
    # 配置参数
    EMBEDDINGS_FILE = "embeddings.parquet"
    MODEL_NAME = './local_model'  # 指向本地路径
    # MODEL_NAME = 'all-MiniLM-L6-v2'
    TOP_N = 7  # 新增返回的文本块数量
    SIMILARITY_THRESHOLD = 0.5  # 新增相似度阈值
    
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"错误：嵌入文件不存在 {EMBEDDINGS_FILE}")
        return
    
    print("加载知识库...")
    df = load_embeddings(EMBEDDINGS_FILE)
    
    print("加载词嵌入模型...")
    model = SentenceTransformer(MODEL_NAME)
    
    while True:
        query = input("\n请输入查询内容（输入'exit'退出）：")
        if query.lower() == 'exit':
            break
        
        try:
            query_vec = generate_query_embedding(query, model)
            similar_texts = find_similar_texts(
                query_vec, df, 
                top_n=TOP_N,
                similarity_threshold=SIMILARITY_THRESHOLD
            )
            
            rag_prompt = create_rag_prompt(query, similar_texts)
            
            print("\n增强后的Prompt：")
            print("--------------------------------------------------")
            print(rag_prompt)
            print("--------------------------------------------------")
            
        except Exception as e:
            print(f"错误：{str(e)}")

if __name__ == "__main__":
    main()
