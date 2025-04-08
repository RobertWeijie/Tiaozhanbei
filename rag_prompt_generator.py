# rag_prompt_generator.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class RAGPromptGenerator:
    def __init__(self, 
                 embeddings_file="embeddings.parquet",
                 model_path='./local_model',
                 top_n=7,
                 similarity_threshold=0.5,
                 max_context_length=2000):
        """
        RAG增强Prompt生成器
        
        参数：
        embeddings_file: 嵌入数据文件路径
        model_path: 本地模型路径
        top_n: 最大返回段落数
        similarity_threshold: 相似度阈值
        max_context_length: 上下文最大长度
        """
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"嵌入文件不存在: {embeddings_file}")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        # 初始化配置
        self.top_n = top_n
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        
        # 加载资源
        print("加载知识库...")
        self.df = self._load_embeddings(embeddings_file)
        
        print("加载词嵌入模型...")
        self.model = SentenceTransformer(model_path)
    
    def _load_embeddings(self, file_path):
        """加载嵌入数据"""
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
            df['embedding'] = df['embedding'].apply(
                lambda x: np.fromstring(x[1:-1], sep=', ')
            )
            return df
    
    def _find_similar_texts(self, query_vec):
        """查找相似文本"""
        embeddings = np.stack(self.df['embedding'].values)
        similarities = cosine_similarity([query_vec], embeddings)[0]
        
        qualified_indices = np.where(similarities >= self.similarity_threshold)[0]
        
        if qualified_indices.size == 0:
            top_indices = np.argsort(-similarities)[:self.top_n]
        else:
            sorted_indices = np.argsort(-similarities[qualified_indices])
            top_indices = qualified_indices[sorted_indices][:self.top_n]
        
        return self.df.iloc[top_indices]
    
    def _create_prompt_context(self, similar_texts):
        """创建上下文内容"""
        context_parts = []
        current_length = 0
        
        for text in similar_texts['text'].values:
            text_segment = f"相关段落：{text}"
            added_length = len(text_segment) + 2
            
            if current_length + added_length > self.max_context_length:
                available_length = self.max_context_length - current_length - 20
                if available_length > 50:
                    truncated = text[:available_length].rsplit('.', 1)[0] + "..."
                    context_parts.append(f"相关段落：{truncated}")
                break
                
            context_parts.append(text_segment)
            current_length += added_length
        
        return "\n\n".join(context_parts)
    
    def generate_prompt(self, user_query):
        """
        生成增强后的prompt
        
        参数：
        user_query: 用户查询文本
        
        返回：
        增强后的prompt字符串
        """
        # 生成查询向量
        query_vec = self.model.encode([user_query])[0]
        
        # 查找相似文本
        similar_texts = self._find_similar_texts(query_vec)
        
        # 构建上下文
        context = self._create_prompt_context(similar_texts)
        
        # 组装最终prompt
        return (
            f"用户对话：{user_query}\n\n"
            "以下是可供参考的资料，请尽可能根据相关段落提供详细回答：\n"
            f"{context}"
        )

# 示例用法
if __name__ == "__main__":
    # 初始化生成器
    generator = RAGPromptGenerator(
        embeddings_file="embeddings.parquet",
        model_path="./local_model",
        top_n=7,
        similarity_threshold=0.5
    )
    
    # 示例查询
    test_query = "世界基本港？"
    print("生成的Prompt：\n")
    print(generator.generate_prompt(test_query))

# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import os

# # 提前运行一次模型保存代码!!（只需执行一次）
# # from sentence_transformers import SentenceTransformer
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # model.save('local_model/')  # 保存到本地文件夹


# class RAGPromptGenerator:
#     def __init__(
#         self,
#         embeddings_file: str = "embeddings.parquet",
#         model_name: str = "./local_model",
#         top_n: int = 7,
#         similarity_threshold: float = 0.5
#     ):
#         """
#         初始化RAG增强Prompt生成器
        
#         参数:
#         embeddings_file: 嵌入文件路径（支持CSV/Parquet格式）
#         model_name: 句向量模型路径或名称
#         top_n: 返回的相似段落最大数量
#         similarity_threshold: 相似度阈值（低于此值的段落会被过滤）
#         """
#         self.embeddings_file = embeddings_file
#         self.model_name = model_name
#         self.top_n = top_n
#         self.similarity_threshold = similarity_threshold
        
#         # 加载知识库数据
#         self.df = self._load_embeddings()
        
#         # 加载预训练模型
#         self.model = SentenceTransformer(self.model_name)
        
#         print("初始化完成：")
#         print(f"- 知识库文件：{embeddings_file}")
#         print(f"- 模型路径：{model_name}")
#         print(f"- 最大返回段落数：{top_n}")
#         print(f"- 相似度阈值：{similarity_threshold:.2f}")

#     def _load_embeddings(self):
#         """加载嵌入数据（优化CSV解析速度）"""
#         if not os.path.exists(self.embeddings_file):
#             raise FileNotFoundError(f"嵌入文件不存在：{self.embeddings_file}")
        
#         if self.embeddings_file.endswith('.parquet'):
#             df = pd.read_parquet(self.embeddings_file)
#         else:
#             df = pd.read_csv(self.embeddings_file)
#             # 使用更高效的np.fromstring解析嵌入向量
#             df['embedding'] = df['embedding'].apply(
#                 lambda x: np.fromstring(x[1:-1], sep=', ')
#             )
#         return df

#     def generate_prompt(self, query: str, max_context_length: int = 2000) -> str:
#         """
#         生成增强Prompt
        
#         参数:
#         query: 用户查询内容
#         max_context_length: 上下文最大长度（默认2000字符）
        
#         返回:
#         增强后的Prompt字符串
#         """
#         # 生成查询向量
#         query_vec = self._generate_query_embedding(query)
        
#         # 查找相似文本
#         similar_texts = self._find_similar_texts(query_vec)
        
#         # 创建最终Prompt
#         return self._create_rag_prompt(query, similar_texts, max_context_length)

#     def _generate_query_embedding(self, query: str) -> np.ndarray:
#         """生成查询的词向量"""
#         return self.model.encode([query])[0]

#     def _find_similar_texts(self, query_vec: np.ndarray) -> pd.DataFrame:
#         embeddings = np.stack(self.df['embedding'].values)
#         similarities = cosine_similarity([query_vec], embeddings)[0]
        
#         # 获取符合相似度阈值的索引
#         qualified_indices = np.where(similarities >= self.similarity_threshold)[0]
        
#         if qualified_indices.size == 0:
#             # 若无满足条件的，取最相似的top_n
#             top_indices = np.argsort(-similarities)[:self.top_n]
#         else:
#             # 否则按相似度排序取前top_n
#             sorted_indices = np.argsort(-similarities[qualified_indices])
#             top_indices = qualified_indices[sorted_indices][:self.top_n]
        
#         return self.df.iloc[top_indices]

#     def _create_rag_prompt(
#         self,
#         query: str,
#         similar_texts: pd.DataFrame,
#         max_context_length: int
#     ) -> str:
#         context_parts = []
#         current_length = 0
        
#         for text in similar_texts['text'].values:
#             text_segment = f"相关段落：{text}"
#             added_length = len(text_segment) + 2
            
#             if current_length + added_length > max_context_length:
#                 available_length = max_context_length - current_length - 20
#                 if available_length > 50:
#                     # 按句子截断保留更多语义
#                     truncated = text[:available_length].rsplit('.', 1)[0] + "..."
#                     context_parts.append(f"相关段落：{truncated}")
#                 break
                
#             context_parts.append(text_segment)
#             current_length += added_length
        
#         context = "\n\n".join(context_parts)
#         return f"用户查询：{query}\n\n以下是可供参考的资料，请尽可能从相关段落提取有效信息辅助详细回答：\n{context}"

# # 示例用法（如果需要直接运行测试）
# if __name__ == "__main__":
#     import sys
    
#     # 配置参数
#     EMBEDDINGS_FILE = "embeddings.parquet"
#     MODEL_NAME = './local_model'
    
#     try:
#         generator = RAGPromptGenerator(
#             embeddings_file=EMBEDDINGS_FILE,
#             model_name=MODEL_NAME
#         )
        
#         while True:
#             query = input("\n请输入查询内容（输入'exit'退出）：")
#             if query.lower() == 'exit':
#                 break
                
#             prompt = generator.generate_prompt(query)
#             print("\n增强后的Prompt：")
#             print("--------------------------------------------------")
#             print(prompt)
#             print("--------------------------------------------------")
            
#     except Exception as e:
#         print(f"发生错误：{str(e)}")
#         sys.exit(1)