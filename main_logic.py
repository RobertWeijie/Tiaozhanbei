# main_logic.py

from openai import OpenAI
from datetime import datetime
import json
from rag_prompt_generator import RAGPromptGenerator
from weather_service import WeatherService
import google.generativeai as genai

# 配置 Google API 密钥（如果有使用）
genai.configure(api_key="your_google_api_key")  # 替换为实际的API Key

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-0f50b7ba05fa41bba388684a8ca669fc",  # 替换为你的通义千问 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 使用通义千问的兼容模式
)

# 初始化天气服务
weather_service = WeatherService(
    geonames_user="shouxc",  # 替换为你的用户名
    owm_api_key="4aa6b87fb08c3542988cc4bdf67da3af"  # 替换为你的 OpenWeather API Key
)

# 初始化RAG组件（增强型分析模型）
rag_generator = RAGPromptGenerator(
    embeddings_file="embeddings.parquet",  # 用到的嵌入文件路径
    model_path="./local_model",  # 本地模型路径
    top_n=5,  # 控制返回段落数
    similarity_threshold=0.4,  # 相似度阈值
    max_context_length=1500  # 上下文长度
)


# 提取和处理工具调用（天气、时间等）
def get_current_weather(arguments):
    try:
        location = arguments.get("location")
        geo_data = weather_service.get_geodata(location)
        weather_data = weather_service.get_weather(lat=geo_data["lat"], lon=geo_data["lon"])
        return f"{weather_data['location_name']} 当前天气：{weather_data['weather_desc']}, 温度：{weather_data['temp']}°C"
    except Exception as e:
        return f"天气获取失败：{str(e)}"


def get_current_time():
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def process_tool_calls(assistant_message):
    tool_responses = []
    for tool_call in assistant_message.get("tool_calls", []):
        try:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])

            if func_name == "get_current_weather":
                content = get_current_weather(arguments)
            elif func_name == "get_current_time":
                content = get_current_time()
            else:
                content = "未知工具"

            tool_responses.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call["id"]
            })
        except Exception as e:
            tool_responses.append({
                "role": "tool",
                "content": f"工具调用失败：{str(e)}",
                "tool_call_id": tool_call.get("id", "")
            })
    return tool_responses


# 4.7模型分析逻辑
def run_4_7_logic(user_input: str) -> str:
    system_prompt = """作为海运智能决策系统，请按以下结构输出分析报告：

    【航线推荐】
    - 主推路线：路线名称（基于XXX因素）
    - 替代方案：方案名称（简要优势）
    - 航程（XX海里）、预估耗时（XX天）

    【风险评估】
    1. 气象风险（台风/季风等）
    2. 地缘风险（如苏伊士通行）
    3. 成本波动（±X%）

    【决策建议】
    - 最优方案 / 备选策略 / 启航时间建议

    输出要求：
    - 清晰小标题 + **加粗数值**
    - 实时数据用工具获取
    """

    # 消息列表，用于传递给模型
    messages = [{"role": "system", "content": system_prompt}]

    try:
        # 通过 RAG 生成增强的提示词
        enhanced_prompt = rag_generator.generate_prompt(user_input)
    except Exception as e:
        print(f"RAG处理失败：{str(e)}")
        enhanced_prompt = user_input  # 如果 RAG 失败则降级使用原始输入

    messages.append({"role": "user", "content": enhanced_prompt})

    # 向模型发送请求
    completion = client.chat.completions.create(
        model="qwen-plus",  # 使用通义千问模型（或你需要的其他模型）
        messages=messages,
        tools=[{
            "type": "function",
            "function": {"name": "get_current_weather", "arguments": {"location": "上海"}}
        }],
        parallel_tool_calls=True
    )

    assistant_message = completion.choices[0].message
    messages.append(assistant_message)

    # 处理工具调用响应
    if "tool_calls" in assistant_message:
        tool_responses = process_tool_calls(assistant_message)
        messages.extend(tool_responses)

        final_response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )
        return final_response.choices[0].message.content.strip()

    return assistant_message.content.strip()
