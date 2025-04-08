# main_logic.py

import os
import json
from datetime import datetime
from openai import OpenAI
from rag_prompt_generator import RAGPromptGenerator
from weather_service import WeatherService
import google.generativeai as genai

# 配置 Google Gemini（可选，未使用可忽略）
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "dummy"))

# ✅ 读取通义千问 API Key（强烈推荐使用环境变量）
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("❌ 未检测到 DASHSCOPE_API_KEY 环境变量，请在 Render 设置正确的值")

# 初始化 OpenAI 兼容客户端（通义千问）
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 初始化天气服务
weather_service = WeatherService(
    geonames_user="shouxc",  # 可改为你自己的账号
    owm_api_key=os.getenv("OWM_API_KEY", "dummy")  # 推荐也通过环境变量注入
)

# 初始化 RAG 组件（增强型分析）
rag_generator = RAGPromptGenerator(
    embeddings_file="embeddings.parquet",
    model_path="./local_model",
    top_n=5,
    similarity_threshold=0.4,
    max_context_length=1500
)

# 工具调用：天气
def get_current_weather(arguments):
    try:
        location = arguments.get("location")
        geo_data = weather_service.get_geodata(location)
        weather_data = weather_service.get_weather(lat=geo_data["lat"], lon=geo_data["lon"])
        return f"{weather_data['location_name']} 当前天气：{weather_data['weather_desc']}, 温度：{weather_data['temp']}°C"
    except Exception as e:
        return f"天气获取失败：{str(e)}"

# 工具调用：时间
def get_current_time():
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# 工具调用调度
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

# 主逻辑：4.7 航线分析逻辑
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

    messages = [{"role": "system", "content": system_prompt}]

    try:
        enhanced_prompt = rag_generator.generate_prompt(user_input)
        print("✅ RAG 提示词生成成功")
    except Exception as e:
        print(f"❌ RAG 处理失败：{str(e)}")
        enhanced_prompt = user_input

    messages.append({"role": "user", "content": enhanced_prompt})

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "获取城市天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }]
            # ❌ 注意：不加 parallel_tool_calls，通义不支持
        )
    except Exception as e:
        print(f"❌ 第一次模型调用失败：{str(e)}")
        return "模型调用失败，请检查 API Key 或服务状态"

    assistant_message = completion.choices[0].message
    messages.append(assistant_message)

    if "tool_calls" in assistant_message:
        tool_responses = process_tool_calls(assistant_message)
        messages.extend(tool_responses)

        try:
            final_response = client.chat.completions.create(
                model="qwen-plus",
                messages=messages
            )
            return final_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ 生成最终回复失败：{str(e)}")
            return "工具调用成功，但生成最终分析报告失败。"

    return assistant_message.content.strip()
