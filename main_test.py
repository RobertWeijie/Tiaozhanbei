from weather_service import WeatherService
from openai import OpenAI
from datetime import datetime
import json
import os
from rag_prompt_generator import RAGPromptGenerator  # 新增导入

client = OpenAI(
    api_key="DASHSCOPE_API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 初始化服务
weather_service = WeatherService(
    geonames_user="shouxc",
    owm_api_key="4aa6b87fb08c3542988cc4bdf67da3af"
)

# 初始化RAG组件（新增部分）
rag_generator = RAGPromptGenerator(
    embeddings_file="embeddings.parquet",
    model_path="./local_model",
    top_n=5,               # 控制返回段落数
    similarity_threshold=0.4,  # 相似度阈值
    max_context_length=1500  # 上下文长度
)

# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            # 因为获取当前时间无需输入参数，因此parameters为空字典
            "parameters": {}
        }
    },  
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {  
                "type": "object",
                "properties": {
                    # 查询天气时需要提供位置，因此参数设置为location
                    "location": {
                        "type": "string",
                        "description": "城市或县区或经纬度位置。"
                    }
                }
            },
            "required": [
                "location"
            ]
        }
    }
]

# 模拟天气查询工具。返回结果示例：“北京今天是雨天。”
# 修改原有的get_current_weather函数
def get_current_weather(arguments):
    """新版天气查询函数"""
    try:
        location = arguments.get("location")
        if not location:
            return "需要提供有效位置信息"
        
        # 地理编码获取坐标
        geo_data = weather_service.get_geodata(location)
        if not geo_data:
            return "无法获取该位置的地理信息"
        
        # 获取天气数据
        weather_data = weather_service.get_weather(
            lat=geo_data["lat"], 
            lon=geo_data["lon"]
        )
        if not weather_data:
            return "天气查询服务暂时不可用"
        
        # 格式化输出
        return (
            f"{weather_data['location_name']}当前天气：\n"
            f"- 天气状况：{weather_data['weather_desc']}\n"
            f"- 温度：{weather_data['temp']}°C\n"
            f"- 体感温度：{weather_data['feels_like']}°C\n"
            f"- 湿度：{weather_data['humidity']}%\n"
            f"- 风速：{weather_data['wind_speed']}米/秒"
        )
        
    except Exception as e:
        print(f"天气查询异常：{str(e)}")
        return "天气信息获取失败，请稍后重试"

# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
def get_current_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"


def get_response_stream(messages):
    return client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
        parallel_tool_calls=True,
        # tool_choice={"type": "function", "function": {"name": "get_current_time"}},#请不要去掉以备不时之需
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        stream_options={
            "include_usage": True
        }
    )
        
        

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
                content = "未知工具调用"
                
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


def call_with_messages_stream():
    global messages
    conversation_idx = 1
    
    while True:
        print('\n' + "="*20 + f"第{conversation_idx}轮对话" + "="*20)
        conversation_idx += 1
    
        user_input = input("请输入（输入q退出）: ").strip()
        if user_input.lower() == 'q':
            break
            
        # ========== 新增RAG处理部分 ==========
        try:
            # 生成增强后的prompt
            enhanced_prompt = rag_generator.generate_prompt(user_input)
            print("\n[系统提示] 已注入{}个相关段落".format(len(enhanced_prompt.split("相关段落："))-1))
        except Exception as e:
            print(f"\n[系统警告] RAG检索失败：{str(e)}")
            enhanced_prompt = user_input  # 降级处理
            
        # 添加增强后的用户消息
        messages.append({
            "role": "user",
            "content": enhanced_prompt  # 使用增强后的prompt
        })
        # ========== RAG处理结束 ==========
    
        full_content = ""
        tool_calls_accumulator = {}
    
        while True:
            try:
                completion = get_response_stream(messages)
                assistant_message = {"role": "assistant"}
                
                # 流式处理
                for chunk in completion:
                    if not chunk.choices or not chunk.choices[0].delta:
                        continue
                        
                    delta = chunk.choices[0].delta
                    
                    # 处理文本内容
                    if delta.content:
                        full_content += delta.content
                        print(delta.content, end="", flush=True)
                    
                    # 处理工具调用
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            index = tool_call.index
                            if index not in tool_calls_accumulator:
                                tool_calls_accumulator[index] = {
                                    "id": "",
                                    "function": {"name": "", "arguments": ""},
                                    "index": index
                                }
                            
                            if tool_call.id:
                                tool_calls_accumulator[index]["id"] = tool_call.id
                            if tool_call.function.name:
                                tool_calls_accumulator[index]["function"]["name"] = tool_call.function.name
                            if tool_call.function.arguments:
                                tool_calls_accumulator[index]["function"]["arguments"] += tool_call.function.arguments

                # 构建符合规范的Assistant Message
                assistant_message["content"] = full_content.strip()
                if tool_calls_accumulator:
                    assistant_message["tool_calls"] = [
                        {
                            "id": data["id"],
                            "type": "function",
                            "function": {
                                "name": data["function"]["name"],
                                "arguments": data["function"]["arguments"]
                            },
                            "index": data["index"]
                        }
                        for data in tool_calls_accumulator.values()
                    ]
                    # 当存在工具调用时content可为空
                    if not assistant_message["content"]:
                        del assistant_message["content"]

                messages.append(assistant_message)
                
                # 处理工具调用（生成Tool Message）
                if "tool_calls" in assistant_message:
                    tool_responses = process_tool_calls(assistant_message)
                    messages.extend(tool_responses)
                    
                    # 自动获取最终回复
                    final_response = ""
                    completion = get_response_stream(messages)
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content:
                            final_response += chunk.choices[0].delta.content
                            print(chunk.choices[0].delta.content, end="", flush=True)
                    
                    # 添加最终Assistant Message
                    if final_response:
                        messages.append({
                            "role": "assistant",
                            "content": final_response
                        })
                    break
                else:
                    break

            except Exception as e:
                print(f"\n发生错误：{str(e)}")
                # 回滚无效消息
                while len(messages) > 0 and messages[-1]["role"] in ["assistant", "tool"]:
                    messages.pop()
                break

system_prompt = """作为海运智能决策系统，请按以下结构输出分析报告：

【航线推荐】
- 主推路线：路线名称（基于XXX因素）
- 替代方案：方案名称（简要优势）
- 关键参数：航程（XX海里）、预估耗时（XX天）

【风险评估】
1. 气象风险：当前主要风险点（台风/季风等）
2. 地缘风险：关键区域状态（如苏伊士运河通行情况）
3. 经济风险：燃油成本波动范围（±X%）

【决策建议】
- 最优方案：XXX
- 备选策略：XXX
- 时间窗口：建议XX日前启航

输出要求：
1. 使用清晰的小标题分层
2. 关键数值用**加粗**标注
3. 如数据不足请明确标注"需补充XX信息"
4. 在保证合理的强狂下尽可能内容充分翔实，尽可能全面
5. 实时数据（如当前时间、天气）必须通过工具获取
6. 参考资料来自历史数据库，实时决策请以工具获取数据为准
7. 综合实时数据和资料库判断生成航行建议
"""

if __name__ == '__main__':
    # 初始化一个 messages 数组
    messages = [
        {
            "role": "system",
            "content": """你是一名气象与海运航线智能助手，能根据用户选择的地点及要求，自主通过工具获取气象信息判断是否适合出航，并进行航线规划。
            如果用户提供的时间信息不明，则用工具获取当前时间，如果地点信息或要求不明确，你可以继续追问他，让他提供更丰富的信息便于判断。
            你的回答要充实且具有参考价值，体现专业性。"""+system_prompt,
        }
    ]
    print("欢迎光临气象与海运航线智能助手，有什么可以帮助您的？")
    call_with_messages_stream()