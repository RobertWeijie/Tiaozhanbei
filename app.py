#!/usr/bin/env python3

import os
from flask import Flask, render_template_string, request
import google.generativeai as genai
import requests
from main_logic import run_4_7_logic  # 引入4.7分析逻辑

# 设置 Google Gemini API Key（推荐从环境变量读取，更安全）
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "00fe8681e06234c50dae98fafeef312e")

app = Flask(__name__)


# 获取实时天气
def get_real_time_weather(port):
		try:
				url = f"https://api.openweathermap.org/data/2.5/weather?q={port}&appid={WEATHER_API_KEY}&units=metric"
				r = requests.get(url)
				return r.json()['weather'][0]['description']
		except Exception as e:
				return f"天气获取失败：{e}"
	
	
# 航线优化逻辑
def generate_analysis(start, end, middle_ports):
		start_weather = get_real_time_weather(start)
		end_weather = get_real_time_weather(end)
		middle_weather = [get_real_time_weather(p) for p in middle_ports]
	
		# 构建模型的输入内容（prompt）
		prompt = (
				f"你是专业海运航线规划师。用户需求：从 {start} 到 {end}。"
				f"{start} 当前天气 {start_weather}，{end} 当前天气 {end_weather}。"
		)
		if middle_ports:
				prompt += f"途径港口：{', '.join(middle_ports)}，天气分别为 {', '.join(middle_weather)}。"
			
		prompt += (
				"请考虑以下因素，提供优化航线建议。\n\n"
				"1. **天气条件**：如果天气良好，选择最短且经济的航线；"
				"如果天气恶劣，如大风、大雾或其他影响航行的因素，建议选择避开影响区域的替代航线。\n\n"
			
				"2. **港口情况**：评估各港口的繁忙程度，避免在高峰期到达。"
				"如果中途港口拥堵或费用较高，建议选择其他港口或绕道，避免不必要的延误和额外费用。\n\n"
			
				"3. **航程时间**：计算预计的航程时间，避免不必要的绕行或时间过长的航线。"
				"如果某些航段时间过长，请提供节省时间的方案。\n\n"
		)
	
		if middle_ports:
				prompt += (
						"4. **港口收费**：如果中途港口有额外收费，请提供绕道港口和新路线的建议，"
						"并给出预计费用变化。\n\n"
				)
			
		prompt += (
				"5. **航行成本**：估算运输成本，考虑油耗、港口费、船只停靠费等。"
				"若有替代路线，计算费用差异，推荐最具成本效益的航线。\n\n"
			
				"6. **船只适配**：请确保选择的航线适合当前船只的规格和载重能力。"
				"如果船只无法通过某些港口，建议调整路线。\n\n"
			
				"请提供中文和英文的优化建议，中文在前，英文用 'English:' 分隔。"
				"特殊情况下，如避灾或绕道，需详细解释替代路径的原因。"
		)
	
	
		model = genai.GenerativeModel("gemini-2.0-flash")
		response = model.generate_content(prompt)
		text = response.text
	
		return {
				"中文": text.split("English:")[0].strip() if "English:" in text else text,
				"English": text.split("English:")[1].strip() if "English:" in text else "N/A"
		}
		

# HTML 模板
html_template = """
<!doctype html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>航线与文本智能平台</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"> <!-- 引入图标库 -->
    <style>
        /* 全局样式 */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            scroll-behavior: smooth;
        }

        body {
            background: #f0f5ff; /* 更柔和的背景色 */
            color: #333;
            line-height: 1.6;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            background: white;
            border-radius: 16px; /* 更大的圆角 */
            box-shadow: 0 4px 12px rgba(26, 115, 232, 0.1); /* 更细腻的阴影 */
            padding: 32px;
            width: 100%;
            max-width: 960px;
            margin: 32px 0;
            transition: transform 0.3s ease; /* 容器悬停动画 */
        }

        .container:hover {
            transform: scale(1.01);
        }

        h2 {
            color: #1a73e8;
            font-size: 2.25rem;
            text-align: center;
            margin-bottom: 24px;
            position: relative;
            display: inline-block;
        }

        h2::after {
            content: "";
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 3px;
            background: #1a73e8;
            border-radius: 2px;
        }

        /* 导航栏升级 */
        nav {
            background: linear-gradient(135deg, #1a73e8, #1558b6); /* 渐变背景 */
            padding: 20px 40px;
            width: 100%;
            box-shadow: 0 8px 24px rgba(26, 115, 232, 0.15);
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 960px;
            margin: 0 auto;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo img {
            width: 40px; /* 更大的Logo */
            height: 40px;
            border-radius: 50%; /* Logo圆形 */
            object-fit: cover;
        }

        .logo span {
            font-size: 1.75rem;
            font-weight: 700;
            color: white;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .nav-links {
            display: flex;
            gap: 32px;
        }

        .nav-links a {
            color: white;
            text-decoration: none;
            font-size: 1.125rem;
            font-weight: 500;
            position: relative;
            transition: color 0.3s ease;
        }

        .nav-links a:hover {
            color: #e0f0ff;
        }

        .nav-links a::before {
            content: "";
            position: absolute;
            bottom: -6px;
            left: 0;
            width: 100%;
            height: 2px;
            background: white;
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }

        .nav-links a:hover::before {
            transform: scaleX(1);
        }

        /* 表单样式优化 */
        form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        label {
            font-size: 1.125rem;
            color: #666;
        }

        input {
            padding: 14px 20px;
            border: 2px solid #e0e8f9; /* 更明显的边框 */
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        input:focus {
            outline: none;
            border-color: #1a73e8;
            box-shadow: 0 0 0 4px rgba(26, 115, 232, 0.2);
        }

        input[type="submit"] {
            background: #1a73e8;
            color: white;
            font-size: 1.125rem;
            font-weight: 600;
            padding: 16px;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        input[type="submit"]:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 18px rgba(26, 115, 232, 0.2);
        }

        /* 结果展示区 */
        .result h3 {
            color: #1a73e8;
            font-size: 1.5rem;
            margin: 32px 0 16px;
        }

        .result p {
            font-size: 1.125rem;
            color: #444;
            line-height: 1.8;
        }

        /* 关于我们与页脚 */
        #about {
            text-align: center;
            padding: 64px 32px;
            background: ;
            color: black;
            margin-top: 64px;
        }

        footer {
            background: #1558b6;
            color: white;
            text-align: center;
            padding: 32px;
            width: 100%;
            margin-top: 64px;
            font-size: 1rem;
            box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.1);
        }

        /* 响应式设计 */
        @media (max-width: 768px) {
            nav {
                padding: 20px;
            }

            .nav-links {
                gap: 24px;
            }

            .container {
                padding: 24px;
            }

            h2 {
                font-size: 1.75rem;
            }
        }
    </style>
</head>
<body>
    <!-- 导航栏 -->
    <nav>
        <div class="nav-container">
            <div class="logo">
                <img src="https://github.com/RobertWeijie/Tiaozhanbei/demo1.png" alt="Logo"> <!-- 建议替换为实际Logo -->
                <span>航线优化平台</span>
            </div>
            <ul class="nav-links">
                <li><a href="#home">主页</a></li>
                <li><a href="#optimize">航线优化</a></li>
                <li><a href="#gallery">对话模式</a></li>
                <li><a href="#about">关于我们</a></li>
            </ul>
        </div>
    </nav>

    <section id="optimize">
    <div class="container">
        <h2>航线优化</h2>
        <form method="post">
            <label>起始港口:</label>
            <input type="text" name="start" required>
            <label>目的港口:</label>
            <input type="text" name="end" required>
            <label>中间港口（最多两个，用逗号分隔）:</label>
            <input type="text" name="middle">
            <input type="submit" value="生成建议">
        </form>
    </div>

    <div class="container">
        <h3>平台介绍</h3>
        <p>我们是一个专业的航运航线优化平台，利用先进的人工智能技术，结合实时天气数据、路径风险和费用因素，为用户提供最优的航线规划方案和对话模式。</p>
        <p>普通用户：只需选择起始港口和目的地港口，我们将为您生成最快且经济的推荐路线。</p>
        <p>商家用户：除了起始和目的地港口，您还可以填写最多两个中间港口，我们会根据您的需求提供不同的航线方案，同时会考虑天气变化、中途港口收费等特殊情况，为您提供灵活的绕道选择。</p>
    </div>

      <section id="gallery">
    <div class="container">
        <h2>对话模式</h2>
        <form method="post">
            <input type="hidden" name="action" value="model4.7">
            <label>请输入文本内容：</label>
            <input type="text" name="user_input" required>
            <input type="submit" value="提交对话">
        </form>
    </div>

    {% if result %}
    <div class="container result">
        <h3>📌 中文建议：</h3>
        <p>{{ result.中文|replace('**', '')|replace('*', '') }}</p>
        <h3>🌐 English Suggestion:</h3>
        <p>{{ result.English|replace('**', '')|replace('*', '') }}</p>
    </div>
    {% endif %}

    {% if result_47 %}
    <div class="container result">
        <h3>📘 模型分析结果：</h3>
        <p>{{ result_47 }}</p>
    </div>
    {% endif %}

    <!-- 关于我们 -->
    <section id="about">
        <h2>关于我们</h2>
        <p class="text-xl mb-6">航线优化平台致力于通过AI技术为用户提供最佳航线推荐，帮助节省运输时间和成本。</p>
        <p class="text-lg">我们结合天气、港口拥堵和费用等因素，为商家和个人用户提供个性化的航线优化建议。</p>
    </section>

    <!-- 页脚 -->
    <footer>
        <p>&copy; 2025 BUAA-挑战杯航线优化平台 | 保留所有权利</p>
    </footer>
</body>
</html>
"""


# Flask 路由
@app.route("/", methods=["GET", "POST"])
def home():
		if request.method == "POST":
				if request.form.get("action") == "model4.7":
						user_input = request.form["user_input"]
						result_47 = run_4_7_logic(user_input)
						return render_template_string(html_template, result_47=result_47)
			
				start = request.form["start"]
				end = request.form["end"]
				middle_ports = [p.strip() for p in request.form["middle"].split(",") if p.strip()]
				if len(middle_ports) > 2:
						return render_template_string(html_template,
																					result={"中文": "最多两个中间港口", "English": "Up to 2 middle ports only"})
				result = generate_analysis(start, end, middle_ports)
				return render_template_string(html_template, result=result)
		return render_template_string(html_template)


# 启动服务，适配云服务器监听
if __name__ == "__main__":
		app.run(host="0.0.0.0", port=5000)
	
