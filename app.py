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
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>航线优化平台</title>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f4f4f9;
        color: #333;
        margin: 0;
    }

    /* 示例库容器样式 */
    .gallery-container {
        width: 100%;
        position: relative;
        overflow: hidden;
    }

    .gallery {
        display: flex;
        width: 400%;
        animation: slide 150s infinite linear;
    }

    .gallery-item {
        width: 25%;
        padding: 10px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        transition: transform 0.3s ease-in-out;
    }

    .gallery-item:hover {
        transform: scale(1.05);
    }

    .gallery-item img {
        width: 100%;
        height: 200px;
        object-fit: cover;
    }

    /* 定义动画 */
    @keyframes slide {
        0% {
            transform: translateX(0);
        }
        25% {
            transform: translateX(-100%);
        }
        50% {
            transform: translateX(-200%);
        }
        75% {
            transform: translateX(-300%);
        }
        100% {
            transform: translateX(0);
        }
    }
  </style>
</head>
<body class="bg-gray-100">

  <!-- 导航栏 -->
  <nav class="bg-blue-600 text-white p-4">
    <div class="container mx-auto flex justify-between items-center">
      <a href="#" class="text-2xl font-bold flex items-center">
        <img src="https://via.placeholder.com/40" alt="Logo" class="mr-2">航线优化平台
      </a>
      <ul class="flex space-x-4">
        <li><a href="#home" class="hover:underline">主页</a></li>
        <li><a href="#optimize" class="hover:underline">航线优化</a></li>
        <li><a href="#gallery" class="hover:underline">示例库</a></li>
        <li><a href="#about" class="hover:underline">关于我们</a></li>
      </ul>
    </div>
  </nav>

  <!-- 主页 -->
  <section id="home" class="container mx-auto py-16 px-4 text-center">
    <h1 class="text-4xl font-bold text-blue-600 mb-6">欢迎来到航线优化平台</h1>
    <p class="text-xl mb-6">通过先进的AI技术，帮助您优化海上航线，节省时间和成本。</p>
    <a href="#optimize" class="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700">立即开始优化</a>
  </section>

  <!-- 对话模式 -->
  <div class="container mx-auto py-16 px-4">
    <h2 class="text-3xl font-semibold text-center text-blue-600 mb-8">对话模式</h2>
    <form method="post" class="max-w-lg mx-auto bg-white p-8 rounded-lg shadow-lg">
      <input type="hidden" name="action" value="model4.7">
      <label>请输入文本内容：</label>
      <input type="text" name="user_input" required class="w-full p-3 border border-gray-300 rounded-md">
      <input type="submit" value="提交对话" class="mt-4 px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700">
    </form>
  </div>

  <!-- 航线优化表单 -->
  <section id="optimize" class="container mx-auto py-16 px-4">
    <h2 class="text-3xl font-semibold text-center text-blue-600 mb-8">航线优化</h2>
    <form id="optimize-form" action="#" method="post" class="max-w-lg mx-auto bg-white p-8 rounded-lg shadow-lg">
      <div class="mb-4">
        <label for="start" class="block text-lg font-medium">起始港口</label>
        <input type="text" name="start" class="w-full p-3 border border-gray-300 rounded-md" required>
      </div>
      <div class="mb-4">
        <label for="end" class="block text-lg font-medium">目的港口</label>
        <input type="text" name="end" class="w-full p-3 border border-gray-300 rounded-md" required>
      </div>
      <div class="mb-4">
        <label for="middle" class="block text-lg font-medium">中间港口（最多两个）</label>
        <input type="text" name="middle" class="w-full p-3 border border-gray-300 rounded-md" placeholder="例如：新加坡，洛杉矶">
      </div>
      <div class="flex justify-center">
        <button type="submit" class="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700">生成建议</button>
      </div>
    </form>

    <!-- 优化结果展示 -->
    <div id="result" class="mt-8 hidden">
      <div class="bg-white p-6 rounded-lg shadow-lg">
        <h3 class="text-xl font-semibold text-blue-600">中文建议：</h3>
        <p>根据您提供的起始港口和目的港口，AI将为您生成最优化的航线推荐...</p>
        <h3 class="text-xl font-semibold text-blue-600 mt-4">English Suggestion:</h3>
        <p>Based on the ports you selected, AI will provide the most optimized route...</p>
      </div>
    </div>
  </section>

  <!-- 示例库 -->
  <section id="gallery" class="bg-gray-50 py-16 px-4">
    <h2 class="text-3xl font-semibold text-center text-blue-600 mb-8">航线优化示例库</h2>
    <div class="gallery-container">
      <div class="gallery">
        <div class="gallery-item">
          <img src="https://via.placeholder.com/400x250" alt="航线示例">
          <h3 class="text-lg font-semibold text-blue-600 mt-4">示例1：上海到洛杉矶</h3>
        </div>
        <div class="gallery-item">
          <img src="https://via.placeholder.com/400x250" alt="航线示例">
          <h3 class="text-lg font-semibold text-blue-600 mt-4">示例2：巴拿马到鹿特丹</h3>
        </div>
        <div class="gallery-item">
          <img src="https://via.placeholder.com/400x250" alt="航线示例">
          <h3 class="text-lg font-semibold text-blue-600 mt-4">示例3：香港到纽约</h3>
        </div>
        <div class="gallery-item">
          <img src="https://via.placeholder.com/400x250" alt="航线示例">
          <h3 class="text-lg font-semibold text-blue-600 mt-4">示例4：广州到东京</h3>
        </div>
      </div>
    </div>
  </section>

  <!-- 关于我们 -->
  <section id="about" class="bg-gray-800 text-white py-16 px-4">
    <div class="container mx-auto text-center">
      <h2 class="text-3xl font-semibold mb-6">关于我们</h2>
      <p class="text-xl mb-6">航线优化平台致力于通过AI技术为用户提供最佳航线推荐，帮助节省运输时间和成本。</p>
      <p class="text-lg">我们结合天气、港口拥堵和费用等因素，为商家和个人用户提供个性化的航线优化建议。</p>
    </div>
  </section>

  <!-- 页脚 -->
  <footer class="bg-gray-900 text-white text-center py-4">
    <p>&copy; 2025 航线优化平台 | 保留所有权利</p>
  </footer>

  <script>
    // 处理表单提交并显示优化结果
    document.getElementById('optimize-form').addEventListener('submit', function(event) {
      event.preventDefault();
      document.getElementById('result').classList.remove('hidden');
    });
  </script>

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

    
