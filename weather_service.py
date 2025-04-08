# weather_service.py
import requests
import time
from datetime import datetime
from pprint import pformat

class WeatherService:
    def __init__(self, geonames_user, owm_api_key):
        self.GEONAMES_USER = geonames_user
        self.OWM_API_KEY = owm_api_key
        self.TIMEOUT = (10, 15)
        self.MAX_RETRIES = 3
        self.REQUEST_DELAY = 2

    def safe_api_call(self, url, params, service_name):
        """增强版安全API请求"""
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                print(f"\n🔧 [{service_name}] 请求尝试 {attempt+1}/{self.MAX_RETRIES+1}")
                response = requests.get(url, params=params, timeout=self.TIMEOUT)
                response.raise_for_status()
                print(f"✅ [{service_name}] 请求成功 (状态码: {response.status_code})")
                return response
            except requests.exceptions.Timeout as e:
                print(f"⌛ [{service_name}] 请求超时: {str(e)}")
                if attempt == self.MAX_RETRIES:
                    raise Exception(f"{service_name} 请求超过最大重试次数")
                time.sleep(2 ** (attempt + 1))
            except requests.exceptions.RequestException as e:
                print(f"⚠️ [{service_name}] 请求失败: {str(e)}")
                if attempt == self.MAX_RETRIES:
                    raise
                time.sleep(1)
        return None

    def get_geodata(self, place_name):
        """地理编码服务"""
        base_url = "http://api.geonames.org/searchJSON"
        params = {
            "q": place_name,
            "maxRows": 3,
            "username": self.GEONAMES_USER,
            "featureClass": "P",
            "orderby": "relevance",
            "isNameRequired": True,
            "style": "FULL"
        }

        try:
            response = self.safe_api_call(base_url, params, "GeoNames")
            if not response:
                return None

            data = response.json()
            
            # 结果筛选逻辑
            best_result = None
            for result in data.get('geonames', []):
                # 优先选择首都或人口密集区
                if result.get('fcode') == 'PPLC':  # 首都
                    best_result = result
                    break
                if not best_result and result.get('population', 0) > 1000000:
                    best_result = result
            
            if not best_result and data.get('geonames'):
                best_result = data['geonames'][0]

            if best_result:
                return {
                    "name": best_result.get('name'),
                    "lat": float(best_result['lat']),
                    "lon": float(best_result['lng']),
                    "country_code": best_result.get('countryCode'),
                    "region_code": best_result.get('adminCodes1', {}).get('ISO3166_2'),
                    "population": best_result.get('population', 0)
                }

            raise ValueError(f"未找到有效地理信息: {place_name}")

        except Exception as e:
            print(f"🗺️ 地理编码失败: {str(e)}")
            return None


    def get_weather(self, location=None, lat=None, lon=None):
        """增强版天气查询"""
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "appid": self.OWM_API_KEY,
            "units": "metric",
            "lang": "zh_cn"
        }

        # 构建查询参数
        if lat is not None and lon is not None:
            print(f"🌐 使用经纬度查询: {lat},{lon}")
            params.update({"lat": lat, "lon": lon})
        elif location:
            print(f"🌍 使用地名直接查询: {location}")
            params["q"] = location
        else:
            raise ValueError("必须提供位置参数")

        try:
            response = self.safe_api_call(base_url, params, "OpenWeatherMap")
            if not response:
                return None

            data = response.json()
            if data.get('cod') != 200:
                raise Exception(f"天气接口错误 {data.get('cod')}: {data.get('message')}")

            main = data.get('main', {})
            wind = data.get('wind', {})
            weather = data.get('weather', [{}])[0]
            sys = data.get('sys', {})

            return {
                "location_name": data.get('name', '未知'),
                "country_code": sys.get('country'),
                "temp": main.get('temp'),
                "feels_like": main.get('feels_like'),
                "humidity": main.get('humidity'),
                "pressure": main.get('pressure'),
                "weather_desc": weather.get('description'),
                "wind_speed": wind.get('speed'),
                "wind_deg": wind.get('deg'),
                "coord": data.get('coord', {}),
                "dt": datetime.fromtimestamp(data.get('dt', 0))
            }

        except Exception as e:
            print(f"天气查询失败: {str(e)}")
            return None