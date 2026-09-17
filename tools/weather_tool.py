"""Weather query tool."""

import requests
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前实时天气。当用户询问某个城市当前的天气、温度等信息时使用。不适合查询未来多日天气预报。"""
    url = f"https://wttr.in/{city}?format=j1&lang=zh"

    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        temperature = current["temp_C"]
        description = current["weatherDesc"][0]["value"]

        return f"{city}当前天气：{description}，温度{temperature}℃"
    except Exception as exc:
        return f"天气查询失败：{str(exc)}"
