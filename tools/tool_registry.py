"""Central registry for all Agent tools."""

from tools.calculator_tool import calculator
from tools.database_tool import query_product
from tools.memory_tool import get_user_info, save_user_info
from tools.weather_tool import get_weather
from tools.web_search_tool import web_search


ALL_TOOLS = [
    get_weather,
    calculator,
    web_search,
    save_user_info,
    get_user_info,
    query_product,
]

REAL_TOOL_NAMES = {tool.name for tool in ALL_TOOLS}


def get_tools():
    """返回一份 Tool 列表副本，供 Agent 创建时使用。"""
    return list(ALL_TOOLS)
