"""Internet search tool."""

from ddgs import DDGS
from langchain.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """联网搜索公开互联网信息。当用户询问最新信息、当前事件、不确定的外部事实或需要从互联网检索资料时使用。"""
    try:
        max_results = max(1, min(max_results, 8))
        results = list(DDGS(timeout=8).text(query, max_results=max_results) or [])

        if not results:
            return "没有搜索到相关结果。"

        output = []
        for index, item in enumerate(results, start=1):
            title = item.get("title", "无标题")
            body = item.get("body", "")
            url = item.get("href", "")
            output.append(
                f"结果 {index}\n"
                f"标题：{title}\n"
                f"摘要：{body}\n"
                f"链接：{url}".strip()
            )

        return "\n\n".join(output)
    except Exception as exc:
        return f"联网搜索失败：{str(exc)}"
