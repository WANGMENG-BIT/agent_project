"""Agent construction and structured response schema."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

from agent.prompts import SYSTEM_PROMPT
from tools.memory_tool import Context
from tools.tool_registry import get_tools


class AgentResponse(BaseModel):
    """Agent 最终结构化输出。"""

    answer: str = Field(description="最终给用户的自然语言回答")
    intent: Literal[
        "chat",
        "weather",
        "calculation",
        "search",
        "database",
        "memory",
        "mixed",
    ] = Field(
        description=(
            "用户本次请求的主要意图。如果同时涉及多个不同任务，"
            "例如天气+计算，使用 mixed。"
        )
    )
    success: bool = Field(
        description=(
            "本次用户任务是否真正完成。如果工具超时、查询失败、"
            "没有获取到必要信息，则应为 False。"
        )
    )


def create_chat_agent(memory, store):
    """根据传入的 checkpointer 与 store 创建 Agent。"""
    load_dotenv(override=True)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    llm = ChatDeepSeek(
        api_key=api_key,
        api_base=base_url,
        model="deepseek-v4-flash",
        extra_body={"thinking": {"type": "disabled"}},
    )

    return create_agent(
        model=llm,
        tools=get_tools(),
        system_prompt=SYSTEM_PROMPT,
        response_format=AgentResponse,
        checkpointer=memory,
        store=store,
        context_schema=Context,
    )
