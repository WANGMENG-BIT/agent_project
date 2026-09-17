"""Long-term user-memory tools and runtime context schema."""

from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool
from typing_extensions import TypedDict


@dataclass
class Context:
    """传递给 Agent ToolRuntime 的当前用户上下文。"""

    user_id: str


class UserInfo(TypedDict, total=False):
    """允许写入长期记忆的用户字段。"""

    name: str
    school: str
    job: str
    hobby: str
    learning: str


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """保存或更新当前用户的长期信息。当用户主动告诉你姓名、学校、职业、兴趣爱好或正在学习的方向等未来仍然有用的信息时使用。"""
    user_id = runtime.context.user_id
    assert runtime.store is not None

    old_item = runtime.store.get(("users",), user_id)
    merged_info = dict(old_item.value) if old_item is not None else {}
    merged_info.update(dict(user_info))

    runtime.store.put(("users",), user_id, merged_info)
    return f"用户 {user_id} 的长期信息已更新。"


@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """读取当前用户已经保存的长期信息。当用户询问“我是谁”“你还记得我吗”或当前任务需要使用以前保存的用户资料时使用。"""
    user_id = runtime.context.user_id
    assert runtime.store is not None

    item = runtime.store.get(("users",), user_id)
    if item is None:
        return "当前没有保存该用户的长期信息。"

    return str(item.value)
