"""Command-line entry point for the Agent application."""

import time

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from agent.agent import create_chat_agent
from tools.database_tool import init_demo_database
from tools.memory_tool import Context
from tools.tool_registry import REAL_TOOL_NAMES


def full_thread_id(user_id: str, short_id: str) -> str:
    """将 user_id 和短 thread_id 组合成全局唯一 thread_id。"""
    return f"{user_id}::{short_id}"


def show_history(agent, user_id: str, short_id: str) -> None:
    """显示指定用户当前会话的聊天历史。"""
    full_id = full_thread_id(user_id, short_id)
    config = {"configurable": {"thread_id": full_id}}

    state = agent.get_state(config)
    messages = state.values.get("messages", [])

    print(f"\n===== 会话 {short_id} 聊天记录 =====")

    if not messages:
        print("\n当前会话暂无聊天记录。")

    for message in messages:
        if message.type == "human":
            print(f"\n你：{message.content}")
        elif message.type == "ai":
            if message.content:
                print(f"\nAgent：{message.content}")
        elif message.type == "tool":
            tool_name = getattr(message, "name", None)
            if tool_name in REAL_TOOL_NAMES:
                print(f"\n[工具 {tool_name} 返回]：\n{message.content}")

    print("\n============================")


def list_threads(memory, user_id: str) -> None:
    """列出当前用户已有的短期会话。"""
    prefix = f"{user_id}::"
    thread_ids = set()

    for checkpoint in memory.list(None):
        full_id = checkpoint.config["configurable"]["thread_id"]
        if full_id.startswith(prefix):
            short_id = full_id[len(prefix):]
            thread_ids.add(short_id)

    print("\n===== 你的所有会话 =====")

    if not thread_ids:
        print("暂无会话记录。")

    for short_id in sorted(thread_ids):
        print(short_id)

    print("======================")


def show_tool_trace(messages):
    """打印本轮真正发生的业务 Tool 调用。"""
    used_tools = []

    print("\n========== Tool 调用过程 ==========")

    for message in messages:
        if message.type == "ai":
            tool_calls = getattr(message, "tool_calls", []) or []

            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})

                if tool_name in REAL_TOOL_NAMES:
                    used_tools.append(tool_name)
                    print(
                        f"\nAgent 调用工具：{tool_name}\n"
                        f"参数：\n{tool_args}"
                    )

        elif message.type == "tool":
            tool_name = getattr(message, "name", None)
            if tool_name in REAL_TOOL_NAMES:
                print(
                    f"\n工具 {tool_name} 返回：\n"
                    f"{message.content}"
                )

    if not used_tools:
        print("\n本轮没有调用任何业务 Tool。")

    print("\n===================================")
    return used_tools


def print_commands() -> None:
    """打印 CLI 支持的命令。"""
    print(
        "\n可用命令："
        "\n/users              查看已有长期记忆的用户"
        "\n/user <用户ID>      切换用户"
        "\n/threads            查看当前用户所有会话"
        "\n/switch <会话ID>    切换或创建会话"
        "\n/history            查看当前聊天记录"
        "\n/delete <会话ID>    删除指定会话"
        "\nexit / quit         退出程序"
    )


def run_cli(agent, memory, store) -> None:
    """运行命令行聊天循环。"""
    print("\n===== 用户登录 =====")

    default_user = "user"
    user_input_id = input(
        f"请输入你的用户ID（直接回车使用默认 '{default_user}'）："
    ).strip()

    current_user = user_input_id if user_input_id else default_user
    current_short_thread = "chat_001"

    print(f"当前用户：{current_user}")
    print(f"当前会话：{current_short_thread}")
    print_commands()

    while True:
        user_input = input(
            f"\n[{current_user}][{current_short_thread}] 你想说啥就说啥："
        ).strip()

        if not user_input:
            continue

        if user_input == "/users":
            items = store.search(("users",), limit=100)
            users = [item.key for item in items]

            if users:
                print("已保存长期信息的用户：", ", ".join(users))
            else:
                print("暂无用户长期信息。")
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Agent：下班！")
            break

        if user_input == "/threads":
            list_threads(memory, current_user)
            continue

        if user_input == "/history":
            show_history(agent, current_user, current_short_thread)
            continue

        if user_input.startswith("/switch "):
            new_short = user_input.split(" ", 1)[1].strip()
            if not new_short:
                print("请输入会话ID。")
                continue

            current_short_thread = new_short
            print(f"已切换到会话：{current_short_thread}")
            continue

        if user_input.startswith("/delete "):
            short_to_delete = user_input.split(" ", 1)[1].strip()
            if not short_to_delete:
                print("请输入需要删除的会话ID。")
                continue

            full_to_delete = full_thread_id(current_user, short_to_delete)
            memory.delete_thread(full_to_delete)
            print(f"已删除会话：{short_to_delete}")

            if short_to_delete == current_short_thread:
                current_short_thread = f"chat_{int(time.time())}"
                print(
                    "当前会话已删除，"
                    f"已自动切换到新会话：{current_short_thread}"
                )
            continue

        if user_input.startswith("/user "):
            new_user = user_input.split(" ", 1)[1].strip()
            if not new_user:
                print("请输入用户ID。")
                continue

            current_user = new_user
            current_short_thread = "chat_001"
            print(
                f"已切换到用户：{current_user}\n"
                f"当前会话：{current_short_thread}"
            )
            continue

        current_full_thread = full_thread_id(
            current_user,
            current_short_thread,
        )
        config = {
            "configurable": {
                "thread_id": current_full_thread,
            }
        }

        old_state = agent.get_state(config)
        old_count = len(old_state.values.get("messages", []))

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config=config,
                context=Context(user_id=current_user),
            )
        except Exception as exc:
            print(f"\nAgent运行失败：\n{str(exc)}")
            continue

        all_messages = result.get("messages", [])
        new_messages = all_messages[old_count:]
        used_tools = show_tool_trace(new_messages)

        response = result.get("structured_response")
        if response is None:
            print("\n没有获得结构化输出。")

            if all_messages:
                last_message = all_messages[-1]
                if getattr(last_message, "content", None):
                    print("\nAgent：", last_message.content)
            continue

        print("\nAgent：", response.answer)
        print("\n========== Structured Output ==========")
        print(
            f"answer：{response.answer}\n"
            f"intent：{response.intent}\n"
            f"success：{response.success}\n"
            f"实际调用 Tools：{used_tools if used_tools else '无'}"
        )
        print("=======================================")


def main() -> None:
    """初始化依赖并启动 Agent CLI。"""
    init_demo_database()

    with (
        SqliteSaver.from_conn_string("memory/agent_memory.db") as memory,
        SqliteStore.from_conn_string("memory/user_memory.db") as store,
    ):
        store.setup()
        agent = create_chat_agent(memory=memory, store=store)
        run_cli(agent=agent, memory=memory, store=store)


if __name__ == "__main__":
    main()
