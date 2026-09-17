"""Safe calculator tool."""

import ast
import operator

from langchain.tools import tool


ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_calculate(node):
    """递归计算经过白名单限制的 AST 表达式。"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只允许数字")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in ALLOWED_BINARY_OPERATORS:
            raise ValueError("不支持该运算")

        left = _safe_calculate(node.left)
        right = _safe_calculate(node.right)

        if operator_type is ast.Pow and abs(right) > 10:
            raise ValueError("指数过大")

        return ALLOWED_BINARY_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in ALLOWED_UNARY_OPERATORS:
            raise ValueError("不支持该运算")
        return ALLOWED_UNARY_OPERATORS[operator_type](_safe_calculate(node.operand))

    raise ValueError("表达式包含不允许的内容")


@tool
def calculator(expression: str) -> str:
    """执行数学计算。当用户要求加减乘除、百分比、括号运算、幂运算或其他明确的算术计算时使用。"""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_calculate(parsed.body)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "计算失败：不能除以0。"
    except Exception as exc:
        return f"计算失败：{str(exc)}"
