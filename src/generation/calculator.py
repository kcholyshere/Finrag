"""Basic arithmetic tool for Gemini function calling.

Financial questions often need a small calculation on retrieved figures
(deltas, percentage changes, ratios). LLMs are unreliable at arithmetic, so
the model is given this tool instead of computing in its head. A single
expression-based function (rather than one per operation) lets multi-step
maths like "(31654 - 27704) / 27704 * 100" resolve in one tool call.

The expression is evaluated by walking Python's own AST with an explicit
operator whitelist - no eval(), no names, no calls - so model-supplied input
cannot execute anything.
"""

import ast
import operator

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Large exponents can hang the process (Python bignum blow-up) - no financial
# calculation on this report needs anything close to this.
_MAX_EXPONENT = 100


def _evaluate(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Exponent {right} exceeds the allowed maximum of {_MAX_EXPONENT}")
        return _BINARY_OPS[type(node.op)](left, right)
    raise ValueError(f"Unsupported syntax in expression: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Supports numbers, +, -, *, /, % (modulo), ** (power), unary minus, and
    parentheses. Example: "(31654 - 27704) / 27704 * 100". Use this for any
    arithmetic instead of computing mentally.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    tree = ast.parse(expression, mode="eval")
    return float(_evaluate(tree.body))
