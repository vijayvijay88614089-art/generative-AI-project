import ast

ALLOWED_NODES = (
    ast.Expression, ast.Call, ast.Name, ast.Load, ast.Attribute, ast.Constant,
    ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.BinOp,
    ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.keyword, ast.IfExp,
)
ALLOWED_METHODS = {
    "sum", "mean", "median", "min", "max", "count", "nunique", "idxmax", "idxmin",
    "groupby", "sort_values", "head", "tail", "nlargest", "nsmallest", "value_counts",
    "isna", "notna", "to_dict", "tolist", "item", "round", "astype", "fillna", "dropna", "unique",
    "assign", "corr", "describe", "reset_index", "set_index", "resample", "agg",
    "dt", "to_period", "date", "to_datetime", "str", "lower", "contains", "eq", "gt", "lt",
    "isin", "between", "loc", "shape", "columns", "empty", "values", "index",
}
ALLOWED_NAMES = {"df", "len", "int", "float", "bool", "abs", "round", "pd", "np", "True", "False", "None"}


def validate_expression(expression: str) -> str:
    if len(expression) > 1500:
        raise ValueError("Expression exceeds the 1,500 character limit.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc.msg}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise ValueError(f"Disallowed name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ALLOWED_METHODS:
                raise ValueError(f"Disallowed attribute: {node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id not in {"len", "int", "float", "bool", "abs", "round"}:
                raise ValueError(f"Disallowed function: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr not in ALLOWED_METHODS:
                raise ValueError(f"Disallowed method: {func.attr}")
    return expression.strip()
