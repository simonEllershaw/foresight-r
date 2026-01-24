from collections.abc import Callable
from importlib import import_module

import polars as pl


def create_prefix_or_chain(prefixes: list[str]) -> pl.Expr:
    expr = pl.lit(False)
    for prefix in prefixes:
        expr = expr | pl.col("code").str.starts_with(prefix)
    return expr


def apply_vocab_to_multitoken_codes(
    df: pl.DataFrame, cols: list[str], vocab: list[str]
) -> pl.DataFrame:
    df = df.with_columns(pl.when(pl.col(col).is_in(vocab)).then(col).alias(col) for col in cols)
    for l_col, r_col in zip(cols, cols[1:]):
        df = df.with_columns(pl.when(pl.col(l_col).is_not_null()).then(r_col).alias(r_col))
    return df


def unify_code_names(col: pl.Expr) -> pl.Expr:
    return (
        col.str.to_uppercase().str.replace_all(r"[,.]", "").str.replace_all(" ", "_", literal=True)
    )


def static_class(cls):
    return cls()


def load_function(function_name: str, module_name: str) -> Callable:
    module = import_module(module_name)
    if "." in function_name:
        cls_name, function_name = function_name.split(".")
        module = getattr(module, cls_name)
    return getattr(module, function_name)
