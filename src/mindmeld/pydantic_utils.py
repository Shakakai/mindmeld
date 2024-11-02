from typing import Any, Optional, Literal
from pydantic import BaseModel
from datetime import datetime


def header(
        label: str,
        value_type: Literal["dict", "list", "string", "number", "type", "null"],
        level: int,
        add_linebreak: bool = True
):
    h = "#" * (level + 1)
    result = f"{h} {label} [type={value_type}]"
    if add_linebreak:
        result += "\n"
    return result


def model_to_md(item: BaseModel, label: Optional[str] = None, level: int = 0) -> str:
    label = label or item.__class__.__name__
    result = header(label, "dict", level)
    for name, field in item.model_fields.items():
        sublabel = name
        if field.description is not None:
            sublabel += f" ({field.description})"
        value = getattr(item, name)
        result += pydantic_to_md(value, sublabel, level=level + 1)
    return result


def dict_to_md(item: dict, label: Optional[str] = None, level: int = 0) -> str:
    result = header(label, "dict", level) if label is not None else ""
    for key, value in item.items():
        result += pydantic_to_md(value, key, level=level + 1)
    return result


def itr_to_md(item: list | tuple | set, label: Optional[str] = None, level: int = 0) -> str:
    result = header(label, "list", level) if label is not None else ""
    count = 0
    for i in item:
        result += pydantic_to_md(i, label=f"Item {count}", level=level + 1)
        count += 1
    return result


def dt_to_md(item: datetime, label: Optional[str] = None, level: int = 0) -> str:
    if item is None:
        value_type = "null"
        value = "null"
    else:
        value_type = "date"
        value = item.strftime("%Y-%m-%dT%H:%M:%SZ")
    label = header(label, value_type, level, add_linebreak=False)
    return f"{label}: {value}\n"


def basic_to_md(item: str | int | float | None, label: str, level: int = 0) -> str:
    if item is None:
        value_type = "null"
    else:
        value_type = "string" if isinstance(item, str) else "number"
    label = header(label, value_type, level, add_linebreak=False)
    return f"{label}: {item}\n"


def pydantic_to_md(item: Any, label: Optional[str] = None, level: int = 0) -> str:
    if isinstance(item, BaseModel):
        return model_to_md(item, label=label, level=level)
    if isinstance(item, dict):
        return dict_to_md(item, label=label, level=level)
    if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, set):
        return itr_to_md(item, label=label, level=level)
    if isinstance(item, datetime):
        return dt_to_md(item, label=label, level=level)
    if isinstance(item, str) or isinstance(item, int) or isinstance(item, float) or item is None:
        return basic_to_md(item, label=label, level=level)
    if isinstance(item, type):
        return header(item.__name__, "type", level)
    raise Exception("Unknown type")


def model_to_vs(item: BaseModel) -> str:
    result = ""
    for name, field in item.model_fields.items():
        value = getattr(item, name)
        result += pydantic_to_vs(value)
    return result


def dict_to_vs(item: dict) -> str:
    result = ""
    for key, value in item.items():
        result += pydantic_to_vs(value)
    return result


def itr_to_vs(item: list | tuple | set) -> str:
    result = ""
    for i in item:
        result += pydantic_to_vs(i)
    return result


def dt_to_vs(item: datetime) -> str:
    if item is None:
        return ""

    value = item.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{value}\n"


def basic_to_vs(item: str | int | float | None) -> str:
    if item is None:
        return ""

    return f"{item}\n"


def pydantic_to_vs(item: Any) -> str:
    if isinstance(item, BaseModel):
        return model_to_vs(item)
    if isinstance(item, dict):
        return dict_to_vs(item)
    if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, set):
        return itr_to_vs(item)
    if isinstance(item, datetime):
        return dt_to_vs(item)
    if isinstance(item, str) or isinstance(item, int) or isinstance(item, float) or item is None:
        return basic_to_vs(item)
    if isinstance(item, type):
        return ""
    raise Exception("Unknown type")
