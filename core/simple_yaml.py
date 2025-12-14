"""Very small YAML subset parser for configs without external deps."""
from typing import Any, Dict, List


def parse_value(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if value.startswith("0") and "." in value:
            return float(value)
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        if value.lower() == "null":
            return None
        return value


def load(lines: List[str], indent: int = 0):
    data: Dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        key, _, rest = line.strip().partition(":")
        if rest.strip() == "":
            # nested block
            sub_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent <= current_indent:
                    break
                sub_lines.append(next_line)
                i += 1
            data[key] = load(sub_lines, indent=current_indent + 2)
            continue
        else:
            data[key] = parse_value(rest)
            i += 1
    return data


def safe_load(stream: str) -> Dict[str, Any]:
    lines = stream.splitlines()
    return load(lines, indent=0)
