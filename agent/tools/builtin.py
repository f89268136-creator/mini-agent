"""内置基础工具：时间、计算、文件读写/编辑/检索/列目录、shell。

文件类工具统一受 :mod:`.policy` 的工作区围栏约束；``run_shell`` 默认禁用，
需要在配置里显式设置 ``allow_shell=true`` 才会暴露给模型并允许执行。

``glob`` / ``grep`` 是只读检索工具，标记为 ``concurrency_safe``：同一轮里
模型可以并发发起多次搜索，而不会互相阻塞。
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import fnmatch
import locale
import math
import operator
import os
import re
import signal
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .policy import resolve_workspace_path, shell_enabled, workspace_root
from .registry import register_tool

_MAX_READ_CHARS = 100_000
_MAX_READ_LINES = 2_000
_MAX_READ_LINES_HARD = 5_000
_MAX_WRITE_CHARS = 1_000_000
_MAX_LIST_ENTRIES = 500
_MAX_CALC_CHARS = 200
_MAX_CALC_EXPONENT = 100
_MAX_CALC_RESULT = 1e12
_MAX_COMMAND_CHARS = 4_000
_MAX_SHELL_OUTPUT_CHARS = 4_000
# 管道最多保留的字节数；上限按 UTF-8 最宽字符预留，解码后再按字符截断。
_MAX_SHELL_PIPE_BYTES = _MAX_SHELL_OUTPUT_CHARS * 4
_MAX_PATTERN_CHARS = 500
_MAX_GLOB_RESULTS = 200
_MAX_GREP_MATCHES = 100
_MAX_GREP_FILE_BYTES = 2_000_000
_MAX_GREP_LINE_SEARCH = 4_000
_MAX_GREP_LINE_DISPLAY = 300

# 检索时跳过的目录：依赖、缓存、版本库和本机会话数据都没有检索价值，
# 递归进去只会浪费时间并制造噪音。
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".mini_agent_sessions",
    }
)


@register_tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_ast(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        raise ValueError("表达式里只允许数字常量")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_CALC_EXPONENT:
            raise ValueError(f"指数绝对值不能超过 {_MAX_CALC_EXPONENT}")
        result = _BIN_OPS[type(node.op)](left, right)
        return _check_calc_result(result)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _check_calc_result(_UNARY_OPS[type(node.op)](_eval_ast(node.operand)))
    raise ValueError("表达式里只允许数字和 + - * / // % ** ( )")


def _check_calc_result(result: int | float) -> int | float:
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("计算结果不是有限数字")
    if abs(result) > _MAX_CALC_RESULT:
        raise ValueError(f"计算结果超过上限 {_MAX_CALC_RESULT:g}")
    return result


@register_tool
def calculate(expression: str) -> str:
    """计算一个数学表达式，只支持数字和 + - * / // % ** 与括号，例如 "(1+2)*3"。"""
    if len(expression) > _MAX_CALC_CHARS:
        raise ValueError(f"表达式长度不能超过 {_MAX_CALC_CHARS} 个字符")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc.msg}") from exc
    return str(_eval_ast(tree.body))


@register_tool
def read_file(
    path: str,
    start_line: int = 1,
    max_lines: int = 0,
    line_numbers: bool = True,
) -> str:
    """读取工作区内的文本文件；默认从第 1 行读 2000 行并带行号。

    ``start_line`` 从 1 开始；``max_lines=0`` 表示使用默认行数（最多 5000）；
    ``line_numbers=False`` 关闭行号。单次读取仍受 10 万字符上限约束。
    """
    full = resolve_workspace_path(path, must_exist=True)
    if not full.is_file():
        raise IsADirectoryError(f"不是文件: {path}")
    start_line = max(1, int(start_line))
    limit = int(max_lines) if int(max_lines) > 0 else _MAX_READ_LINES
    limit = min(limit, _MAX_READ_LINES_HARD)

    chunks: list[str] = []
    chars = 0
    last_number = start_line - 1
    truncated_chars = False
    truncated_lines = False
    with open(full, "r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            if number < start_line:
                continue
            if len(chunks) >= limit:
                truncated_lines = True
                break
            rendered = f"{number:>6}\t{line}" if line_numbers else line
            if chars + len(rendered) > _MAX_READ_CHARS:
                truncated_chars = True
                break
            chunks.append(rendered)
            chars += len(rendered)
            last_number = number

    if not chunks:
        if start_line > 1:
            return f"（从第 {start_line} 行起没有内容，文件可能没有这么长）"
        return "（空文件）"
    text = "".join(chunks)
    if not text.endswith("\n"):
        text += "\n"
    if truncated_lines:
        text += (
            f"…（本次读取到第 {last_number} 行；"
            f"可用 start_line={last_number + 1} 继续读取）"
        )
    elif truncated_chars:
        text += f"…（单次读取超过 {_MAX_READ_CHARS} 字符，已截断）"
    return text


@register_tool
def list_directory(path: str = ".") -> str:
    """列出工作区内某个目录下的文件和子目录（最多显示 500 项）。"""
    full = resolve_workspace_path(path, must_exist=True)
    if not full.is_dir():
        raise NotADirectoryError(f"不是目录: {path}")
    entries = sorted(os.listdir(full))
    if not entries:
        return "(空目录)"
    shown = entries[:_MAX_LIST_ENTRIES]
    suffix = ""
    if len(entries) > _MAX_LIST_ENTRIES:
        suffix = f"\n…（共 {len(entries)} 项，仅显示前 {_MAX_LIST_ENTRIES} 项）"
    return "\n".join(shown) + suffix


def _atomic_write_text(full: Path, text: str) -> None:
    """同目录临时文件 + os.replace：写到一半崩溃也不会留下半截文件。"""
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{full.name}.", suffix=".tmp", dir=str(full.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if full.exists():
            # mkstemp 默认 0600，保持原文件权限，避免改写后变得不可读。
            os.chmod(temp_path, full.stat().st_mode)
        os.replace(temp_path, full)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@register_tool(requires_approval=True)
def write_file(path: str, content: str) -> str:
    """把 content 写入工作区内的文件，父目录不存在时自动创建。"""
    if len(content) > _MAX_WRITE_CHARS:
        raise ValueError(f"单次写入不能超过 {_MAX_WRITE_CHARS} 个字符")
    full = resolve_workspace_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(full, content)
    return f"已写入 {len(content)} 个字符到 {path}"


@register_tool(requires_approval=True)
def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    """把文件中的 old 文本替换成 new。

    old 必须唯一匹配（通常要求带上几行上下文）；确实要批量替换时传
    ``replace_all=true``。原文匹配不到或匹配到多处都会报错，避免改错位置。
    """
    if not old:
        raise ValueError("old 不能为空字符串")
    full = resolve_workspace_path(path, must_exist=True)
    with open(full, "r", encoding="utf-8") as handle:
        text = handle.read()
    count = text.count(old)
    if count == 0:
        raise ValueError(f"{path} 中不存在要替换的文本，请先 read_file 确认原文再试")
    if count > 1 and not replace_all:
        raise ValueError(
            f"要替换的文本在 {path} 中出现 {count} 次；"
            "请提供更长的唯一上下文，或设置 replace_all=true"
        )
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    _atomic_write_text(full, updated)
    replaced = count if replace_all else 1
    return f"已替换 {path} 中的 {replaced} 处文本"


def _normalize_pattern(pattern: str) -> str:
    pattern = pattern.replace("\\", "/").strip()
    if not pattern:
        raise ValueError("pattern 不能为空")
    if len(pattern) > _MAX_PATTERN_CHARS:
        raise ValueError(f"pattern 长度不能超过 {_MAX_PATTERN_CHARS} 个字符")
    while pattern.startswith("./"):
        pattern = pattern[2:]
    pattern = pattern.lstrip("/")
    if any(part == ".." for part in pattern.split("/")):
        raise ValueError("pattern 不能包含 ..")
    return pattern


def _match_parts(parts: list[str], pattern_parts: list[str]) -> bool:
    """逐段匹配 glob；``**`` 可以跨任意多层目录（含零层）。"""
    if not pattern_parts:
        return not parts
    head, rest = pattern_parts[0], pattern_parts[1:]
    if head == "**":
        if not rest:
            return True
        return any(_match_parts(parts[index:], rest) for index in range(len(parts) + 1))
    if not parts:
        return False
    return fnmatch.fnmatchcase(parts[0], head) and _match_parts(parts[1:], rest)


def _match_glob(path: str, pattern: str) -> bool:
    return _match_parts(path.split("/"), pattern.split("/"))


def _inside_workspace(path: Path) -> Path | None:
    """解析符号链接后仍位于工作区内才返回真实路径。"""
    try:
        resolved = path.resolve()
        resolved.relative_to(workspace_root())
        return resolved
    except (OSError, ValueError):
        return None


def _walk_files(base: Path):
    """遍历 base 下的普通文件，跳过依赖/缓存目录，不跟随符号链接目录。"""
    if base.is_file():
        yield base
        return
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED_DIRS)
        for name in sorted(files):
            yield Path(root) / name


@register_tool(concurrency_safe=True)
def glob(pattern: str, path: str = ".", max_results: int = _MAX_GLOB_RESULTS) -> str:
    """按 glob 通配符查找工作区内的文件，例如 "**/*.py"、"src/*.md"。

    相对路径按工作区或 ``path`` 目录匹配；自动跳过 .git、node_modules、
    .venv、__pycache__ 等目录。默认最多返回 200 条。
    """
    normalized = _normalize_pattern(pattern)
    base = resolve_workspace_path(path, must_exist=True)
    if not base.is_dir():
        raise NotADirectoryError(f"不是目录: {path}")
    limit = max(1, min(int(max_results), _MAX_GLOB_RESULTS))
    root = workspace_root()

    results: list[str] = []
    truncated = False
    for candidate in _walk_files(base):
        if _inside_workspace(candidate) is None:
            continue
        try:
            rel_workspace = candidate.relative_to(root).as_posix()
            rel_base = candidate.relative_to(base).as_posix()
        except ValueError:
            continue
        if not (
            _match_glob(rel_workspace, normalized)
            or _match_glob(rel_base, normalized)
        ):
            continue
        if len(results) >= limit:
            truncated = True
            break
        results.append(rel_workspace)

    if not results:
        return f"未找到匹配 {pattern!r} 的文件"
    text = "\n".join(results)
    if truncated:
        text += f"\n…（结果达到上限 {limit} 条，可缩小 pattern 范围）"
    return text


@register_tool(concurrency_safe=True)
def grep(
    pattern: str,
    path: str = ".",
    include: str = "",
    max_matches: int = _MAX_GREP_MATCHES,
    case_sensitive: bool = True,
) -> str:
    """按正则表达式搜索工作区内文件内容，返回 "路径:行号: 内容"。

    ``include`` 可按文件名或相对路径过滤，例如 "*.py"、"src/**/*.md"；
    自动跳过 .git、node_modules、.venv、__pycache__ 等目录和二进制文件。
    默认最多返回 100 条匹配。
    """
    if not pattern:
        raise ValueError("pattern 不能为空")
    if len(pattern) > _MAX_PATTERN_CHARS:
        raise ValueError(f"pattern 长度不能超过 {_MAX_PATTERN_CHARS} 个字符")
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"正则表达式语法错误: {exc}") from exc

    include_pattern = _normalize_pattern(include) if include.strip() else ""
    base = resolve_workspace_path(path, must_exist=True)
    limit = max(1, min(int(max_matches), _MAX_GREP_MATCHES))
    root = workspace_root()

    matches: list[str] = []
    truncated = False
    for candidate in _walk_files(base):
        resolved = _inside_workspace(candidate)
        if resolved is None or not resolved.is_file():
            continue
        try:
            rel_workspace = candidate.relative_to(root).as_posix()
            rel_base = candidate.relative_to(base).as_posix()
        except ValueError:
            continue
        if include_pattern and not (
            fnmatch.fnmatchcase(candidate.name, include_pattern)
            or _match_glob(rel_workspace, include_pattern)
            or _match_glob(rel_base, include_pattern)
        ):
            continue

        try:
            with open(resolved, "rb") as handle:
                raw = handle.read(_MAX_GREP_FILE_BYTES + 1)
        except OSError:
            continue
        if len(raw) > _MAX_GREP_FILE_BYTES or b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="replace")

        for number, line in enumerate(text.splitlines(), 1):
            probe = line[:_MAX_GREP_LINE_SEARCH]
            if not regex.search(probe):
                continue
            if len(matches) >= limit:
                truncated = True
                break
            shown = line[:_MAX_GREP_LINE_DISPLAY]
            if len(line) > _MAX_GREP_LINE_DISPLAY:
                shown += "…"
            matches.append(f"{rel_workspace}:{number}: {shown}")
        if truncated:
            break

    if not matches:
        return f"未找到匹配 {pattern!r} 的内容"
    output = "\n".join(matches)
    if truncated:
        output += f"\n…（匹配数达到上限 {limit} 条，可缩小 pattern 或 include 范围）"
    return output


async def _read_limited(
    stream: asyncio.StreamReader | None, limit: int
) -> tuple[bytes, bool]:
    """读取一个管道，超过上限后继续排空但不再保留，避免子进程被管道堵死。"""
    if stream is None:
        return b"", False
    chunks: list[bytes] = []
    size = 0
    truncated = False
    while True:
        chunk = await stream.read(65_536)
        if not chunk:
            break
        if size >= limit:
            truncated = True
            continue
        keep = chunk[: limit - size]
        chunks.append(keep)
        size += len(keep)
        if len(keep) < len(chunk):
            truncated = True
    return b"".join(chunks), truncated


async def _terminate_tree(proc: asyncio.subprocess.Process) -> None:
    """终止子进程及其子进程；超时或取消都要调用，避免命令继续产生副作用。"""
    if proc.returncode is None:
        try:
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(proc.pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.wait()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        finally:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pass


@register_tool(requires_approval=True)
async def run_shell(command: str, timeout: float = 30.0) -> str:
    """在系统 shell 里执行一条命令，返回退出码和输出；默认禁用，需显式开启。

    工作目录固定为工作区；超时或回合被取消时会终止整个进程树，
    输出只保留前 ``_MAX_SHELL_OUTPUT_CHARS`` 个字符。
    """
    if not shell_enabled():
        raise PermissionError(
            "run_shell 未启用；如确认需要执行本机命令，请在配置中设置 allow_shell=true"
        )
    if len(command) > _MAX_COMMAND_CHARS:
        raise ValueError(f"命令长度不能超过 {_MAX_COMMAND_CHARS} 个字符")
    timeout = max(1.0, min(float(timeout), 300.0))
    options: dict[str, object] = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workspace_root()),
        **options,
    )
    try:
        ((stdout_raw, stdout_cut), (stderr_raw, stderr_cut)) = await asyncio.wait_for(
            asyncio.gather(
                _read_limited(proc.stdout, _MAX_SHELL_PIPE_BYTES),
                _read_limited(proc.stderr, _MAX_SHELL_PIPE_BYTES),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        await _terminate_tree(proc)
        return f"命令执行超时（超过 {timeout:g} 秒，已终止进程树）"
    except asyncio.CancelledError:
        await _terminate_tree(proc)
        raise

    await proc.wait()
    encoding = locale.getencoding()
    output = stdout_raw.decode(encoding, errors="replace")
    stderr = stderr_raw.decode(encoding, errors="replace")
    if stderr:
        output += f"\n[stderr]\n{stderr}"
    output = output.strip()
    truncated = stdout_cut or stderr_cut or len(output) > _MAX_SHELL_OUTPUT_CHARS
    if len(output) > _MAX_SHELL_OUTPUT_CHARS:
        output = output[:_MAX_SHELL_OUTPUT_CHARS]
    if truncated:
        output += "\n…（输出过长，已截断）"
    return (
        f"退出码 {proc.returncode}\n{output}"
        if output
        else f"退出码 {proc.returncode}（无输出）"
    )
