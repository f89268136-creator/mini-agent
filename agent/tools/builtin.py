"""内置工具（对应 nanobot 自带的 filesystem / shell 等）。

这里放的是"开箱即用"的 7 个基础工具：时间、计算器、读/写/编辑文件、列目录、执行命令。
RAG（knowledge.py）与网络（web.py）两个子模块的工具注册方式与此完全一样——
写一个普通函数 + @register_tool，即可被模型看见。
"""

from __future__ import annotations

import ast
import os
import subprocess
from datetime import datetime

from .registry import register_tool


@register_tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 计算器允许出现的 AST 节点：数字常量 + 四则/幂/取模运算
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


@register_tool
def calculate(expression: str) -> str:
    """计算一个数学表达式，只支持数字和 + - * / // % ** 与括号，例如 "(1+2)*3"。"""
    node = ast.parse(expression, mode="eval")
    for sub in ast.walk(node):
        if isinstance(sub, _ALLOWED_NODES):
            continue
        if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)):
            continue
        raise ValueError("表达式里只允许数字和 + - * / // % ** ( )")
    # 关掉内置函数，避免执行任意代码
    return str(eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, {}))


@register_tool
def read_file(path: str) -> str:
    """读取指定路径的文本文件内容。"""
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return f.read()


@register_tool
def list_directory(path: str = ".") -> str:
    """列出某个目录下的文件和子目录。"""
    entries = os.listdir(os.path.expanduser(path))
    return "\n".join(entries) if entries else "(空目录)"


@register_tool
def write_file(path: str, content: str) -> str:
    """把 content 写入 path 指定的文件，父目录不存在时自动创建。"""
    full = os.path.expanduser(path)
    parent = os.path.dirname(full)
    if parent:
        # 自动创建父目录：模型常直接写 "out/report.md" 这类多级路径，
        # 让"目录不存在"这种琐碎错误不必占用一次模型纠错回合。
        os.makedirs(parent, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {len(content)} 个字符到 {path}"


@register_tool
def edit_file(path: str, old: str, new: str) -> str:
    """在文件中把 old 文本精确替换成 new（只替换第一次出现处）。"""
    full = os.path.expanduser(path)
    with open(full, "r", encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        # 报可读错误而不是静默什么都不做：模型看到原因后能自我修正
        # （比如它把文件代码风格记错了），这比"改完没生效"好排查得多。
        # nanobot 生产版的 edit_file 还支持 replace_all / 指定第几处出现，教学版只留最小能力。
        raise ValueError(f"{path} 中不存在要替换的文本，请先 read_file 确认原文再试")
    with open(full, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new, 1))
    return f"已替换 {path} 中的 1 处文本"


# shell 输出可能非常长（比如 ls 整个磁盘），截断后再回灌给模型，避免撑爆上下文
_MAX_SHELL_OUTPUT_CHARS = 4000


@register_tool
def run_shell(command: str, timeout: float = 30.0) -> str:
    """在系统 shell 里执行一条命令，返回退出码和输出；超过 timeout 秒强制终止。

    ⚠️ 安全红线：shell=True 等于把整台机器交给模型生成的命令（可删文件、外发数据）。
    本项目只在本地、可信环境的教学场景这样写；生产环境必须沙箱化——nanobot 会用
    bwrap（bubblewrap）等隔离后端限制文件系统与网络后才执行命令，绝不能裸跑。
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # errors="replace"：命令输出混入乱码字节时也不会让工具抛解码异常
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # subprocess 会先杀掉子进程再抛异常；转成文字结果让模型知道是超时而非命令错
        return f"命令执行超时（超过 {timeout} 秒，已强制终止）"

    output = proc.stdout or ""
    if proc.stderr:
        output += f"\n[stderr]\n{proc.stderr}"
    output = output.strip()
    if len(output) > _MAX_SHELL_OUTPUT_CHARS:
        output = output[:_MAX_SHELL_OUTPUT_CHARS] + "…（输出过长，已截断）"
    return f"退出码 {proc.returncode}\n{output}" if output else f"退出码 {proc.returncode}（无输出）"
