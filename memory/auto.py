"""回合推进时的确定性记忆固化。

模型偶尔会漏调 memory_save，这里用少数高置信句式兜底：只认"我叫 X"
这类身份陈述，以及用户明确说"记住 …"的指令，其余交给模型自己判断。
规则故意保守——把提问（"我叫什么"）或普通陈述误存成事实，比漏存更糟。
"""

from __future__ import annotations

import re

_MAX_FACTS_PER_TURN = 3
_MAX_FACT_CHARS = 200
_NAME_MAX_CHARS = 20
# 名字里出现这些词说明是提问（"我叫什么"）而不是陈述。
_NOT_A_NAME = ("谁", "什么", "啥", "哪", "几", "多少", "吗")
# 名字的终止符：标点、空白，以及"我叫雄安你呢"这类紧贴的第二人称。
_NAME_STOP = r"[^\s，。,.！!？?；;：:、你您]+"

_NAME_PATTERNS = (
    re.compile(rf"我(?:的名字)?叫(?P<name>{_NAME_STOP})"),
    re.compile(rf"我的名字是(?P<name>{_NAME_STOP})"),
    re.compile(rf"叫我(?P<name>{_NAME_STOP})"),
)
_REMEMBER = re.compile(r"记住[：:，,\s]*(?P<fact>.+)")


def extract_memory_facts(text: str) -> list[str]:
    """从一条用户消息里提取值得持久化的高置信事实。"""
    facts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        for pattern in _NAME_PATTERNS:
            match = pattern.search(stripped)
            if match is None:
                continue
            name = match.group("name").strip()
            if _is_name(name):
                facts.append(f"用户的名字是{name}")
            break

        match = _REMEMBER.search(stripped)
        if match is not None:
            content = match.group("fact").strip(" ：:，,。.！!？?")
            if content:
                facts.append(f"用户要求记住：{content}"[:_MAX_FACT_CHARS])

    # 同一句话里"我叫X"和"记住我叫X"可能各命中一次，去重后限量。
    return list(dict.fromkeys(facts))[:_MAX_FACTS_PER_TURN]


def _is_name(name: str) -> bool:
    if not name or len(name) > _NAME_MAX_CHARS:
        return False
    return not any(word in name for word in _NOT_A_NAME)
