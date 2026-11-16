"""本地技能包：把 prompt + 说明文件打包成可按需调用的能力。

目录约定（项目级同名技能覆盖用户级）：

    ~/.mini_agent/skills/<name>/SKILL.md
    <workspace>/.mini_agent_skills/<name>/SKILL.md
    <workspace>/.mini_agent/skills/<name>/SKILL.md

也接受直接放在根目录下的 ``<name>.md``。文件开头的 YAML frontmatter 只解析
``name`` / ``description`` 两个简单字段，不引入额外依赖。

技能目前只提供提示词正文：调用时把正文作为临时系统内容注入当轮请求，
不会自动执行脚本，也不会写入会话历史。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SKILL_FILENAME = "SKILL.md"
_MAX_SKILLS = 200
_MAX_BODY_CHARS = 50_000
_MAX_DESCRIPTION_CHARS = 200
_META_READ_CHARS = 8_192
_NAME_RE = re.compile(r"^[\w.-]{1,64}$", re.UNICODE)
_SOURCE_LABELS = {"user": "用户", "project": "项目"}


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    source: str

    def load_body(self) -> str:
        """按需读取正文；frontmatter 会被剥离，超长内容截断。"""
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("读取技能文件失败 %s: %s", self.path, exc)
            return ""
        _, body = _split_frontmatter(text)
        body = body.strip()
        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + "\n…（技能正文过长，已截断）"
        return body

    @property
    def source_label(self) -> str:
        return _SOURCE_LABELS.get(self.source, self.source)


class SkillRegistry:
    """扫描技能目录并维护 name -> Skill 映射（项目级覆盖用户级）。"""

    def __init__(
        self,
        roots: list[tuple[str, str | Path]] | None = None,
        *,
        enabled: bool = True,
        catalog_max_chars: int = 4_000,
    ) -> None:
        self.enabled = enabled
        self.roots = [(source, Path(path).expanduser()) for source, path in roots or []]
        self.catalog_max_chars = max(0, catalog_max_chars)
        self._skills: dict[str, Skill] = {}
        self._catalog = ""

    def scan(self) -> int:
        """重新扫描全部根目录；返回发现的技能数。"""
        self._skills = {}
        if not self.enabled:
            self._catalog = ""
            return 0
        for source, root in self.roots:
            self._scan_root(root, source)
        self._catalog = self._build_catalog()
        return len(self._skills)

    def all(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda skill: skill.name)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def catalog(self) -> str:
        """供系统提示使用的技能目录；未扫描或为空时返回空串。"""
        return self._catalog

    def _scan_root(self, root: Path, source: str) -> None:
        if not root.is_dir():
            return
        try:
            entries = sorted(root.iterdir())
        except OSError as exc:
            logger.warning("无法读取技能目录 %s: %s", root, exc)
            return
        for entry in entries:
            if len(self._skills) >= _MAX_SKILLS:
                logger.warning("技能数量达到上限 %d，其余文件已忽略", _MAX_SKILLS)
                return
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                candidate = entry / _SKILL_FILENAME
                if not candidate.is_file():
                    continue
                fallback_name = entry.name
            elif entry.is_file() and entry.suffix.lower() == ".md":
                candidate = entry
                fallback_name = entry.stem
            else:
                continue
            skill = _load_skill(candidate, source, fallback_name)
            if skill is None:
                continue
            existing = self._skills.get(skill.name)
            if existing is not None and existing.source != skill.source:
                logger.debug(
                    "技能 %s 被 %s 级同名技能覆盖", skill.name, skill.source_label
                )
            self._skills[skill.name] = skill

    def _build_catalog(self) -> str:
        skills = self.all()
        if not skills or self.catalog_max_chars <= 0:
            return ""
        lines: list[str] = []
        used = 0
        for skill in skills:
            line = f"- {skill.name}（{skill.source_label}）：{skill.description}"
            if used + len(line) + 1 > self.catalog_max_chars:
                lines.append("…（技能目录过长，其余已省略）")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)


def _load_skill(path: Path, source: str, fallback_name: str) -> Skill | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            head = handle.read(_META_READ_CHARS)
    except OSError as exc:
        logger.warning("读取技能文件失败 %s: %s", path, exc)
        return None
    meta, body_preview = _split_frontmatter(head)
    name = str(meta.get("name") or fallback_name).strip()
    if not _NAME_RE.match(name):
        logger.warning("技能名不合法，已跳过: %s（来自 %s）", name, path)
        return None
    description = str(meta.get("description") or _first_line(body_preview)).strip()
    description = " ".join(description.split())
    if len(description) > _MAX_DESCRIPTION_CHARS:
        description = description[:_MAX_DESCRIPTION_CHARS] + "…"
    return Skill(
        name=name,
        description=description or "（无描述）",
        path=path,
        source=source,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析开头的 ``---`` frontmatter；没有或未闭合时原样返回正文。"""
    normalized = text.lstrip("\ufeff")
    lines = normalized.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, normalized
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            meta = _parse_meta(lines[1:index])
            return meta, "\n".join(lines[index + 1 :])
    return {}, normalized


def _parse_meta(lines: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        if key not in ("name", "description"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        meta[key] = value
    return meta


def _first_line(body: str) -> str:
    for line in body.splitlines():
        text = line.strip().lstrip("#").strip()
        if text:
            return text
    return ""
