"""部件：网络工具（对应 nanobot 里的 web search / fetch）。

- web_search：用必应搜索（www.bing.com 国内通常可访问），解析结果标题/链接/摘要；
- fetch_webpage：抓取某个网址，粗提取正文（去掉脚本、样式和 HTML 标签）。

两者都只用标准库（urllib + 正则），零第三方依赖。
刻意保持同步：工具由 runner 用 asyncio.to_thread 丢到工作线程执行，
线程里阻塞式网络调用无所谓，不必为了 async 而 async。
注意：搜索引擎返回的 HTML 结构可能变化，正则解析属于"轻量可用"方案；
生产中建议改用官方搜索 API。
"""

from __future__ import annotations

import html
import re
import urllib.request
from urllib.parse import quote

from .registry import register_tool

# 搜索引擎会拦截没有浏览器标识的请求，需要带 User-Agent
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_MAX_FETCH_CHARS = 4000


def _strip_tags(markup: str) -> str:
    """去掉 HTML 标签并还原 &amp; 等实体，压缩多余空白。"""
    text = re.sub(r"<[^>]+>", " ", markup)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _http_get(url: str, timeout: float = 20.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


@register_tool
def web_search(query: str) -> str:
    """用必应搜索网络，返回若干条结果的标题、链接和摘要。"""
    url = f"https://www.bing.com/search?q={quote(query)}&setlang=zh-CN"
    page = _http_get(url)

    # 每个自然结果在 <li class="b_algo"> ... </li> 中
    blocks = re.findall(r'<li class="b_algo".*?</li>', page, flags=re.DOTALL)
    results: list[str] = []
    for block in blocks[:5]:
        link_match = re.search(
            r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            flags=re.DOTALL,
        )
        if not link_match:
            continue
        link = link_match.group(1)
        title = _strip_tags(link_match.group(2))
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, flags=re.DOTALL)
        snippet = _strip_tags(snippet_match.group(1)) if snippet_match else ""
        results.append(f"{title}\n{link}\n{snippet}")

    return "\n\n".join(results) if results else "未解析到搜索结果（可能被反爬拦截）。"


@register_tool
def fetch_webpage(url: str) -> str:
    """抓取网页并提取正文（自动去掉脚本、样式和标签），用于阅读某个链接的内容。"""
    page = _http_get(url)
    # 先去掉 script / style 整块，避免把代码当正文
    page = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", page)
    text = _strip_tags(page)
    if len(text) > _MAX_FETCH_CHARS:
        text = text[:_MAX_FETCH_CHARS] + "…（内容过长，已截断）"
    return text or "未提取到正文。"
