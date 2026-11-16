"""网络工具：必应搜索与网页正文抓取。

默认拒绝访问内网/回环/保留地址（SSRF 防护），并在跟随重定向前后都做校验。
如需访问内网服务，显式设置 ``allow_private_urls=true``。
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
from typing import Any

from .policy import private_urls_enabled
from .registry import register_tool

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_MAX_DOWNLOAD_BYTES = 2_000_000
_MAX_FETCH_CHARS = 4_000


def _strip_tags(markup: str) -> str:
    text = re.sub(r"<[^>]+>", " ", markup)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def validate_url(url: str) -> None:
    """校验协议与目标地址；默认只允许访问公网 HTTP(S)。"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"只支持 http/https URL，收到: {parsed.scheme or '(空)'}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL 缺少主机名: {url}")
    if private_urls_enabled():
        return

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"无法解析主机名: {host}") from exc
    for info in infos:
        raw_ip = info[4][0].split("%", 1)[0]
        try:
            address = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if not address.is_global:
            raise PermissionError(
                f"拒绝访问内网/保留地址 {address}（host={host}）；"
                "如确需访问，请设置 allow_private_urls=true"
            )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _http_get(url: str, timeout: float = 20.0) -> str:
    validate_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"},
    )
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(_MAX_DOWNLOAD_BYTES + 1)
        truncated = len(raw) > _MAX_DOWNLOAD_BYTES
        raw = raw[:_MAX_DOWNLOAD_BYTES]
        charset = response.headers.get_content_charset() or "utf-8"
    text = raw.decode(charset, errors="ignore")
    if truncated:
        text += "\n…（响应体过大，已截断）"
    return text


@register_tool
def web_search(query: str) -> str:
    """用必应搜索网络，返回若干条结果的标题、链接和摘要。"""
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&setlang=zh-CN"
    page = _http_get(url)

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
    """抓取公网网页并提取正文（自动去掉脚本、样式和 HTML 标签）。"""
    page = _http_get(url)
    page = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", page)
    text = _strip_tags(page)
    if len(text) > _MAX_FETCH_CHARS:
        text = text[:_MAX_FETCH_CHARS] + "…（内容过长，已截断）"
    return text or "未提取到正文。"
