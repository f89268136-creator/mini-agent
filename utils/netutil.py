"""URL 判定的共享小工具（配置层与 Provider 层共用）。"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def is_loopback_url(url: str) -> bool:
    """判断 URL 是否指向本机回环地址（不解析 DNS，只看字面主机名）。"""
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
