"""消息总线包（对应 nanobot 的 bus/）。

    queue.py  两个 asyncio.Queue 组成的 MessageBus + 入站/出站消息类型

渠道与内核互不认识：渠道 publish_inbound 进来、consume_outbound 取回复；
内核 consume_inbound 取消息、publish_outbound 发回复——新增渠道不用改核心。
"""

from .queue import InboundMessage, MessageBus, OutboundMessage

__all__ = ["InboundMessage", "MessageBus", "OutboundMessage"]
