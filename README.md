# mini_agent：参考 nanobot 的最小可用 Agent（asyncio 完整版）

一个部件完整、初学者能读懂的 AI Agent，模型调用使用官方 **openai SDK**。它复刻了 nanobot 最核心的机制：
**ReAct 循环 + 工具调用 + 会话记忆 + 可替换模型 + RAG 检索 + 网络搜索**，并补齐了 nanobot 的经典骨架：
**全异步 asyncio + MessageBus + 按会话分片 + 多渠道 + JSONL 持久化 + 重试**，整体向 nanobot 早期最小版看齐。

- 运行环境：Python 3.11+；依赖：官方 `openai` 包（首次运行前安装一次：`pip install openai`）

---

## 一、该有的部件（对照 nanobot）

| 部件 | 文件 | 职责 | nanobot 对应 |
|---|---|---|---|
| 数据结构 | `providers/base.py` | `Message` / `ToolCall` / `LLMResponse` / `Usage`；`has_tool_calls` 与 `should_execute_tools` 两个判定 | `providers/base.py` |
| 对话模型 | `providers/openai_compat.py` | `LLMProvider` 抽象 + `OpenAICompatProvider`（AsyncOpenAI）+ `RetryingProvider` 重试包装 | `providers/` |
| 嵌入模型 | `providers/embeddings.py` | `Embedder` 抽象 + OpenAI 兼容 `/embeddings`（async） | `providers/` 的 embeddings |
| 基础工具 | `agent/tools/registry.py` + `builtin.py` | 工具注册表、`@register_tool`、参数 schema 自动生成、7 个基础工具 | `agent/tools/` |
| RAG 知识库 | `agent/tools/knowledge.py` | 文本切块、纯 Python 向量库、余弦检索 + 2 个知识库工具 | RAG 相关能力 |
| 网络工具 | `agent/tools/web.py` | 必应搜索 + 网页正文抓取 | web search / fetch |
| 消息总线 | `bus/queue.py` | `MessageBus`（inbound/outbound 双队列）+ `InboundMessage` / `OutboundMessage` | `bus/queue.py` + `bus/events.py` |
| 会话仓库 | `session/manager.py` | `Session` / `SessionManager`：按 key 存历史、截断、JSONL 增量落盘 | `session/manager.py` |
| 配置 | `config.py` | `MiniConfig` + `load_config`（默认值 < JSON 文件 < 环境变量） | `config/schema.py` + `config/loader.py` |
| 渠道基类 | `channels/base.py` | `Channel` 抽象：`start()` / `send()` / `stop()` | `channels/base.py` |
| CLI 渠道 | `channels/cli.py` | stdin 读入丢总线；回复打印到终端（`input()` 用 to_thread 包） | 本地 CLI 渠道 |
| Echo 渠道 | `channels/echo.py` | TCP 行协议服务端，演示"第二个渠道"如何接入总线 | Telegram / WebSocket 等平台渠道 |
| 渠道工厂 | `channels/__init__.py` | `build_channels`：名字列表 -> 渠道实例字典 | `channels/manager.py` |
| ReAct 引擎 | `agent/runner.py` | 调模型 -> 判定工具 -> 执行（to_thread）-> 回灌 -> 再调，循环 | `agent/runner.py` |
| 编排循环 | `agent/loop.py` | 总线消费、按会话分片（一队列一 worker）、命令层、落盘、出站路由 | `agent/loop.py` |
| 最简门面 | `agent/agent.py` | 单会话 + 历史记忆，一个类就能跑（无总线无渠道） | `agent/loop.py` 的退化形态 |
| 完整门面 | `sdk.py` | `MiniBot`：装配总线/会话/引擎/渠道，对外 `chat()` / `start()` / `stop()` | `nanobot.py` |
| 命令行入口 | `__main__.py` | `asyncio.run` 驱动完整系统 + 启动横幅 + 优雅停机 | `cli/commands.py` |

内置 11 个工具（导入包时全部注册）：

- 基础（`agent/tools/builtin.py`，7 个）：`get_current_time`、`calculate`、`read_file`、`write_file`、`edit_file`、`list_directory`、`run_shell`
- RAG（`agent/tools/knowledge.py`，2 个）：`knowledge_ingest`、`knowledge_search`
- 网络（`agent/tools/web.py`，2 个）：`web_search`、`fetch_webpage`

---

## 二、架构图

```
   外部世界                          进程内（一个 asyncio 事件循环）
 ┌──────────────┐
 │ 渠道 Channel  │  CliChannel：读 stdin    EchoChannel：监听 TCP 8766
 └──────┬───────┘
        │ 收到输入 → await publish_inbound
        ▼
  MessageBus · inbound 队列
        │ _inbound_dispatcher（总线唯一消费者：只分片，不处理）
        ▼
  按 session_key 分片：一个会话一条队列 + 一个专属 worker
    "cli:local" → [queue] → worker ┐ 同会话串行：上一条处理完才取下一条
    "echo:…"    → [queue] → worker ┘ 跨会话并行：不同 worker 各自 await
        ▼
  worker → handle_inbound → handle_text
      ├─ 命令层：/reset、/help（不进模型、不写历史）
      ├─ SessionManager：取/建会话、追加 user 消息、（回复后）增量落盘
      └─ AgentRunner.run：ReAct（await provider.chat ⇄ 执行工具回灌）
        ▼
  回复 → await publish_outbound → 出站队列 → _outbound_dispatcher → 渠道.send(msg) → 外部世界
```

一句话记忆：**渠道只管收发，总线负责解耦，loop 按会话分片，runner 跑一条会话里的 ReAct**；同会话串行、跨会话并行。

---

## 三、一次请求的完整流转

1. 渠道收到输入（CLI 从 stdin 读一行；echo 从 TCP 读一行），包装成
   `InboundMessage(channel, sender, text)`，`await bus.publish_inbound(...)` 丢进入站队列；
2. loop 的 `_inbound_dispatcher` 从总线取出消息，按 `msg.session_key`
   （等于 `f"{channel}:{sender}"`）找该会话专属队列；第一次见到就现场建"队列 + worker"，
   然后把消息入队（分发器只入队不处理，永远不被慢会话卡住）；
3. 该会话的 worker 逐条取出消息，`handle_inbound` -> `handle_text` 处理：
   - 命令层：`/reset` 清空会话（保留 system 人设）并立即重写落盘文件，回复"（对话已清空）"；
     `/help` 返回命令说明；两个命令都不进模型、不写历史（`/exit` 由 CLI 渠道处理成退出信号）；
   - 普通消息：`sessions.get_or_create(key)` 取会话，追加 user 消息，再取 `get_history()`
     快照（截断到 `history_limit`、清理孤儿 tool 消息）交给引擎；
4. `AgentRunner.run` 跑 ReAct：`await provider.chat(消息 + 全部工具 schema)`；模型请求工具
   -> `asyncio.to_thread` 执行 -> 结果以 `role="tool"` 回灌 -> 再问模型，直到拿到最终文本
   （或到 `max_iterations` 兜底）；
5. 最终答案写回会话并 `await sessions.save(session)` 增量落盘；再包装成 `OutboundMessage`
   回总线，由 `_outbound_dispatcher` 按 channel 路由给渠道 `send(msg)` 发回外部世界。

---

## 四、三种用法

### 1) 最简路线：一个 Agent 类（逃课版）

不关心总线 / 渠道 / 持久化时，用 `Agent` 就够了（无总线、无渠道、单会话）：

```python
import asyncio
from mini_agent import Agent
from mini_agent.providers import provider_from_env

async def main() -> None:
    agent = Agent(provider_from_env())   # provider 从环境变量构造，自带重试
    print(await agent.chat("现在几点？"))

asyncio.run(main())
```

### 2) SDK：程序里嵌入完整系统

一行版：`answer = await MiniBot.from_config().chat("你好")`（默认会话键 `"sdk:local"`）。
也可以直接构造配置对象，适合脚本 / 测试 / 被别的程序调用：

```python
import asyncio

from mini_agent import MiniBot, MiniConfig

async def main() -> None:
    config = MiniConfig(model="qwen-plus", channels=[])   # 纯 SDK：不装配任何渠道
    bot = MiniBot(config)
    print(await bot.chat("你好"))                          # 走 loop.handle_text，历史照常落盘
    print(await bot.chat("刚才我说了什么？", session_key="sdk:local"))
    await bot.aclose()                                     # 也支持 async with bot: ...

asyncio.run(main())
```

`MiniBot` 的装配是同步的（构造不发网络请求、不起后台任务）；只有调用 `start()` 才启动总线分发任务与各渠道，`stop()` / `aclose()` 负责优雅停机。

### 3) 完整 CLI：多轮对话 + 多渠道

直接运行 `python -m mini_agent`，进入后输入问题；`/reset` 清空当前会话；`/help` 查看命令；`/exit` 退出。
想验证"多渠道"，让 CLI 和 echo 同时开着：

```powershell
$env:MINI_CHANNELS="cli,echo"
python -m mini_agent
```

再从**另一个终端**运行这段小脚本（连上 127.0.0.1:8766，发一行收一行）：

```python
# echo_client.py：验证 echo 渠道
import socket
with socket.create_connection(("127.0.0.1", 8766)) as sock:
    sock.sendall("你好，介绍一下你自己\n".encode("utf-8"))
    reply = sock.makefile("rb").readline().decode("utf-8")
    print("echo 渠道回复:", reply.strip())
```

两个渠道的会话互不干扰（`cli:local` 与 `echo:peer:127.0.0.1:xxxxx` 是两个 session_key），
但共用同一个 Agent 引擎、同一份工具集、同一个事件循环。

---

## 五、配置

优先级（后者覆盖前者）：**默认值 < `.mini_agent.json` < 环境变量**；未显式指定路径时，
`load_config()` 在当前工作目录找 `.mini_agent.json`，不存在就用默认值。
字段名与 `MiniConfig` 完全一致的配置文件示例：

```json
{
  "model": "qwen-plus",
  "api_key": "sk-你的密钥（也可以留空，走环境变量）",
  "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "system_prompt": "你是一个乐于助人的 AI 助手。",
  "max_iterations": 10,
  "history_limit": 50,
  "sessions_dir": ".mini_agent_sessions",
  "channels": ["cli", "echo"],
  "echo_port": 8766,
  "retry_max_attempts": 3,
  "retry_base_delay": 0.5,
  "timeout": 60.0
}
```

环境变量表（优先级最高，不用改文件就能切换环境）：

| 环境变量 | 对应字段 | 说明 |
|---|---|---|
| `MODEL` | `model` | 对话模型名（默认 `gpt-4o-mini`） |
| `OPENAI_API_KEY` | `api_key` | 密钥；本地模型可留空 |
| `OPENAI_BASE_URL` | `api_base` | 端点（默认 OpenAI 官方） |
| `MINI_SESSIONS_DIR` | `sessions_dir` | 会话 JSONL 目录（默认 `.mini_agent_sessions`） |
| `MINI_CHANNELS` | `channels` | 逗号分隔，如 `cli,echo`（默认 `cli`） |
| `MAX_ITERATIONS` | `max_iterations` | 单轮对话工具循环上限（默认 10） |
| `HISTORY_LIMIT` | `history_limit` | 发给模型的历史消息上限（默认 50） |
| `ECHO_PORT` | `echo_port` | echo 渠道监听端口（默认 8766） |
| `MINI_RETRY_MAX_ATTEMPTS` | `retry_max_attempts` | LLM 请求重试次数（默认 3） |
| `MINI_RETRY_BASE_DELAY` | `retry_base_delay` | 重试指数退避基数秒数（默认 0.5） |
| `MINI_TIMEOUT` | `timeout` | 单次 LLM 请求超时秒数（默认 60.0） |

---

## 六、运行方法

在 **nanobot-main 项目根目录**执行（`mini_agent` 是其中的包）。若尚未安装依赖，先执行一次 `pip install openai`。

### 1) 接入阿里云百炼 DashScope（通义千问）

```powershell
$env:MODEL="qwen-plus"
$env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_API_KEY=$env:DASHSCOPE_API_KEY
python -m mini_agent
```

### 2) 接入本地 Ollama（无需密钥）

```powershell
$env:MODEL="qwen2.5:7b"
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
python -m mini_agent
```

### 3) 接入 OpenAI 官方

```powershell
$env:MODEL="gpt-4o-mini"
$env:OPENAI_API_KEY="sk-你的密钥"
python -m mini_agent
```

在 PyCharm 中运行：直接对项目根目录的 `run.py` 点绿色三角（首次会提示粘贴 Key），无需手动配置。

---

## 七、会话与持久化

- **一条会话一个文件**：`SessionManager` 把 session_key 清洗成安全文件名（`"cli:local"` ->
  `cli_local.jsonl`）存进 `sessions_dir`，每行一条 JSON（`ensure_ascii=False`，中文直接可读）；
- **增量 append**：落盘只 append 水位线之后的新消息，不整文件重写；重启后 `get_or_create` 恢复历史；
- **/reset 的正确性**：清空后历史变短，若继续"增量追加"会错位，所以此时改为整文件重写
  （只剩 system 一行）并重置水位线，之后的增量才继续正确；
- **历史截断**：发给模型的是 `get_history()` 快照——保留 system + 最近 `history_limit` 条，
  并丢弃"找不到发起调用的 assistant"的孤儿 tool 消息；
- 一条会话文件长这样：

```
{"role": "system", "content": "你是一个乐于助人的 AI 助手。..."}
{"role": "user", "content": "你好"}
{"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
```

删除某个 `.jsonl` 文件等价于删除那条会话的记忆；删掉整个目录则是"失忆重启"。

---

## 八、RAG 怎么用（典型流程）

RAG 让模型基于你提供的文档回答，分两步——先"喂"文档，再提问：

```
你: 请把 ./notes.txt 加入知识库
助手: 已从 ./notes.txt 摄入 6 个片段，知识库当前共 6 个片段。
你: notes.txt 里关于上线流程说了什么？
```

第二个问题时模型会自动调用 `knowledge_search` 检索相关片段，再据此回答，而不是凭记忆编造。原理（见 `agent/tools/knowledge.py`）：

1. **切块**：长文档按 500 字、50 字重叠切成小段（`chunk_text`）；
2. **嵌入**：每段经 `/embeddings` 变成向量（`await embedder.embed(pieces)`，不卡事件循环）；
3. **检索**：问题向量化后（`await kb.search(query, top_k)`），用余弦相似度取最相关的几段；
4. **生成**：相关片段作为参考交给对话模型。

两个知识库工具对外仍是普通同步函数——它们跑在 runner 的 `asyncio.to_thread` 工作线程里
（线程内没有事件循环），所以在工具末尾用 `asyncio.run` 驱动 `await kb.ingest/search` 正合适。

> 知识库保存在内存中（重启清空），生产可换成真正的向量数据库；嵌入模型可用
> `EMBEDDING_MODEL` 环境变量指定，不填按端点自动选择（DashScope 用 `text-embedding-v3`）。

---

## 九、网络搜索与网页抓取

- `web_search("关键词")`：用必应搜索，返回标题、链接、摘要（必应国内可访问）；
- `fetch_webpage("https://...")`：打开链接并提取正文（去掉脚本/样式/标签）。

典型用法：先搜索拿到链接，再抓取具体页面内容，最后总结。两个工具保持同步实现，
同样由 runner 丢进 `asyncio.to_thread` 执行，不会阻塞事件循环。

---

## 十、离线自测（不需要模型和网络）

```powershell
$env:PYTHONIOENCODING="utf-8"     # Windows GBK 控制台建议设置，避免中文输出报错
python -m mini_agent.tests.test_offline            # ReAct 主回路 / 工具守卫 / 重试策略
python -m mini_agent.tests.test_rag_offline        # 切块 / 假嵌入 / 向量检索
python -m mini_agent.tests.test_pipeline_offline   # 总线 / 会话分片 / 持久化 / 命令 / SDK 全链路 E2E
```

三条都看到 `离线自测通过` 即说明机制正确：`test_offline` 验证 ReAct 回路的工具回灌、
`should_execute_tools` 守卫（finish_reason 异常时不得执行工具）与 `RetryingProvider`
（瞬时错误重试、非瞬时错误立即抛、`finish_reason="error"` 按失败重试）；`test_rag_offline`
用二维关键词假嵌入验证 `await kb.ingest/search`、空库检索与 ingest 返回值；`test_pipeline_offline`
用真装配（总线 + 会话仓库 + 引擎 + 假渠道）跑 E2E——同会话串行且按序、跨会话并行（门闩法，
不靠时间差）、JSONL 落盘与重启恢复、`/reset` 后水位线正确、MiniBot 换假 provider 后的 SDK 路径。

---

## 十一、添加自己的工具

写一个普通函数并加 `@register_tool`，参数 schema 从类型注解自动生成：

```python
from mini_agent.agent.tools import register_tool

@register_tool
def weather(city: str) -> str:
    """查询某个城市的天气。"""
    return f"{city} 今天晴，25℃"
```

规则：docstring 是工具描述；参数加类型注解（`str`/`int`/`float`/`bool`），无默认值的必填。
函数保持同步即可——runner 会用 `asyncio.to_thread` 丢进线程池，同步 IO 可以放心写。

---

## 十二、安全注意

- 密钥走环境变量，不写进代码、不进截图；
- `fetch_webpage` 与知识库在生产中要做 **SSRF 防护**（限制访问内网地址），不要抓取不可信链接；
- 搜索结果是正则解析网页 HTML，网站改版可能失效，生产建议改用官方搜索 API；
- `calculate` 用 AST 白名单，避免模型注入并执行任意代码；
- **新增提醒**：`write_file` / `edit_file` / `run_shell` 会直接操作本机——`run_shell` 能执行
  任意 shell 命令（可删文件、可外发数据）。教学版只在本地可信环境使用；生产环境必须加
  沙箱与命令白名单（nanobot 用 bubblewrap 等隔离后端），并给危险工具加审批机制。

---

## 十三、相对 nanobot，刻意简化了什么

现在**已经有了**：全异步 asyncio、MessageBus、按会话分片、JSONL 持久化与重启恢复、
指数退避重试、多渠道抽象、配置文件 + 环境变量。**仍然刻意简化**的部分：

| 维度 | mini_agent | nanobot（生产版） |
|---|---|---|
| 渠道 | 只有 cli + echo | Telegram / Discord / Slack / 飞书 / WebUI 等 16+ |
| 命令 | /reset /help /exit 三条 | 完整命令路由（command router） |
| 长对话 | history_limit 截断 + 孤儿 tool 清洗 | 上下文压缩 + Dream 记忆固化 |
| 会话回收 | worker 常驻，同会话一个任务 | TTL 回收、弱引用锁、归档 |
| 工具 | 11 个基础工具，无审批机制 | 沙箱化执行、审批、cron、MCP、子代理等 |
| 流式 | 无（一次返回完整回复） | 流式输出（streaming） |
| 其他 | 无 hooks | 钩子、fallback 链、图片、语音、WebUI 等 |

---

## 十四、目录结构

```
mini_agent/
├── __init__.py               # 包入口：导入即注册全部工具，导出常用类
├── providers/                # 模型功能包（对应 nanobot/providers/）
│   ├── __init__.py           # 统一导出 + provider_from_env / embedder_from_env
│   ├── base.py               # 数据结构：Message / ToolCall / LLMResponse / Usage + LLMProvider 抽象
│   ├── openai_compat.py      # OpenAICompatProvider（AsyncOpenAI）+ RetryingProvider
│   └── embeddings.py         # Embedder 抽象 + OpenAI 兼容嵌入（async）
├── agent/                    # Agent 核心包（对应 nanobot/agent/）
│   ├── __init__.py           # 导出 Agent / AgentLoop / AgentRunner + tools
│   ├── runner.py             # AgentRunner：一条会话里的 ReAct 循环
│   ├── loop.py               # AgentLoop：总线 -> 按会话分片 -> 引擎 -> 渠道
│   ├── agent.py              # Agent：最简门面（单会话 + 记忆）
│   └── tools/                # 工具功能包（对应 nanobot/agent/tools/）
│       ├── __init__.py       # 导出 REGISTRY 等；导入即注册全部工具
│       ├── registry.py       # 工具注册表 + @register_tool + schema 自动生成
│       ├── builtin.py        # 7 个基础工具：时间 / 计算 / 文件读写 / 目录 / 命令
│       ├── knowledge.py      # RAG：切块 / 余弦检索 + 2 个知识库工具
│       └── web.py            # web_search / fetch_webpage
├── bus/                      # 消息总线（对应 nanobot/bus/）
│   ├── __init__.py
│   └── queue.py              # MessageBus + InboundMessage / OutboundMessage
├── session/                  # 会话仓库（对应 nanobot/session/）
│   ├── __init__.py
│   └── manager.py            # Session / SessionManager（JSONL 增量持久化）
├── channels/                 # 渠道（对应 nanobot/channels/）
│   ├── __init__.py           # Channel 导出 + build_channels 工厂
│   ├── base.py               # Channel 抽象：start / send / stop
│   ├── cli.py                # CliChannel：stdin -> 总线；回复打印终端
│   └── echo.py               # EchoChannel：TCP 行协议服务端（演示多渠道）
├── tests/                    # 离线自测（python -m mini_agent.tests.test_*）
│   ├── __init__.py
│   ├── test_offline.py       # 离线自测 1：ReAct / 工具守卫 / 重试
│   ├── test_rag_offline.py   # 离线自测 2：切块 / 嵌入 / 检索
│   └── test_pipeline_offline.py  # 离线自测 3：总线 / 分片 / 持久化 / 命令 / SDK（E2E）
├── config.py                 # MiniConfig + load_config（JSON + 环境变量）
├── sdk.py                    # MiniBot：完整门面（装配以上全部）
├── __main__.py               # 命令行入口：python -m mini_agent
└── README.md
```
