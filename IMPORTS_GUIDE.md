# mini_agent 导入与依赖说明

> 范围：`mini_agent/` 包。
> 目标：说明每个 `import` 来自哪里、实际用到了哪些函数/类、用在哪里、为什么选它，以及它的边界和代价。
> 文中 `文件:行号` 均对应当前代码。

---

## 0. 先说结论

| 项目 | 结论 |
|---|---|
| 运行时第三方依赖 | 只有 `openai`（`requirements.txt` 中为 `openai>=1.60,<4`） |
| 其余运行时依赖 | 全部来自 CPython 标准库 |
| Python 版本 | 3.11+（代码使用 `X \| None`、`asyncio.Task[...]` 等写法） |
| 明确未使用 | `requests`、`httpx`（仅作为 openai 的传递依赖存在）、`pydantic`、`numpy`、`faiss`、`click`、`rich`、`python-dotenv`、`langchain` |

选择思路可以概括为一句话：

**用一个官方 SDK 解决模型协议，其余能力尽量用标准库自己搭；依赖越少，部署、审计和替换越简单。**

这带来两个直接结果：

1. 安装只需一个第三方包，不编译原生扩展，Windows / Linux / macOS 行为一致；
2. 每个敏感能力（子进程、网络、文件、长期记忆、表达式）都集中在少数模块里，安全审计有明确入口。

---

## 1. 依赖总览

### 1.1 运行时

| 来源 | 包/模块 | 用途 | 主要位置 |
|---|---|---|---|
| 第三方 | `openai` | 调用 OpenAI 兼容的 Chat Completions 与 Embeddings | `providers/openai_compat.py`、`providers/embeddings.py` |
| 标准库 | `asyncio` | 事件循环、队列、任务、锁、信号量、超时、线程桥接、并发工具调用 | 全局；`agent/subagent.py`、`agent/tools/registry.py`、`agent/runner.py` |
| 标准库 | `logging` | 分层日志，不直接 print | 全局 |
| 标准库 | `dataclasses` | 配置、消息、会话、工具、子代理状态等值对象 | 全局 |
| 标准库 | `abc` | 渠道 / 模型 / 嵌入的抽象接口 | `channels/`、`providers/` |
| 标准库 | `typing` | 类型标注与 `TYPE_CHECKING` | 全局 |
| 标准库 | `collections.abc` | 子代理结果回调的 `Callable` / `Awaitable` 协议 | `agent/subagent.py` |
| 标准库 | `json` | 配置、工具参数、JSONL 会话、`--check` 输出 | `config.py`、`runner.py`、`session/`、`__main__.py` |
| 标准库 | `os` | 环境变量、路径、`fsync`、原子替换 | `config.py`、`session/`、`builtin.py`、`policy.py` |
| 标准库 | `pathlib` | 工作区路径解析与越界判断 | `agent/tools/policy.py` |
| 标准库 | `tempfile` | 会话 / 记忆文件原子重写的临时文件 | `session/manager.py`、`memory/store.py` |
| 标准库 | `argparse` | CLI 参数解析 | `__main__.py` |
| 标准库 | `sys` | stdin/stdout/stderr、直接运行时的 `sys.path` | `__main__.py`、`utils/observability.py` |
| 标准库 | `signal` | SIGINT/SIGTERM 优雅停机 | `__main__.py` |
| 标准库 | `contextlib` | 抑制任务取消异常、子代理可选的 `nullcontext` 记忆作用域 | `channels/cli.py`、`agent/subagent.py` |
| 标准库 | `subprocess`、`asyncio.subprocess` | `run_shell` 异步子进程与 Windows 进程组 | `agent/tools/builtin.py` |
| 标准库 | `ast`、`operator`、`math` | 安全计算器 | `agent/tools/builtin.py` |
| 标准库 | `math` | 余弦相似度、有限数校验 | `agent/tools/knowledge.py`、`config.py` |
| 标准库 | `re` | 会话键清洗、记忆检索分词、HTML 解析 | `utils/naming.py`、`memory/store.py`、`agent/tools/web.py` |
| 标准库 | `html` | HTML 实体反转义 | `agent/tools/web.py` |
| 标准库 | `urllib.request`、`urllib.parse` | HTTP 抓取、URL 解析、查询编码 | `agent/tools/web.py` |
| 标准库 | `socket` | DNS 解析，用于 SSRF 判定 | `agent/tools/web.py` |
| 标准库 | `ipaddress` | 判断公网 / 内网 / 回环 / 保留地址 | `agent/tools/web.py` |
| 标准库 | `inspect` | 从函数签名生成工具 JSON Schema、识别协程函数 | `agent/tools/registry.py` |
| 标准库 | `datetime` | `get_current_time`、提示词中的当前日期、子代理系统提示词 | `agent/tools/builtin.py`、`agent/context.py`、`agent/subagent.py` |
| 标准库 | `time` | 子代理开始/结束时间的单调时钟 | `agent/subagent.py` |
| 标准库 | `random` | 重试抖动 | `providers/openai_compat.py` |
| 标准库 | `threading` | 保护知识库索引与长期记忆缓存 | `agent/tools/knowledge.py`、`memory/store.py` |
| 标准库 | `contextvars` | 把当前 `session_key` 传给记忆工具；把渠道/发送者/会话键传给子代理工具 | `memory/scope.py`、`agent/request.py` |
| 标准库 | `uuid` | 长期记忆条目 id、子代理任务 id | `memory/store.py`、`agent/subagent.py` |

### 1.2 不是一个"包"的导入

每个文件顶部的：

```python
from __future__ import annotations
```

不是第三方包，而是把类型注解延迟为字符串求值。好处是：

- 运行时不需要解析前向引用；
- 减少循环导入风险；
- 注解只在类型检查器里生效。

代价是 `inspect.signature()` 读到的注解也是字符串，所以工具注册表准备了 `_TYPE_BY_NAME` 做还原（`agent/tools/registry.py:30-32`、`103-118`）。

---

## 2. 第三方库：`openai`

### 2.1 它解决什么问题

模型调用本质上是一次 HTTP + JSON 请求，但手写会立刻遇到一堆重复问题：

- 鉴权头、Base URL、超时；
- 请求/响应 JSON 的字段映射；
- 流式与非流式响应；
- 工具调用（`tool_calls`）的嵌套结构；
- 不同 HTTP 状态码对应的异常类型；
- 连接复用与连接池关闭；
- OpenAI 兼容端点之间的细微差异。

`openai` 官方 SDK 把这些都封装好了，并且支持所有"OpenAI 兼容"端点。因此项目只保留一个适配层 `OpenAICompatProvider`，就能接入 OpenAI、DashScope、Ollama、vLLM、各类网关。

### 2.2 实际用到的类与函数

| 符号 | 作用 | 调用位置 |
|---|---|---|
| `AsyncOpenAI(...)` | 创建异步客户端；参数含 `api_key`、`base_url`、`timeout`、`max_retries` | `providers/openai_compat.py:52-57`、`providers/embeddings.py:51-56` |
| `client.chat.completions.create(...)` | 发起一次对话请求；传 `model`、`messages`、可选 `tools` | `providers/openai_compat.py:67-73` |
| `raw.choices[0].message.content` | 取模型文本回复 | `providers/openai_compat.py:78-99` |
| `message.tool_calls` | 取工具调用列表（`id` / `function.name` / `function.arguments`） | `providers/openai_compat.py:80-87` |
| `choice.finish_reason` | 判断正常结束、工具调用、过滤、报错 | `providers/openai_compat.py:98-103` |
| `raw.usage` | 取 token 用量；缺失时保持 `None` | `providers/openai_compat.py:91-97` |
| `client.embeddings.create(...)` | 批量文本转向量 | `providers/embeddings.py:57-60` |
| `raw.data[].embedding`、`raw.data[].index` | 取向量并按 `index` 排序，保证与输入一一对应 | `providers/embeddings.py:61-68` |
| `client.close()` | 关闭底层 HTTP 连接池 | `providers/openai_compat.py:105-106`、`providers/embeddings.py:57-60` |
| `APITimeoutError` | 请求超时，可重试 | `providers/openai_compat.py:21`、`120` |
| `APIConnectionError` | 连接失败，可重试 | 同上 |
| `RateLimitError` | 限流（429），可重试 | 同上 |
| `InternalServerError` | 5xx，可重试 | 同上 |
| `APIStatusError` | 兜底读取 `status_code`，按 408/409/429/5xx 分类 | `providers/openai_compat.py:122-123` |

### 2.3 两处客户端配置为什么不同

**对话客户端**（`openai_compat.py:52-57`）：

```python
self.client = AsyncOpenAI(
    api_key=api_key or "EMPTY",
    base_url=api_base,
    timeout=timeout,
    max_retries=0,
)
```

- `max_retries=0`：项目外层已经有 `RetryingProvider`，必须关掉 SDK 内建重试，否则会叠成 `3 × 3` 次请求；
- `"EMPTY"`：本地模型（Ollama/vLLM）通常不校验密钥，但 SDK 要求非空；
- `timeout` 来自配置，默认 60 秒。

**嵌入客户端**（`embeddings.py:51-60`）：

```python
client = AsyncOpenAI(
    api_key=self._api_key,
    base_url=self._api_base,
    timeout=self._timeout,
    max_retries=2,
)
try:
    raw = await client.embeddings.create(model=self.model, input=texts)
finally:
    await client.close()
```

- `max_retries=2`：嵌入调用没有外层 `RetryingProvider`，保留 SDK 内建重试；
- **每次调用创建、用完即关**：RAG 工具在 `asyncio.to_thread` 的工作线程里用 `asyncio.run()` 调 `embed()`，不同线程有不同事件循环，httpx 连接池不能跨事件循环共享。若复用同一个客户端，会出现"连接绑定到已关闭事件循环"的隐性故障。

### 2.4 为什么选择 `openai`

1. **官方维护，协议最跟得上**：`chat.completions`、`tools`、`tool_calls`、`usage` 的字段变化由 SDK 跟进，项目不用追着网关改 JSON。
2. **一个 SDK 覆盖两件事**：对话与嵌入共用同一套 SDK、鉴权方式、Base URL 与超时配置，省掉第二套 HTTP 代码；不过嵌入客户端按调用创建，二者并不共享连接池。
3. **异步原生**：`AsyncOpenAI` 基于 `await`，不会像同步 `requests` 那样卡住事件循环。
4. **异常类型可分类**：`RateLimitError`、`APITimeoutError`、`APIStatusError.status_code` 让重试策略可以"精确重试"，而不是靠字符串猜。
5. **兼容生态最大**：OpenAI、DashScope、Ollama、vLLM、LiteLLM 等都能用同一个适配器，换供应商通常只改 `api_base` 和 `model`。
6. **重试策略可关闭**：`max_retries=0` 让"重试"这件事只保留在一层，职责清晰。

### 2.5 代价与边界

- SDK 内部仍然依赖 `httpx`，所以它不是"零依赖"，只是"零直接依赖"；
- SDK 只解决 HTTP/JSON，不解决：上下文截断、工具安全、会话持久化、并发编排；
- 不同兼容端点对 `usage`、`finish_reason`、`tool_calls` 的支持并不一致，所以代码里做了空值容忍与异常状态判定（`LLMResponse.should_execute_tools`）；
- **不要把 `openai` 类型泄漏到上层**：`AgentRunner`、`AgentLoop`、`MiniBot` 只认识 `LLMProvider` 和 `LLMResponse`。这样换掉 SDK 时只需改 `providers/`。

### 2.6 被拒绝的替代方案

| 方案 | 不选的原因 |
|---|---|
| 直接用 `requests` | 同步阻塞事件循环；要自己处理 JSON、异常、连接池 |
| 直接用 `httpx` | 能异步，但要手写完整协议适配与异常分类，等于重造 SDK |
| 每家厂商一个 SDK | 依赖数量、版本冲突、许可证、审计面都会膨胀 |
| `litellm` 等聚合层 | 多一层抽象和依赖；本项目只需要 OpenAI 兼容协议这一条主路径 |
| 本地推理库（`transformers` 等） | 引入原生扩展与模型权重，超出"轻量运行时"的范围 |

---

## 3. 标准库逐项说明

### 3.1 异步与并发

#### `asyncio`

**用到的 API**

| API | 作用 | 典型位置 |
|---|---|---|
| `asyncio.run(...)` | CLI 入口启动事件循环；同步工具里调用异步嵌入 | `__main__.py:114`、`123`；`knowledge.py:142`、`152` |
| `asyncio.create_task(...)` | 启动后台任务（分发器、会话 worker、CLI 读循环、后台子代理） | `loop.py:134-139`、`150-153`；`cli.py:33`；`subagent.py:205` |
| `asyncio.Queue` | 入站/出站队列、每个会话的分片队列 | `bus/queue.py:52-53`；`loop.py:148` |
| `asyncio.Lock` | 会话锁、落盘锁、分片表锁 | `loop.py:53-54`、`282-287`；`manager.py:230-235` |
| `asyncio.Semaphore` | 限制同时运行的子代理数量；超出的在信号量上排队 | `agent/subagent.py:103`、`253` |
| `asyncio.Event` | CLI 退出信号、单轮回复完成信号 | `cli.py:26-29`；`__main__.py:180` |
| `asyncio.wait_for(...)` | 给"取消息"和"排空队列"加超时；给单个子代理加执行超时 | `loop.py:163-166`、`235-240`、`250-253`；`subagent.py:258-260` |
| `asyncio.TimeoutError` | 捕获上述超时 | `loop.py:166`、`239`、`253`；`subagent.py:265` |
| `asyncio.gather(...)` | 并行执行同一轮里的多个 `concurrency_safe` 工具（典型是多个 `delegate`）；取消后等待任务收尾 | `runner.py:100-116`；`loop.py:227`、`245`、`256`；`__main__.py:195` |
| `asyncio.to_thread(...)` | 把同步阻塞函数丢进线程池 | `registry.py:93`；`cli.py:40`；`manager.py:166`、`175` |
| `asyncio.sleep(...)` | 重试退避、出站队列轮询 | `openai_compat.py:189`、`203`；`loop.py:265` |
| `asyncio.CancelledError` | 区分"被取消"与"业务异常" | `loop.py:175`；`cli.py:67`；`subagent.py:271` |

**为什么选它**

- 标准库自带，不需要 `trio` / `anyio`；
- `Queue + Task + Lock + Event` 足够覆盖本项目的并发模型；
- `to_thread` 是同步工具与异步核心之间的标准桥；
- 与 `openai.AsyncOpenAI` 天然兼容。

**边界**

- 单事件循环内，CPU 密集型代码仍会阻塞；项目通过 `to_thread` 把工具执行移出主循环；
- `asyncio.Lock` 不能跨事件循环共享，所以跨线程共享的向量库改用 `threading.RLock`；
- 会话文件锁是"单进程内"的，多进程同时写同一会话不在保证范围内。

#### `threading`

**用到的 API**

- `threading.RLock`：`agent/tools/knowledge.py`，保护 `chunks` / `vectors` / `sources` 三个列表；
- `threading.RLock`：`memory/store.py`，保护长期记忆的内存缓存与写入。

**为什么不用 `asyncio.Lock`**

`KnowledgeBase.ingest()` 与 `search()` 是协程，但工具函数是同步的，会在 `asyncio.to_thread` 的工作线程里通过 `asyncio.run()` 执行。不同线程拥有不同的事件循环，`asyncio.Lock` 绑定创建它的事件循环，跨线程使用会出错；`RLock` 是线程级锁，且可重入，正好覆盖这个场景。长期记忆仓库同理：模型工具与 `/memory` 命令都可能在线程池里读写它。

**边界**

- 锁只保护内存结构，不保护嵌入网络请求；
- 进程级知识库与记忆缓存不跨进程共享。

#### `contextvars`

**用到的 API**

- `ContextVar`：`memory/scope.py` 保存当前对话的 `session_key`；`agent/request.py` 保存当前请求的 `RequestContext`（渠道 / 发送者 / `session_key` / 是否可后台）；
- `contextmanager` + token 重置：`memory_scope()` 与 `request_scope()` 在进入 / 退出时设置与恢复作用域。

**解决什么问题**

记忆工具（`memory_save` 等）只能操作"当前会话"，但工具签名由模型参数决定，不能也不应该让模型传入 `session_key`——否则模型可以读写任意用户的记忆。`AgentLoop` 在调用 `runner.run()` 前用 `with memory_scope(key)` 设置上下文，工具内部通过 `current_memory_scope()` 读取。

子代理工具（`delegate` / `spawn`）需要父会话的渠道与发送者才能把后台结果投递回去；这些信息同样由 `AgentLoop` 通过 `request_scope(RequestContext(...))` 绑定。模型只能传任务描述与标签，**无法指定 session_key、渠道或发送者**。

**为什么 `asyncio.to_thread` 还能读到**

`asyncio.to_thread` 会复制当前 `contextvars.Context` 到工作线程，因此 `runner` 把同步工具丢进线程池后，工具仍能看到正确的会话键。子代理在自己的 `asyncio.Task` 中运行，任务会复制创建时的上下文；`SubagentManager` 还会额外用 `memory_scope(session_key)` 把长期记忆绑定到父会话。

**边界**

- `ContextVar` 是协程 / 线程上下文，不是安全边界；它防的是"模型误传参数"，不是恶意代码；
- 没有设置作用域时（例如直接用 `Agent` 或手工调用工具），记忆工具会拒绝操作，`delegate` / `spawn` 会返回"无法确定当前会话"；
- 子代理结果通过总线注入父会话时，使用的是服务端保存的 `session_key`，而不是子代理或模型生成的字符串。

### 3.2 数据建模

#### `dataclasses`

**用到的 API**

| API | 作用 | 位置 |
|---|---|---|
| `@dataclass` | 自动生成 `__init__` / `__repr__` / `__eq__` | `providers/base.py`、`bus/queue.py`、`config.py`、`session/manager.py`、`agent/tools/registry.py`、`agent/request.py`、`agent/subagent.py` |
| `field(default_factory=...)` | 避免可变默认值共享（如 `list`、单调时钟起点） | `providers/base.py:79`、`session/manager.py:31-32`、`config.py:67`、`subagent.py:54` |
| `frozen=True, slots=True` | 不可变、低开销的请求上下文与结果对象 | `agent/request.py`、`agent/subagent.py:59` |
| `fields(MiniConfig)` | 遍历全部配置字段，生成脱敏快照 | `config.py:161-166` |
| `replace(config, **overrides)` | 基于现有配置生成覆盖后的新对象 | `__main__.py:20`、`92`、`154` |

**为什么选它**

- 值对象字段固定，`dataclass` 的样板代码最少；
- 不引入运行时校验框架；
- `replace` 提供"不可变风格"的覆盖，避免 CLI 参数直接改原配置；
- `fields()` 让 `describe()` 不需要手写全部字段名。

**为什么不用 `pydantic`**

本项目的配置校验规则很具体（范围、互斥、中文错误信息、CLI 退出码 2），手写 `validate()` 更直接、可控，也不会把 Pydantic 的版本/行为差异带进运行时。`ConfigError` 还能精确映射到退出码。

#### `abc`

**用到的 API**

- `ABC`、`abstractmethod`：`Channel`（`channels/base.py:12`）、`LLMProvider`（`providers/base.py:110`）、`Embedder`（`providers/embeddings.py:19`）。

**为什么选它**

- 渠道必须实现 `start/send/stop`，抽象基类能在实例化时立即报错；
- 模型与嵌入的接口固定，便于替换实现；
- 比 `typing.Protocol` 多一层运行时约束，适合"明确的可插拔点"。

#### `typing`

**用到的 API**

- `Any`：动态 JSON / 配置字段；
- `Callable`：工具函数的类型；
- `TYPE_CHECKING`：只在类型检查时导入 `Channel`，避免运行时循环依赖（`loop.py:21`、`28-29`）。

**说明**

配合 `from __future__ import annotations`，所有注解默认是字符串，运行时开销接近零。工具注册表因此需要把字符串注解映射回真实类型（`registry.py:30-32`）。

### 3.3 CLI 与进程生命周期

#### `argparse`

**用到的 API**

- `ArgumentParser`、`add_argument`、`parse_args`；
- `RawDescriptionHelpFormatter` 保留 epilog 换行；
- `action="version"`、`choices=`、`type=str.upper`。

位置：`__main__.py:36-78`。

**为什么选它**

- CLI 只有 7 个选项，没有子命令树；
- 标准库自动生成 `--help` / `--version`；
- 不引入 `click` / `typer` / `rich`，保持零依赖；
- `choices` 能在解析阶段就拒绝非法日志级别。

#### `sys`

**用到的 API**

| API | 作用 | 位置 |
|---|---|---|
| `sys.path.insert(0, ...)` | 直接运行 `python mini_agent/__main__.py` 时补齐包路径 | `__main__.py:23-24` |
| `sys.stdin.buffer.read()` | 管道模式读取原始字节，处理 UTF-16 BOM | `__main__.py:219-225` |
| `sys.stderr` | 配置错误、运行失败、日志输出 | `__main__.py:94`、`102`、`230`；`utils/observability.py:34` |
| `SystemExit(main())` | 把返回值转成进程退出码 | `__main__.py:234` |

**为什么用二进制 stdin**

Windows PowerShell 管道可能输出 UTF-16LE（带 BOM）。直接 `sys.stdin.read()` 会依赖控制台编码；读取 `buffer` 后显式识别 BOM，可以同时支持 UTF-8 与 UTF-16。

#### `os`

**用到的 API**

| API | 作用 | 位置 |
|---|---|---|
| `os.environ.get(...)` | 读取配置与模型端点 | `config.py:193-269`、`openai_compat.py:230-232`、`embeddings.py:87-90` |
| `os.path.exists/join/dirname` | 会话文件与目录判断 | `session/manager.py:129`、`213`、`227` |
| `os.makedirs(..., exist_ok=True)` | 创建会话目录、写入父目录 | `session/manager.py:160`、`214`；`builtin.py:117` |
| `os.listdir(...)` | 列目录工具、列出会话文件 | `builtin.py:101`；`session/manager.py:241` |
| `os.remove/unlink(...)` | 删除会话、清理临时文件 | `session/manager.py:195`、`225` |
| `os.replace(temp, path)` | 原子替换会话文件 | `session/manager.py:222` |
| `os.fsync(fd)` | 刷盘，避免"已确认写入但崩溃丢失" | `session/manager.py:209`、`221` |
| `os.fdopen(fd, ...)` | 把 `mkstemp` 返回的 fd 包成文件对象 | `session/manager.py:217` |
| `os.SEEK_END` | 读取文件最后一个字节，判断是否缺换行 | `session/manager.py:202` |

**为什么不用纯 `pathlib`**

`pathlib` 负责路径表达更清晰，但 `fsync`、`fdopen`、`SEEK_END` 这类底层操作仍需 `os`。项目采用"`pathlib` 管路径语义，`os` 管持久化细节"的分工。

#### `signal`

**用到的 API**

- `signal.SIGINT`、`signal.SIGTERM`；
- `loop.add_signal_handler(sig, stop.set)`；
- 捕获 `NotImplementedError / RuntimeError / ValueError` 回退到 `KeyboardInterrupt`。

位置：`__main__.py:200-207`。

**为什么选它**

优雅停机需要在收到 Ctrl+C / SIGTERM 时先停止接新消息、排空队列、关闭渠道与客户端，而不是立即杀进程。Windows 的事件循环通常不支持 `add_signal_handler`，所以保留回退路径。

#### `contextlib`

**用到的 API**

- `contextlib.suppress(asyncio.CancelledError)`：取消 CLI 读任务后等待其结束，吞掉预期的取消异常（`channels/cli.py:67`）。

**为什么选它**

比 `try/except CancelledError: pass` 更清楚地表达"这里是预期取消"。

### 3.4 子进程：`asyncio` + `subprocess`（重点）

#### 用到的 API

| API | 作用 |
|---|---|
| `asyncio.create_subprocess_shell(...)` | 在事件循环里启动 shell 命令，不占用线程池 |
| `asyncio.subprocess.PIPE` | 捕获 stdout/stderr |
| `asyncio.wait_for(...)` | 给整次命令设置超时；超时或取消时进入终止分支 |
| `subprocess.CREATE_NEW_PROCESS_GROUP` | Windows：让命令进入独立进程组，便于 `taskkill /T` 终止整棵树 |
| `locale.getencoding()` | 按本机控制台编码解码输出，坏字节用 `errors="replace"` |

位置：`agent/tools/builtin.py` 的 `run_shell` / `_terminate_tree` / `_read_limited`。

#### 代码路径

```python
if not shell_enabled():
    raise PermissionError("run_shell 未启用；...")
options = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if os.name == "nt"
    else {"start_new_session": True}
)
proc = await asyncio.create_subprocess_shell(
    command,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=str(workspace_root()),
    **options,
)
try:
    (stdout, cut), (stderr, _) = await asyncio.wait_for(
        asyncio.gather(
            _read_limited(proc.stdout, _MAX_SHELL_PIPE_BYTES),
            _read_limited(proc.stderr, _MAX_SHELL_PIPE_BYTES),
        ),
        timeout=timeout,
    )
except asyncio.TimeoutError:
    await _terminate_tree(proc)
    return f"命令执行超时（超过 {timeout:g} 秒，已终止进程树）"
```

#### 每个参数为什么这样选

| 参数 | 原因 |
|---|---|
| `create_subprocess_shell` | 工具语义就是"执行一条 shell 命令"，用户会自然使用管道、重定向、环境变量；同时不占用同步工具的工作线程 |
| `cwd=workspace_root()` | 命令的工作目录固定为工作区，避免相对路径写到意外位置 |
| `PIPE` + `_read_limited` | 捕获输出；超过保留上限后继续排空管道，避免子进程因管道写满而卡死 |
| `wait_for(..., timeout)` | 防止命令挂死；上限 300 秒，下限 1 秒 |
| `start_new_session` / `CREATE_NEW_PROCESS_GROUP` | 让命令及其子进程属于同一进程组，超时后才能整棵树终止 |
| `locale.getencoding()` + `errors="replace"` | 旧式 Windows 控制台可能输出 GBK/混合编码，替换非法字节比抛 `UnicodeDecodeError` 更稳 |

#### 超时为什么必须终止进程树

`subprocess.run(timeout=...)` 只停止等待，命令与它派生的子进程可能继续运行并产生副作用。现在的实现：

- POSIX：`os.killpg(os.getpgid(pid), SIGKILL)`；
- Windows：`taskkill /F /T /PID <pid>`；
- 两侧兜底 `proc.kill()`，并最多等 5 秒回收；
- 回合被取消（`CancelledError`）时同样终止进程树。

#### 为什么选异步子进程而不是其他方式

| 方案 | 问题 |
|---|---|
| `os.system` | 不能捕获输出、不能设置超时、退出码语义弱 |
| `os.popen` | 只能单向读，错误处理与超时都弱 |
| `subprocess.run(timeout=...)` | 超时只停止等待，不能终止进程树，长命令可能继续产生副作用 |
| `shlex.split + shell=False` | 不支持管道/重定向/变量展开，和"shell 工具"的承诺不符 |
| 自带 PTY / 终端模拟 | 复杂度远超需求，且跨平台差异大 |

#### 安全边界（必须明确）

`create_subprocess_shell` **不是沙箱**。它能执行当前用户权限下的任意命令，注入风险是工具语义本身的一部分。项目通过以下方式降低风险：

1. **默认关闭**：`allow_shell=false`，且 `run_shell` 不出现在工具 schema 里（`registry.py` 的 `available_tools`）；
2. **显式开启**：只有配置 `allow_shell=true` 才暴露；
3. **命令长度上限**：4000 字符；
4. **超时钳制**：1–300 秒，超时终止整个进程树；
5. **输出截断**：4000 字符，超出部分继续排空但不保留；
6. **固定工作目录**：始终在 `workspace_root()` 下执行；
7. **不阻塞事件循环**：工具写成 `async def`，由 `Tool.arun_with_status` 直接 await（异步工具不受同步工具超时包装）。

生产部署若需要强隔离，应把 `run_shell` 放到容器 / 虚拟机 / 受限用户中，而不是依赖本项目做权限控制。

### 3.5 文本与解析

#### `re`（重点）

`re` 在项目里有三处职责：**会话键清洗**、**长期记忆关键词检索**与**网页文本提取**。

##### 用途 1：会话键 -> 安全文件名

```python
_SAFE_KEY_RE = re.compile(r"[^0-9A-Za-z]")

def safe_key(key: str) -> str:
    prefix = _SAFE_KEY_RE.sub("_", key).strip("_")[:48] or "key"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"
```

位置：`utils/naming.py`。会话仓库（`session/manager.py`）、长期记忆仓库（`memory/store.py`）与任务清单（`session/todo.py`）共用它，保证 `cli:local` 在三处都映射到同一个文件名。

**为什么需要**

会话键来自渠道与发送者（如 `cli:local`、`telegram:123456`）。直接拼进文件路径会遇到：

- `/`、`\` 造成目录穿越；
- `:`、`*`、`?` 在 Windows 上非法；
- 空字节、控制字符导致文件系统错误。

把非字母数字替换为 `_` 可以得到文件名，但 `a:b` 与 `a_b` 会撞到同一个文件，造成跨会话串数据。现在前缀后追加原始键的 SHA-1 短哈希，不同 key 永不共享文件。

**为什么用 `re` 而不是手写循环**

一行正则即可表达"白名单替换"，且预编译后性能稳定；手写 `str.translate` 也可行，但可读性略差。

##### 用途 2：长期记忆的轻量检索

```python
_EN_TOKEN_RE = re.compile(r"[0-9a-z_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
```

位置：`memory/store.py:31-33`。

英文按词切分（长度 ≥ 2），中文按字二元组切分（`"偏好"` → `"偏好"`，`"用户偏好"` → `"用户"`、`"户偏"`、`"偏好"`）。检索时先看完整查询是否作为子串命中，再累加各词元出现次数，最后按分数与新旧排序。

**为什么不用嵌入模型**

长期记忆是辅助能力，不应因为嵌入服务不可用而失效；条目数量有上限（默认 200），关键词扫描足够。真正的大规模语义检索应替换为向量库，而不是把 RAG 的依赖强加到记忆层。

##### 用途 3：HTML 去标签与必应结果解析

```python
def _strip_tags(markup: str) -> str:
    text = re.sub(r"<[^>]+>", " ", markup)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()
```

位置：`agent/tools/web.py:28-31`。

```python
page = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", page)
```

位置：`web.py:126`。先移除脚本/样式块，再走 `_strip_tags`。

```python
blocks = re.findall(r'<li class="b_algo".*?</li>', page, flags=re.DOTALL)
link_match = re.search(
    r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    block,
    flags=re.DOTALL,
)
snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, flags=re.DOTALL)
```

位置：`web.py:103-116`。

**为什么用正则**

- 只处理两个固定结构（必应结果列表、通用标签剥离）；
- 引入 `BeautifulSoup` / `lxml` 会为两个模式增加依赖，并带来解析器版本与安全问题；
- `re.DOTALL` 让 `.` 跨行匹配，`(?is)` 同时开启忽略大小写与跨行；
- 失败时不崩溃，返回"未解析到搜索结果（可能被反爬拦截）"，让模型知道需要换策略。

**边界与风险**

- 正则解析 HTML 本质脆弱：必应改版、反爬页面、动态渲染都会让结果为空；
- 它不是通用 HTML 解析器，不处理嵌套、实体、命名空间等复杂情况；
- 如果未来要稳定抓取多个站点，应显式引入 HTML 解析库，而不是继续堆正则；
- **`re` 不参与 URL 安全判定**：SSRF 校验使用 `urllib.parse` + `socket` + `ipaddress`，避免"用正则判断 URL 是否安全"这类不可靠做法。

#### `html`

**用到的 API**

- `html.unescape(text)`：`web.py:30`。

**为什么选它**

去掉标签后，文本里仍可能有 `&amp;`、`&quot;`、`&#39;` 等实体。标准库一行解决，不需要额外依赖。

#### `ast` + `operator` + `math`

**用到的 API**

| API | 作用 |
|---|---|
| `ast.parse(expression, mode="eval")` | 把表达式解析成语法树，而不是直接执行 |
| `ast.Constant` / `ast.BinOp` / `ast.UnaryOp` | 只识别数字常量、二元运算、一元运算 |
| `operator.add/sub/mul/truediv/floordiv/mod/pow` | 白名单运算函数 |
| `operator.pos/neg` | 一元正负号 |
| `math.isfinite` | 拒绝 `inf` / `nan` |

位置：`agent/tools/builtin.py:35-81`。

**为什么不用 `eval`**

`eval("__import__('os').system('...')")` 可以直接执行任意代码。AST 白名单只允许：

- 数字常量；
- `+ - * / // % **`；
- 一元 `+/-`；
- 括号（语法树结构）。

属性访问、函数调用、变量名、列表/字典/推导式全部被拒绝。再叠加：

- 表达式长度 ≤ 200；
- 指数绝对值 ≤ 100；
- 结果绝对值 ≤ 1e12；
- 结果必须是有限数。

这样即使模型被诱导生成恶意表达式，也无法触达文件、网络或进程。

**为什么用 `operator`**

把 AST 节点映射到小的纯函数，避免在 `_eval_ast` 里写一长串 `if/elif`；同时天然不涉及 `eval`。

#### `math`

除了计算器的 `isfinite`，还用于：

- `cosine_similarity()` 的 `math.sqrt`（`knowledge.py:36-43`）；
- 配置校验中的 `math.isfinite`（`config.py:154-157`）。

**为什么不用 `numpy`**

RAG 的向量是 `list[float]`，余弦相似度是 O(n) 的点积与模长；对单机、中等文档量足够。`numpy` 会引入原生扩展、平台 wheel 与额外内存开销，不符合"轻量、零编译"的定位。若未来数据量增大，应把 `KnowledgeBase` 替换为专门的向量库，而不是在工具层零散引入。

### 3.6 网络与 SSRF 防护

#### `urllib.request`

**用到的 API**

| API | 作用 | 位置 |
|---|---|---|
| `urllib.request.Request(url, headers=...)` | 构造带 User-Agent 的请求 | `web.py:81-84` |
| `urllib.request.build_opener(_SafeRedirectHandler())` | 构造带安全重定向检查的 opener | `web.py:85` |
| `opener.open(request, timeout=...)` | 发起 GET | `web.py:86` |
| `response.read(limit + 1)` | 限制下载体积，超限截断 | `web.py:87-89` |
| `response.headers.get_content_charset()` | 读取响应字符集 | `web.py:90` |
| `urllib.request.HTTPRedirectHandler` | 继承并重写 `redirect_request` | `web.py:65-76` |

**为什么选它**

- 标准库自带，避免为两个工具引入 `requests` / `httpx`；
- `build_opener` 允许插入自定义重定向处理器，这是做 SSRF 防护的关键；
- 同步阻塞的 `open()` 会在工具线程里执行，不影响事件循环。

**为什么不用 `requests`**

`requests` 是同步的，且默认会自动跟随重定向；要安全地重新校验每一跳，仍需自定义 `Session` / `resolve_redirects`，并没有比 `urllib` 简单，反而多一个依赖。

#### `urllib.parse`

**用到的 API**

- `urlparse(url)`：取 `scheme`、`hostname`、`port`；
- `quote(query)`：把搜索词编码进 URL。

位置：`web.py:36-45`、`100`。

**为什么不用字符串拼接/正则**

`urlparse` 能正确处理用户名、端口、IPv6 方括号、查询串等边界；用正则判断 hostname 很容易被 `http://127.0.0.1@evil.com` 这类形式绕过。

#### `socket`

**用到的 API**

- `socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)`：把域名解析成所有 IP；
- `socket.gaierror`：解析失败时转成中文错误。

位置：`web.py:46-49`。

**为什么需要 DNS 解析**

只检查 URL 里的字符串不够：`http://internal.example.com` 可能解析到 `10.0.0.5`。先解析再逐个检查 IP，才能覆盖"域名指向内网"的情况。

#### `ipaddress`

**用到的 API**

- `ipaddress.ip_address(raw_ip)`：解析 IPv4/IPv6；
- `IPv6Address.ipv4_mapped`：把 `::ffff:127.0.0.1` 还原成 IPv4 再判断；
- `address.is_global`：拒绝回环、私网、链路本地、保留地址。

位置：`web.py:50-62`。

**为什么用 `is_global` 而不是手写网段**

手写 `10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`、`127.0.0.0/8`、`169.254.0.0/16`、`::1`、`fc00::/7` 很容易漏；`is_global` 由标准库维护完整列表，并处理 IPv4-mapped IPv6。

**重定向防护**

`_SafeRedirectHandler.redirect_request()` 在每次跳转前再次调用 `validate_url(newurl)`（`web.py:65-76`），避免"首次 URL 合法、302 跳到内网"。

**已知边界**

- DNS 解析与实际连接之间存在 TOCTOU 窗口，理论上可被 DNS rebinding 利用；本项目是轻量防护，不是完整 egress 代理；
- `allow_private_urls=true` 会完全跳过 IP 检查，只应在可信环境使用；
- 不限制端口、不限制响应类型，只有体积上限 2 MB 与文本截断 4000 字符。

### 3.7 持久化与文件

#### `json`

**用到的 API**

| API | 作用 | 位置 |
|---|---|---|
| `json.load(handle)` | 读取配置文件 | `config.py:209-210` |
| `json.loads(line)` | 解析工具参数、会话 / 记忆 JSONL 行 | `runner.py`；`session/manager.py`；`memory/store.py` |
| `json.dumps(obj, ensure_ascii=False)` | 写会话 / 记忆 JSONL、`--check` 输出 | `session/manager.py`；`memory/store.py`；`__main__.py` |

**为什么选 JSON/JSONL**

- 人类可读，排障时可以直接看文件；
- 一行一条消息，天然支持追加与逐行恢复；
- 不需要数据库或二进制序列化；
- `ensure_ascii=False` 保留中文，便于审计。

**边界**

- JSONL 不是事务日志：进程在 `write` 与 `fsync` 之间崩溃，最后一行可能不完整；读取时会跳过损坏行，并在下次保存时整文件重写；
- 没有加密、没有访问控制，敏感对话不应直接落盘到共享目录。

#### `pathlib`

**用到的 API**

| API | 作用 | 位置 |
|---|---|---|
| `Path(path).expanduser()` | 展开 `~` | `policy.py:44` |
| `candidate.is_absolute()` | 区分相对/绝对路径 | `policy.py:45` |
| `candidate.resolve()` | 展开符号链接与 `..` | `policy.py:47` |
| `resolved.relative_to(_workspace)` | 判断是否仍在工作区内 | `policy.py:48-53` |
| `Path.is_dir()` / `exists()` / `read_text()` | 工具与知识库的文件操作 | `builtin.py`、`knowledge.py` |

**为什么选它**

- `resolve() + relative_to()` 是防路径穿越的核心组合：先消除 `..` 与符号链接，再判断前缀；
- 比字符串拼接更安全，跨平台处理 `/` 与 `\`；
- 可读性优于 `os.path`。

**边界**

- 解析后到实际打开之间仍有 TOCTOU 窗口（符号链接可被替换）；
- Windows 上普通用户默认没有创建符号链接的权限，受保护目录内的链接会被拒绝。

#### `tempfile`

**用到的 API**

- `tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=directory)`：会话文件原子重写；
- `tempfile.mkstemp(prefix=".memory-", suffix=".tmp", dir=directory)`：长期记忆文件原子重写。

位置：`session/manager.py`、`memory/store.py`。

**为什么必须放在同一目录**

`os.replace(temp, path)` 的原子性依赖"同一文件系统"。放在系统临时目录可能跨盘，`replace` 会退化为非原子操作。`mkstemp` 还避免可预测文件名带来的竞态与覆盖风险。

#### 持久化组合：`fsync` + `os.replace`

`_write_lines()` 的顺序是：

1. 写临时文件；
2. `flush()`；
3. `os.fsync()`；
4. `os.replace()` 原子替换；
5. `finally` 清理残留临时文件。

这样读者要么看到旧文件，要么看到完整新文件，不会看到"写了一半"的会话。追加路径同样在 `write` 后 `flush + fsync`（`manager.py:204-209`）。

长期记忆不做增量追加：条目可能被删除、去重或淘汰，整份文件重写更简单且不会留下"已删除但仍在文件里"的记录。写入同样走 `mkstemp + fsync + os.replace`，目录按 0700、文件在 POSIX 上按 0600 创建；Windows 依赖用户目录的 ACL。

### 3.8 工具元数据与时间

#### `inspect`

**用到的 API**

| API | 作用 | 位置 |
|---|---|---|
| `inspect.getdoc(fn)` | 没有显式 description 时取 docstring | `registry.py:88` |
| `inspect.signature(func)` | 读取参数名、注解、默认值 | `registry.py:105` |
| `inspect.Parameter.empty` | 判断"无注解"与"无默认值" | `registry.py:110`、`117` |

**为什么选它**

工具 schema 的唯一事实来源是函数本身：

```python
@register_tool
def write_file(path: str, content: str) -> str:
    """把 content 写入工作区内的文件……"""
```

注册时自动得到参数名、类型、必填项与描述，避免"函数改了、schema 忘了改"。

**边界**

- 只支持 `str/int/float/bool` 四种基础类型；
- 注解字符串无法识别时退化为 `string`；
- 不支持嵌套对象、数组项 schema、枚举。

#### `datetime`

**用到的 API**

- `datetime.now().strftime("%Y-%m-%d %H:%M:%S")`：`get_current_time` 工具（`builtin.py`）；
- `datetime.now().strftime("%Y-%m-%d")`：提示词里的"当前日期"（`agent/context.py`）；
- `datetime.now().astimezone().isoformat(timespec="seconds")`：长期记忆条目的创建时间（`memory/store.py`）。

**为什么选它**

本地时间、固定格式、零依赖；不需要时区库。记忆条目带本地时区偏移，便于跨机器排查；提示词只写日期，避免模型把"当前时间"当成精确时钟。若未来要跨时区，应显式引入 `zoneinfo` 并让格式包含时区。

#### `random`

**用到的 API**

- `random.uniform(0.8, 1.2)`：重试退避抖动，`openai_compat.py:209-211`。

**为什么加抖动**

大量客户端在同一时刻失败时，固定退避会造成"重试风暴"。±20% 抖动打散重试时间。它不是安全随机数，也不用于生成令牌。

### 3.9 日志

#### `logging`

**用到的 API**

| API | 作用 | 位置 |
|---|---|---|
| `logging.getLogger(__name__)` | 每个模块拿命名空间 logger | 全局 |
| `setup_logging(level, force=...)` | 配置 `mini_agent` 命名空间 | `utils/observability.py:19-42` |
| `StreamHandler(sys.stderr)` | 输出到 stderr | `utils/observability.py:40-42` |
| `Formatter(...)` | 时间 + 级别 + logger 名 + 消息 | `utils/observability.py:41`（格式见 `13-15`） |
| `logger.setLevel` / `propagate = False` | 控制级别，避免重复输出 | `utils/observability.py:30-31` |
| `logger.handlers` / `removeHandler` | `force=True` 时替换 handler | `utils/observability.py:33-34` |

**为什么选它**

- 库代码只记日志，不决定输出去向；CLI 与宿主程序各自配置；
- 标准库，无额外依赖；
- 命名空间统一为 `mini_agent`，宿主可整体接管；
- stderr 与 stdout 分离，管道模式下不会污染模型回答。

**边界**

- 没有文件轮转、JSON 日志、采样；这些应由部署侧或宿主程序配置；
- `setup_logging` 只管理 `mini_agent` 命名空间，不会动 root logger。

---

## 4. 为什么选择这些包

### 4.1 八条选型原则

1. **标准库优先**
   能用标准库表达的，不引入第三方。减少安装体积、许可证审查、CVE 跟踪与版本冲突。

2. **一个协议，一个 SDK**
   模型侧只保留 OpenAI 兼容协议。多厂商差异通过 `api_base` / `model` 解决，而不是引入多套 SDK。

3. **异步优先**
   核心是 `asyncio`；阻塞操作必须离开事件循环线程（线程池或守护线程）。不引入同步 HTTP 客户端。

4. **安全能力集中**
   - 子进程只在 `builtin.py`；
   - 网络与 SSRF 只在 `web.py`；
   - 文件路径只在 `policy.py` 与 `session/`；
   - 长期记忆的落盘、容量与去除控制字符只在 `memory/`；
   - 表达式只在 `builtin.py` 的 AST 白名单。

   审计时按模块定位，不需要全仓库搜索危险 API。

5. **可替换**
   `LLMProvider`、`Embedder`、`Channel` 都是抽象接口；`openai` 只出现在实现层。替换模型或渠道不影响核心。

6. **可替换**
   依赖注入 + 抽象接口：`LLMProvider`、`Embedder`、`Channel` 都可以在构造时替换，核心逻辑不需要认识具体实现。

7. **跨平台**
   不编译原生扩展；`os.replace` / `os.fsync` / `pathlib` 在 Windows 与 POSIX 上都有明确语义；`subprocess` 的 shell 语义按平台原生行为执行。

8. **可解释**
   JSON/JSONL、正则、`urllib`、手写校验都可直接阅读与调试，没有框架魔法。

### 4.2 替代方案对比

| 需求 | 本项目选择 | 常见替代 | 不选替代的原因 |
|---|---|---|---|
| 模型调用 | `openai` | `requests` / `httpx` 手写 | 协议适配、异常分类、连接池都要自己做 |
| 模型聚合 | 单一 OpenAI 兼容适配器 | `litellm` / 多 SDK | 依赖膨胀；本项目只维护一条主路径 |
| 配置 | `dataclass` + 手写 `validate()` | `pydantic` / `attrs` | 规则少而具体；需要精确中文错误与退出码 |
| CLI | `argparse` | `click` / `typer` | 选项少；标准库已够用 |
| 日志 | `logging` | `loguru` / `structlog` | 宿主可接管；避免额外依赖 |
| HTTP 抓取 | `urllib.request` | `requests` / `httpx` | 需要自定义重定向校验；同步阻塞问题 |
| HTML 解析 | `re` + `html` | `BeautifulSoup` / `lxml` | 只有两个固定模式；避免解析器依赖 |
| 向量检索 | `list[float]` + `math` | `numpy` / `faiss` / `chromadb` | 单机中等规模足够；避免原生扩展 |
| 长期记忆 | JSONL + 关键词检索 | 向量库 / 数据库 | 记忆条目少、需要可读可审计；避免额外服务 |
| 子进程 | `asyncio.create_subprocess_shell` | `os.system` / `subprocess.run` | 需要捕获输出、超时与进程树终止；异步执行不占用工作线程 |
| 异步 | `asyncio` | `trio` / `anyio` | 标准库；并发模型简单 |
| 环境变量 | `os.environ` | `python-dotenv` | 部署侧管理环境；避免隐式读取 `.env` |
| 类型 | `typing` + `dataclass` | `pydantic` | 运行时校验由显式代码完成 |

### 4.3 什么时候应该新增依赖

满足以下任一条件再考虑：

1. **标准库实现会出现明显安全风险**（例如需要成熟的 HTML 解析器、TLS 策略、沙箱）；
2. **标准库实现显著更长且更难测试**（例如复杂重试、流式协议、向量索引）；
3. **性能成为瓶颈且已有基准**（例如向量规模超过内存暴力检索的承受范围）；
4. **依赖有明确维护者、许可证兼容、可接受供应链风险**；
5. **新增依赖后仍能保持核心抽象不被污染**（例如只替换 `Embedder` 实现，而不是让 `AgentRunner` 认识新库）。

否则，优先写一个小的、可审计的标准库实现。

---

## 5. 依赖边界地图

这张表用于快速回答："某个能力应该在哪一层引入？"

| 层 | 允许的依赖 | 不应该出现 |
|---|---|---|
| `providers/` | `openai`、`asyncio`、`dataclasses`、`abc`、`logging`、`os`、`random` | 业务工具、文件读写、渠道 |
| `agent/runner.py` | `asyncio`、`json`、内部类型 | 具体 SDK、HTTP、文件系统 |
| `agent/context.py` | `os`、`datetime`、内部 `Message` / `MemoryStore` / `sanitize_tool_messages` | `openai`、HTTP、工具执行 |
| `agent/request.py` | `contextvars`、`dataclasses`、`contextlib` | 模型 SDK、工具实现、渠道 |
| `agent/subagent.py` | `asyncio`、`contextlib`、`logging`、`time`、`uuid`、`datetime`、provider 抽象、记忆作用域；运行时延迟导入 runner/tools | `openai` 具体实现、HTTP、渠道 |
| `agent/loop.py` | `asyncio`、`logging`、总线/会话/runner/context/记忆作用域/请求上下文/子代理管理器 | `openai`、`subprocess`、`urllib` |
| `agent/tools/registry.py` | `asyncio`、`inspect`、`dataclasses`、`typing` | 网络、子进程 |
| `agent/tools/builtin.py` | `ast`、`operator`、`math`、`os`、`subprocess`、`datetime` | `openai`、HTTP |
| `agent/tools/web.py` | `urllib`、`socket`、`ipaddress`、`re`、`html` | 文件写入、子进程 |
| `agent/tools/knowledge.py` | `asyncio`、`math`、`threading`、`Embedder` | `openai`（通过抽象）、文件系统（通过 policy） |
| `agent/tools/memory.py` | 工具注册表、`memory/` 作用域与仓库 | 直接文件路径、网络、`openai` |
| `agent/tools/multi_agent.py` | `agent/request`、`agent/subagent`、工具注册表、策略开关 | 模型 SDK、HTTP、渠道 |
| `memory/` | `json`、`os`、`re`、`tempfile`、`threading`、`uuid`、`datetime`、`contextvars` | 网络、模型 SDK、渠道 |
| `utils/naming.py` | `re` | 文件 IO、网络 |
| `session/` | `json`、`os`、`tempfile`、`asyncio`、`dataclasses`、`naming.safe_key` | 网络、模型 SDK |
| `channels/` | `asyncio`、`contextlib`、`logging` | 模型调用、工具执行 |
| `config.py` | `json`、`os`、`logging`、`math`、`dataclasses` | 网络、文件工具 |
| `sdk.py` | 内部装配模块 | 具体协议细节 |
| `__main__.py` | `argparse`、`asyncio`、`signal`、`sys`、`json` | 业务实现 |

**判断规则**：如果某个导入会让上层认识底层实现（例如让 `AgentRunner` 认识 `openai`），说明依赖方向反了，应加抽象或把代码下沉。

---

## 6. 维护清单

### 升级 `openai`

- 重点检查 `chat.completions`、`embeddings`、异常类名与 `usage` 字段；
- 确认 `max_retries` 行为未变；
- 检查 `AsyncOpenAI.close()` 是否仍为协程。

### 新增工具

- 用 `@register_tool` 注册；
- 普通 IO 工具写成同步函数，`runner` 会用 `asyncio.to_thread` 执行；
- 需要 `await` 的能力（例如再跑一个 Agent）写成 `async def`，注册表会识别协程函数；
- 如果工具可以在同一轮里安全并行，注册时加 `concurrency_safe=True`；文件写入、有共享状态的工具不要加；
- 参数只使用 `str/int/float/bool`；
- 文件路径必须走 `resolve_workspace_path()`；
- 网络访问必须先用 `validate_url()`；
- 需要当前会话元数据时，从 `current_request_context()` 读取，不要新增模型参数；
- 失败时返回可读文字，不要抛栈给模型。

### 新增依赖

- 先判断能否用标准库完成；
- 记录许可证、维护状态、体积、原生扩展；
- 新依赖只能出现在对应层，不能泄漏到核心抽象；
- 在 `requirements.txt` 中给出合理的版本上限。

### 已知边界

- `re` 解析必应页面会随页面结构变化而失效；
- `subprocess(shell=True)` 不是沙箱；
- SSRF 防护存在 DNS rebinding 的理论窗口；
- 会话锁是单进程内的，多进程写同一会话不保证安全；
- 长期记忆同样是单进程文件级并发保护；多进程共享 `memory_dir` 可能互相覆盖。记忆按 `session_key` 隔离，但同名键（`a:b` 与 `a_b`）会落到同一文件；
- 长期记忆检索是关键词匹配，不理解同义表达；语义检索请用知识库或外部向量库；
- 提示词预算是**字符**而非 token，并且最后一条用户消息始终保留，可能略微超出预算；
- 知识库是内存态，重启后需重新摄入；
- `run_shell` 的输出与命令长度有上限，超长内容会被截断；
- 子代理共享父会话的 provider 与进程级工具策略；并发数、超时与结果长度都有上限，但**每个子代理都是一次完整 ReAct 循环**，会按倍增加 token 成本；
- 子代理结果按不可信数据处理：它可能来自网页或文件，通告文本有护栏但不等于沙箱；
- `asyncio.wait_for` 能取消子代理的模型请求与协程等待，**不能强杀**线程池中阻塞的同步工具；
- 子代理工具集剔除了 `delegate` / `spawn`，这是有意的防递归设计；需要更深的任务树时应自行编排而不是放开递归。

---

## 7. 一页速查

| 导入 | 最关键的函数/类 | 一句话作用 |
|---|---|---|
| `openai` | `AsyncOpenAI`、`chat.completions.create`、`embeddings.create`、5 个异常类 | 唯一第三方依赖，负责模型协议 |
| `asyncio` | `Queue`、`Task`、`Lock`、`Semaphore`、`Event`、`to_thread`、`wait_for`、`gather` | 并发骨架 |
| `threading` | `RLock`、`Lock` | 跨线程保护知识库索引与记忆缓存 |
| `logging` | `getLogger`、`StreamHandler`、`Formatter` | 统一日志出口 |
| `dataclasses` | `dataclass`、`field`、`fields`、`replace` | 值对象与配置覆盖 |
| `abc` | `ABC`、`abstractmethod` | 渠道/模型/嵌入接口 |
| `typing` | `Any`、`Callable`、`TYPE_CHECKING` | 类型标注与循环导入规避 |
| `collections.abc` | `Callable`、`Awaitable` | 子代理结果回调协议 |
| `json` | `load`、`loads`、`dumps` | 配置、参数、JSONL |
| `os` | `environ`、`replace`、`fsync`、`makedirs` | 环境变量与原子持久化 |
| `pathlib` | `Path.resolve`、`relative_to` | 工作区围栏 |
| `tempfile` | `mkstemp` | 同目录原子重写 |
| `argparse` | `ArgumentParser` | CLI |
| `sys` | `stdin.buffer`、`stderr`、`path` | 输入输出与直接运行 |
| `signal` | `SIGINT`、`SIGTERM`、`add_signal_handler` | 优雅停机 |
| `contextlib` | `suppress`、`nullcontext`、`contextmanager` | 吞掉预期取消异常；可选作用域 |
| `subprocess` | `run`、`TimeoutExpired` | 受控 shell 工具 |
| `ast` | `parse`、`Constant`、`BinOp`、`UnaryOp` | 安全表达式解析 |
| `operator` | `add/sub/mul/...` | 白名单运算 |
| `math` | `sqrt`、`isfinite` | 余弦相似度与数值校验 |
| `re` | `compile`、`sub`、`findall`、`search` | 会话键清洗、记忆分词与 HTML 提取 |
| `html` | `unescape` | HTML 实体反转义 |
| `urllib.request` | `Request`、`build_opener`、`HTTPRedirectHandler` | 带安全重定向的 HTTP GET |
| `urllib.parse` | `urlparse`、`quote` | URL 解析与编码 |
| `socket` | `getaddrinfo`、`gaierror` | DNS 解析 |
| `ipaddress` | `ip_address`、`is_global`、`ipv4_mapped` | SSRF 地址判定 |
| `inspect` | `getdoc`、`signature`、`iscoroutinefunction`、`Parameter.empty` | 自动生成工具 schema、识别异步工具 |
| `datetime` | `datetime.now`、`strftime`、`isoformat` | 当前时间工具、提示词日期、子代理提示词、记忆时间戳 |
| `time` | `monotonic` | 子代理耗时统计（不受系统时间调整影响） |
| `random` | `uniform` | 重试抖动 |
| `contextvars` | `ContextVar` | 把会话键与请求元数据安全地传给工具 |
| `uuid` | `uuid4` | 生成记忆条目 id 与子代理任务 id |
