# mini_agent

面向生产环境的轻量异步 AI Agent 运行时：一条命令启动，自带 Web 界面，工具、记忆、检索、多智能体、评测与部署细节都是默认能力，而不是可选插件。

- 单进程、可自托管：`pip install ./mini_agent` 后 `mini-agent` 即可用
- 严格配置：每个字段做类型与范围校验，坏配置在启动阶段就说清楚
- 密钥不落盘：优先写入系统钥匙串，配置文件只存与默认值的差异
- 可嵌入：`MiniBot` 门面与 `Agent` 轻量门面都能直接 `await bot.chat(...)`

---

## 一、重点功能

### 1. 对话与编排

| 能力 | 说明 |
|---|---|
| ReAct 多轮 | 模型请求工具 → 执行 → 结果回灌 → 继续推理，直到给出最终答案（`max_iterations` 默认 10） |
| 流式输出 | 文本增量实时推送；CLI 直接打印，Web 通过 SSE 推送 |
| 按会话分片 | 每个 `session_key` 一条队列 + 一个 worker：同会话串行，跨会话并行 |
| 过载保护 | 并发回合数、队列长度、活跃会话数都有上限，打满时返回"忙"而不是无限堆积 |
| 空转回收 | 会话空闲超过 `session_idle_timeout`（默认 900s）自动刷盘退出，缓存不无限增长 |

### 2. 工具系统（25 个内置工具）

| 分组 | 工具 | 默认 |
|---|---|---|
| 基础 | `get_current_time`、`calculate` | 开 |
| 文件（工作区内） | `read_file`、`write_file`、`edit_file`、`list_directory`、`glob`、`grep` | 开 |
| 命令 | `run_shell` | **关**（需 `allow_shell=true`） |
| 知识库 | `knowledge_ingest`、`knowledge_ingest_dir`、`knowledge_search`、`knowledge_list`、`knowledge_delete`、`knowledge_stats` | 开 |
| 长期记忆 | `memory_save`、`memory_search`、`memory_list`、`memory_forget` | 随 `memory_enabled` |
| 任务清单 | `todo_write`、`todo_read` | 随 `todos_enabled` |
| 多智能体 | `delegate`（等待结果）、`spawn`（后台派生） | 随 `multi_agent_enabled` |
| 网络 | `web_search`（必应）、`fetch_webpage`（正文提取） | 开 |

- **策略门控**：文件工具限制在工作区内，路径穿越直接拒绝；`allow_private_urls` 控制能否访问内网地址
- **人审**：`tool_approval=ask` 时，有副作用的工具调用先挂起等确认，超时（默认 120s）自动拒绝
- **并发**：标记为 `concurrency_safe` 的只读工具在同一轮内并发执行，其余串行
- **隔离**：同步工具丢进 `asyncio.to_thread` 执行，不阻塞事件循环；每个工具有独立超时

### 3. 上下文、记忆与检索

| 能力 | 说明 |
|---|---|
| 上下文拼装 | 人设 + 运行环境 + 长期记忆 + 任务清单 + 技能目录 + 预算内历史 |
| 双预算裁剪 | 字符硬上限（默认 32000）+ token 估算上限（默认 8000）；裁剪以"完整用户回合"为单位，最后一问永不裁掉 |
| 自动压缩 | 历史占预算超过 `context_compact_ratio`（0.8）时折叠成摘要，保留最近 4 轮原文 |
| 长期记忆 | 按会话隔离的 JSONL；关键词检索后按字数预算注入；模型可用 `memory_*` 主动增删查 |
| 记忆固化 | 规则兜底：识别"我叫 X""记住 …"这类高置信陈述自动入库，宁缺毋滥 |
| RAG 知识库 | 语义切块（默认 500 字 / 重叠 50）+ 持久化索引 + 关键词×向量混合检索 + MMR 去冗余 |
| 离线嵌入 | `embedding_backend=hashing` 时不联网也能检索；`openai` 走远端嵌入模型 |

### 4. 多智能体与技能

- `delegate`：子任务交给子代理，阻塞等结果，适合"查资料 → 汇总"这类可分解工作
- `spawn`：后台派生，完成后把结果注入回同一会话，适合长任务不阻塞对话
- 隔离与限流：子代理看不到父会话历史，工具集剔除 `delegate`/`spawn` 防递归；并发上限 4，单任务超时 300s
- 技能包：`~/.mini_agent/skills/<name>/SKILL.md` 或 `<workspace>/.mini_agent_skills/<name>/SKILL.md`，按需把正文注入当轮提示词；`/skills` 可查看清单

### 5. 渠道与界面

| 渠道 | 会话键 | 用途 |
|---|---|---|
| `web`（默认） | `web:<session>` | 内置单页聊天前端，HTTP + SSE，含首启设置页、知识库导入 |
| `cli` | `cli:local` | 终端交互对话 |
| `echo` | `echo:peer:*` | 本机自测回声渠道（默认端口 8766），用于验证总线与并发 |

Web 界面包含：对话流式渲染、会话切换/删除、记忆与任务清单查看、知识库导入/重建索引、配置编辑与连通性测试、工具审批弹窗。

### 6. 运维与可观测

- **预检三态**：`ready` / `setup_required`（能跑但没配好 → 进设置页）/ `blocked`（环境问题 → 给修复命令）
- **单实例复用**：同一份配置重复启动时探活旧实例，成功就只打开它的地址，避免两个进程抢同一份会话目录
- **端口避让**：本机自动模式下端口被占用会尝试相邻端口；`MINI_SERVICE=1` 或显式 `--port` 时直接报错
- **优雅停机**：SIGINT/SIGTERM 后限时排空，超时强制退出
- **追踪**：可选 Langfuse，默认关闭；开启后默认脱敏，不把对话正文与工具参数原样上报
- **错误翻译**：鉴权、超时、限流、模型不存在等异常转成"原因 + 下一步动作"，而不是甩堆栈

### 7. 评测

内置 65 条固定用例（qa 16 / rag 14 / tools 17 / context 12 / recovery 6），覆盖引用准确性、工具选择、上下文压缩、记忆干扰、工具失败重试、拒答与提示词泄露等场景，输出**回答、引用、工具、上下文、恢复**五项指标并生成 JSON/Markdown 报告。

---

## 二、功能实现

### 数据流

```
渠道（web / cli / echo）
   │  InboundMessage
   ▼
MessageBus（有界入站队列）
   │  按 session_key 分片
   ▼
会话队列 → worker ──► 命令层（/reset /memory /todos …）
                     └► ContextBuilder（拼提示词 + 预算裁剪）
                          └► AgentRunner（ReAct 循环）
                                ├► LLMProvider（OpenAI 兼容 + 重试）
                                └► Tool Registry（策略门控 + 审批 + 超时）
   ▲
   │  OutboundMessage
   ▼
渠道回写（SSE 流式 / 终端打印）
```

### 各层职责

**`providers/` 模型层**
`base.py` 用 dataclass 固定 `Message` / `ToolCall` / `LLMResponse` / `Usage`，`LLMProvider` 是唯一抽象接口。`openai_compat.py` 用 `AsyncOpenAI` 调 `chat.completions`，可直接接 OpenAI、DashScope、Ollama、vLLM 等兼容端点；`RetryingProvider` 以装饰器方式叠加指数退避（默认 3 次、基数 0.5s），把"重试策略"和"发请求"彻底分开。`embeddings.py` 提供 OpenAI 与离线 hashing 两种嵌入。

**`agent/runner.py` ReAct 循环**
发消息 → 收回复 → 有工具调用就执行并回灌 → 再发。同步工具走 `to_thread`，协程工具直接 `await`，`concurrency_safe` 工具同轮并发。每轮产出事件（增量、工具调用、工具结果、用量），渠道据此渲染进度。

**`agent/loop.py` 运行时装配**
消费总线、维护会话 worker、执行斜杠命令、触发压缩与持久化、把子代理结果注入回会话。同一会话的 SDK 直调与总线消息共用一把会话锁。

**`agent/context.py` 提示词拼装**
记忆块、清单块、技能目录都是"临时注入"，不写回 Session，避免 `/reset` 后残留。裁剪同时受字符与 token 两道预算约束，先保最近回合组，工具调用与其结果不拆散。

**`session/` 会话仓库 + 任务清单**
JSONL 增量追加（只写水位线之后的新消息），写完 `flush + fsync`；`/reset` 或文件损坏时整文件原子重写；读取跳过损坏行。`todo.py` 是独立 JSON：`pending` / `in_progress` / `completed` 三态，同一时刻最多一项进行中，写入走临时文件 + `os.replace`。

**`memory/` 长期记忆**
本地后端按 `session_key` 分文件存 JSONL，检索用关键词打分后按 `memory_inject_chars` 预算注入；容量上限 200 条，超限淘汰最旧。`mem0` 为可选后端（本地 Qdrant + 兼容接口抽取）。`auto.py` 用保守正则兜住"身份陈述/明确要求记住"两类事实。

**`agent/tools/knowledge.py` RAG**
按段落语义切块 → 嵌入（或 hashing）→ 落盘索引；检索时关键词与向量分数加权融合（`knowledge_lexical_weight`），再用 MMR（`knowledge_mmr_lambda`）去掉重复片段，最后取 `top_k`。支持按来源删除与重建索引。

**`agent/tools/registry.py` 工具注册表**
`@register_tool` 装饰函数即注册，参数 JSON Schema 从类型注解自动生成；`available` 回调做开关门控，`requires_approval` 触发人审，`is_async` 决定执行方式。模型看到的 schema 就是注册表里的实时状态。

**`agent/subagent.py` 子代理**
每个子代理 = 全新消息历史 + 同一 provider + 过滤后的工具集。`asyncio.Semaphore` 限制并发，后台任务总量另设上限，单任务超时即取消。

**`channels/web.py` Web 渠道**
标准库 `ThreadingHTTPServer` + SSE，不引入额外 Web 框架。只监听回环地址并校验 Host/Origin；无内建登录，对外必须由反向代理补认证与 TLS；页面内容全部转义后渲染；请求体大小、会话 ID、并发流数量均设上限。

| 接口 | 作用 |
|---|---|
| `GET /` `GET /setup` | 聊天页 / 首启设置页 |
| `GET /api/status` `/api/config` `/api/history` `/api/memory` `/api/todos` `/api/sessions` `/api/knowledge` | 状态与数据查询 |
| `GET /api/stream` | SSE 事件流（增量、工具、审批、结束） |
| `POST /api/message` `/api/stop` `/api/approval` | 发消息、中断、审批应答 |
| `POST /api/setup` `/api/setup/test` | 保存配置、测试连通性 |
| `POST /api/knowledge/import` `/delete` `/reindex` | 知识库维护 |
| `GET /healthz` `/readyz` | 存活/就绪探针（部署用） |

**`config.py` + `utils/credentials.py` 配置与凭据**
四层合并：默认值 < 用户配置 `~/.mini_agent/config.json` < 项目配置 `.mini_agent.json` < 环境变量 < 命令行参数。每字段校验类型与范围，失败抛 `ConfigError`；Web 修复模式改用宽松加载，把问题收进报告而不是挡住启动。写回只落"与默认值不同"的键，`api_key` 永不写入；密钥按端点主机名存入系统钥匙串，环境变量优先以便密钥轮换。

**`runtime/` 启动支撑**
`preflight.py` 归纳 `ready` / `setup_required` / `blocked` 三态并给出修复命令；`instance.py` 用 `/healthz` 探活复用已有实例，陈旧记录直接接管。

**`evals/` 评测框架**
用例以 JSONL 定义（期望答案、要求引用的来源、可用工具、超时），跑完后按五项指标打分并输出报告；支持按分类/标签筛选、`--dry-run` 静态校验。

### 目录结构

```text
mini_agent/
├── __init__.py          # 包入口：版本与常用导出
├── __main__.py          # 命令行入口：web / chat / login / doctor / init / check
├── config.py            # 配置模型、分层加载、差异写回
├── sdk.py               # MiniBot 门面：装配全部部件
├── agent/               # ReAct runner、loop、context、commands、skills、subagent、tools/
├── bus/                 # MessageBus：渠道与内核解耦
├── channels/            # web（含内置前端）、cli、echo
├── session/             # 会话仓库 + 任务清单
├── memory/              # 长期记忆（local / mem0）
├── providers/           # 模型抽象、OpenAI 兼容实现、重试、嵌入
├── runtime/             # 启动预检、单实例复用
├── utils/               # 凭据、错误翻译、命名、URL 判定、日志、追踪
├── evals/               # 固定评测集与报告
└── tests/               # 单元测试
```

---

## 三、启动方法

### 环境要求

Python 3.11+，Windows / macOS / Linux。默认依赖只有 `openai` 与 `keyring`。

### 安装

```bash
# 在包含 mini_agent/ 的父目录执行
python -m pip install ./mini_agent

# 或开发模式（改代码即时生效）
python -m pip install -e ./mini_agent

# 可选能力
python -m pip install "./mini_agent[langfuse]"   # Langfuse 追踪
python -m pip install "./mini_agent[mem0]"       # Mem0 记忆后端
```

### 三种启动方式

**1）Web 界面（默认，推荐）**

```bash
mini-agent                 # 打开 http://127.0.0.1:8767
python -m mini_agent       # 等价写法
```

首次运行没有密钥时会落到设置页：填端点、模型、密钥，点"测试连接"通过后即可对话。密钥存入系统钥匙串，之后启动不再询问。

**2）终端对话**

```bash
mini-agent chat
mini-agent chat -s work:daily      # 指定会话键
mini-agent -m "帮我总结 README"    # 单轮：答完即退出
```

**3）一次性配置（无图形界面）**

```bash
mini-agent login           # 终端向导，把密钥写入钥匙串
mini-agent init            # 生成最小配置文件 .mini_agent.json
mini-agent doctor          # 环境诊断：Python、密钥、目录、渠道、工具
mini-agent --check         # 打印生效配置与启用工具后退出
```

### 子命令与常用参数

| 子命令 | 作用 |
|---|---|
| （无） | 预检 → 打开 Web 界面 |
| `web` | 只启动 Web 渠道 |
| `chat` | 终端交互对话 |
| `login` | 终端凭据向导 |
| `doctor` | 环境诊断，不访问网络 |
| `init` | 生成配置模板，不覆盖已有文件 |

| 参数 | 作用 |
|---|---|
| `-c, --config PATH` | 指定配置文件 |
| `--web-port, --port N` | 覆盖 Web 端口（显式指定时被占用直接报错） |
| `--no-browser` | 启动 Web 时不自动开浏览器 |
| `-m, --message TEXT` | 单轮模式，`-` 表示从 stdin 读 |
| `-s, --session KEY` | 会话键（默认 `cli:local`） |
| `--model` / `--base-url` | 覆盖模型名与端点 |
| `--stream` / `--no-stream` | 覆盖流式开关 |
| `--log-level` | 覆盖日志级别 |
| `--check` / `--version` | 校验配置 / 打印版本 |

### 接入本地模型或其它兼容端点

```bash
# Ollama
export OPENAI_BASE_URL="http://127.0.0.1:11434/v1"
export MODEL="qwen2.5:7b"
mini-agent

# 通义千问（DashScope 兼容模式）
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="sk-..."
export MODEL="qwen-plus"
mini-agent
```

本地端点（`127.0.0.1` / `localhost`）默认允许无密钥；远程端点缺密钥会在启动阶段直接报错。

### 环境变量

环境变量优先级高于配置文件，适合 CI、容器与临时覆盖。

| 变量 | 作用 |
|---|---|
| `OPENAI_API_KEY` | 远程端点密钥（覆盖钥匙串） |
| `OPENAI_BASE_URL` | 兼容端点地址 |
| `MODEL` | 模型名 |
| `MINI_CONFIG` | 配置文件路径（等价 `-c`） |
| `MINI_WORKSPACE` | 工具可访问的工作区根目录 |
| `MINI_CHANNELS` | 启用渠道，逗号分隔，如 `cli,web` |
| `MINI_WEB_PORT` | Web 端口（默认 8767） |
| `MINI_LOG_LEVEL` | 日志级别 |
| `MINI_SESSIONS_DIR` / `MINI_MEMORY_DIR` / `MINI_TODOS_DIR` / `MINI_KNOWLEDGE_DIR` | 各类数据目录 |
| `MINI_MEMORY_BACKEND` | `local` 或 `mem0` |
| `MINI_EMBEDDING_BACKEND` | `openai` 或 `hashing`（离线） |
| `MINI_ALLOW_SHELL` / `MINI_ALLOW_PRIVATE_URLS` / `MINI_ALLOW_NO_API_KEY` | 安全开关 |
| `MINI_TOOL_APPROVAL` | `ask`（默认）或 `off` |
| `MINI_SERVICE=1` | 无人值守模式：端口被占用直接报错 |
| `MINI_NO_BROWSER=1` | 启动时不打开浏览器 |

### 配置文件

优先级：**默认值 < `~/.mini_agent/config.json` < `.mini_agent.json` < 环境变量 < 命令行参数**。显式 `--config` 时只读该文件，不再叠加用户配置。

```json
{
  "model": "gpt-4o-mini",
  "api_base": "https://api.openai.com/v1",
  "channels": ["cli", "web"],
  "workspace": ".",
  "allow_shell": false,
  "tool_approval": "ask",
  "memory_enabled": true,
  "embedding_backend": "hashing",
  "knowledge_top_k": 4
}
```

只写需要覆盖的项即可，其余取默认值。完整字段见 `config.example.json`；常用默认值：

| 字段 | 默认 | 说明 |
|---|---|---|
| `max_iterations` | 10 | 单轮最多几轮工具调用 |
| `history_limit` | 50 | 携带的历史消息条数 |
| `prompt_max_chars` / `prompt_max_tokens` | 32000 / 8000 | 上下文双预算 |
| `context_compact_ratio` | 0.8 | 触发自动压缩的比例 |
| `max_concurrent_turns` | 8 | 全局并发回合上限 |
| `session_idle_timeout` | 900 | 空闲回收秒数 |
| `web_port` / `echo_port` | 8767 / 8766 | 端口 |
| `retry_max_attempts` / `retry_base_delay` | 3 / 0.5 | 重试策略 |
| `memory_max_entries` / `memory_inject_chars` | 200 / 4000 | 记忆容量与注入预算 |
| `tool_timeout` / `turn_timeout` | 120 / 300 | 单个工具与整轮超时 |

### 数据落盘位置

| 内容 | 默认位置 |
|---|---|
| 会话历史 | `.mini_agent_sessions/` |
| 长期记忆 | `.mini_agent_memory/`（mem0 用 `.mini_agent_mem0/`） |
| 任务清单 | `.mini_agent_todos/` |
| 知识库索引 | `.mini_agent_knowledge/` |
| 用户级配置 / 技能 / 单实例记录 | `~/.mini_agent/` |
| 密钥 | 系统钥匙串（Windows 凭据管理器 / macOS Keychain / Secret Service） |

### 内置斜杠命令

`/help`、`/status`、`/reset`、`/compact`、`/memory`、`/todos`、`/skills`；终端里 `/exit`、`/quit`、`/q` 退出。

### 测试与评测

```bash
# 单元测试（在包含 mini_agent/ 的父目录执行）
python -m unittest discover -s mini_agent/tests -t .
# 或
python -m pytest mini_agent/tests

# 评测：先静态校验用例，再跑全量
python -m mini_agent.evals --dry-run
python -m mini_agent.evals --only rag,tools --limit 10
python -m mini_agent.evals --output eval_reports
```

报告默认写到 `eval_reports/`，包含每条用例的回答、引用命中、工具轨迹与五项汇总指标。

### 对外部署

Web 渠道只监听 `127.0.0.1` 且没有内建登录，本机使用是安全的；要让别人访问，请在前面加一层反向代理（Nginx / Caddy）做 TLS 与认证，并用 `MINI_SERVICE=1` 关闭端口自动避让，避免健康检查连到错误端口。

---

## 常见问题

**端口被占用**：本机模式下会自动尝试相邻端口；想固定端口就用 `--port` 或 `MINI_SERVICE=1`，冲突时会明确报错。

**提示未配置 API 密钥**：远程端点必须给密钥。跑 `mini-agent login`，或设 `OPENAI_API_KEY`；纯本地模型可加 `allow_no_api_key=true`。

**工具里没有 `run_shell`**：默认关闭，需要显式 `allow_shell=true`；开启后命令仍在工作区沙箱内执行。

**访问文件被拒绝**：文件工具只能访问 `workspace` 之内的路径，`..` 越界会被拒绝。

**中文路径下测试报错**：部分 shell 相关用例在非 ASCII 路径下会因系统编码差异失败，属于环境问题，把项目放到英文路径即可。

**改了代码没生效**：用 `pip install -e ./mini_agent` 安装，或确认运行的是同一个解释器环境。
