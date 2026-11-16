"""跨模块小工具：与业务无关的通用能力。

    credentials.py    凭据读写（系统钥匙串优先，密钥不落配置文件）
    errors.py         把底层异常翻译成用户能照做的提示
    naming.py         session_key -> 安全且互不碰撞的文件名
    netutil.py        回环地址判定（配置层与 Provider 层共用）
    observability.py  进程级日志配置
    tracing.py        可选 Langfuse 追踪（默认关闭）

子模块按需单独导入，包级不做预加载，避免为一个纯函数拉起整条依赖链。
"""
