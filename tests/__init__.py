"""mini_agent 离线自测包（不需要模型和网络）。

三条测试分别运行：
    python -m mini_agent.tests.test_offline           # ReAct 主回路 / 工具守卫 / 重试策略
    python -m mini_agent.tests.test_rag_offline       # 切块 / 假嵌入 / 向量检索
    python -m mini_agent.tests.test_pipeline_offline  # 总线 / 分片 / 持久化 / 命令 / SDK 全链路 E2E
"""
