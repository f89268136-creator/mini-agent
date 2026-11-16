"""进程运行时支撑：启动前的可行性判定与单实例复用。

    preflight.py  ready / setup_required / blocked 三态预检
    instance.py   同一份配置只保留一个 Web 进程，探活失败即接管陈旧记录

两者都只依赖配置层，不碰 Agent 内核，便于在 CLI/SDK 里单独调用。
"""
