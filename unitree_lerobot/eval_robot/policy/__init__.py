"""Policy WebSocket 传输层入口。

真正实现位于 ``web_policy``，其中同时包含服务端和客户端两侧：

  - ``web_policy.PolicyService``：把一个本地 policy 包装成 WebSocket 服务。
    当前设计是一进程服务一个 policy。
  - ``web_policy.RemotePolicy``：连接到上述服务端，并像调用本地 policy 一样调用远端。

这里的代码只负责通用网络传输，不负责定义具体 observation / action 规范。
观测和动作契约由 README 描述，并由 ``example/example_env.py`` 在 policy 边界校验。
"""
