"""通过 WebSocket 在网络两端调用策略（policy）的传输层。

这里的 *policy* 指任何同时暴露下面三个成员的对象：

    policy.metadata           -> dict，描述观测 / 动作的数据契约
    policy.get_action(obs)    -> dict，输入一段观测后返回一段动作
    policy.reset()            -> 清空每个 episode 级别的内部状态

``PolicyService`` 负责把本地 policy 包装成 WebSocket 服务；模型推理等
计算较重的部分可以部署在一台机器上，而 ``RemotePolicy`` 可以从另一台机器
连接过去，并像调用本地 policy 一样调用远端策略。网络上传输的数据使用
msgpack 编码；numpy 数组和 numpy 标量通过本文件里的自定义 hook 转换。
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback

import msgpack
import numpy as np
from websockets.asyncio.server import serve as _open_server
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode
from websockets.sync.client import connect as _open_client

__all__ = ["PolicyService", "RemotePolicy"]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Numpy <-> msgpack                                                           #
# --------------------------------------------------------------------------- #
#
# msgpack 本身不知道如何表示 numpy.ndarray / numpy 标量，所以这里定义两个
# hook 做“编码前转换”和“解码后还原”。数组会被拆成三部分：原始字节、dtype、
# shape；接收端再用这三部分精确重建数组。numpy 标量则先转成 Python 原生值。
# object / void / complex 这类 dtype 很难保证跨进程、跨语言稳定还原，因此提前拒绝。

_OPAQUE_KINDS = frozenset("OVc")  # numpy dtype.kind: object / void / complex


def _to_portable(value):
    """msgpack 的 ``default`` hook：把 numpy 值转换成 msgpack 可编码的字典。"""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in _OPAQUE_KINDS:
        raise ValueError(f"cannot serialize values of dtype {value.dtype!r}")
    if isinstance(value, np.ndarray):
        # ndarray 不能直接 msgpack，因此显式保存字节内容、dtype 字符串和 shape。
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        # numpy scalar 转成 Python 标量，同时保留 dtype，避免 int/float 精度信息丢失。
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _from_portable(obj):
    """msgpack 的 ``object_hook``：把 ``_to_portable`` 编码过的 numpy 值还原回来。"""
    if b"__ndarray__" in obj:
        # buffer 指向 msgpack 解出来的 bytes；dtype 和 shape 保证数组布局与发送端一致。
        return np.ndarray(
            shape=obj[b"shape"],
            dtype=np.dtype(obj[b"dtype"]),
            buffer=obj[b"data"],
        )
    if b"__npgeneric__" in obj:
        # 用保存下来的 dtype 类型包装 data，恢复成 numpy scalar。
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class _Channel:
    """单条连接使用的 msgpack 编解码器。

    ``Packer`` 内部会维护状态，官方也建议复用；因此每条连接都持有自己的
    ``_Channel``。解码没有状态，直接用 ``unpackb`` 即可。
    """

    def __init__(self) -> None:
        # default 指向 _to_portable，让 msgpack 在遇到 numpy 值时自动走自定义转换。
        self._packer = msgpack.Packer(default=_to_portable)

    def freeze(self, obj) -> bytes:
        """把 Python 对象编码成可通过 WebSocket 发送的 bytes。"""
        return self._packer.pack(obj)

    @staticmethod
    def thaw(blob):
        """把 WebSocket 收到的 bytes 解码回 Python 对象。"""
        return msgpack.unpackb(blob, object_hook=_from_portable)


# --------------------------------------------------------------------------- #
#  Caller side                                                                 #
# --------------------------------------------------------------------------- #


class RemotePolicy:
    """远端 policy 的本地代理对象。

    构造 ``RemotePolicy`` 时会阻塞等待服务端连接成功，并读取服务端发来的第一帧
    metadata。之后调用 ``get_action`` / ``reset`` 看起来像普通方法调用，实际会
    通过 WebSocket 做一次请求-响应往返。
    """

    _RECONNECT_GAP = 5  # 连接被拒绝时，两次重试之间等待的秒数

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._uri = self._resolve_uri(host, port)
        self._api_key = api_key
        self._channel = _Channel()
        self._socket, self._metadata = self._handshake()

    @staticmethod
    def _resolve_uri(host: str, port: int | None) -> str:
        # 如果调用方已经给了 ws:// 或 wss:// 完整地址，就直接使用；否则用 host/port 拼接。
        base = host if host.startswith("ws") else f"ws://{host}"
        return base if port is None else f"{base}:{port}"

    @property
    def metadata(self) -> dict:
        return self._metadata

    def _handshake(self):
        # 一直重试直到连接成功；连接建立后服务端发送的第一帧就是 metadata。
        # api_key 如果存在，会放入 Authorization 头，便于外层网关做鉴权。
        auth = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        log.info("connecting to policy server at %s", self._uri)
        while True:
            try:
                socket = _open_client(
                    self._uri,
                    additional_headers=auth,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    ping_timeout=None,
                )
                return socket, self._channel.thaw(socket.recv())
            except ConnectionRefusedError:
                # 服务端可能还在启动中；这里不直接失败，而是持续等待。
                log.info("server not up yet; retrying in %ss", self._RECONNECT_GAP)
                time.sleep(self._RECONNECT_GAP)

    def _request(self, payload: dict):
        # 请求和响应是一一对应的同步往返：发送 msgpack bytes，等待服务端回复。
        # 正常回复是二进制帧；如果收到文本帧，约定为服务端 traceback，转成本地异常。
        self._socket.send(self._channel.freeze(payload))
        reply = self._socket.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"policy server raised an error:\n{reply}")
        return self._channel.thaw(reply)

    def get_action(self, obs: dict) -> dict:
        """把观测发送到远端 policy，并返回远端算出的动作 chunk。"""
        return self._request({"type": "get_action", "obs": obs})

    def reset(self):
        """通知远端 policy 重置 episode 状态。"""
        return self._request({"type": "policy_reset"})


# --------------------------------------------------------------------------- #
#  Hosting side                                                                #
# --------------------------------------------------------------------------- #


class PolicyService:
    """把一个本地 policy 发布为 WebSocket 服务。

    每个客户端连上来后，服务端会先发送 msgpack 编码的 metadata。之后客户端通过
    带 ``type`` 字段的请求字典来驱动 policy：``get_action`` 调用推理，
    ``policy_reset`` 重置状态。若 handler 抛异常，服务端会先把 traceback 作为
    文本帧发给客户端，再用错误状态码关闭连接，方便客户端侧看到真实错误原因。
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        # 先放入外部传入的 metadata，再叠加 policy.metadata；如果 key 冲突，以 policy 为准。
        self._metadata = {**(metadata or {}), **policy.metadata}
        # 请求中的 "type" 字符串 -> 实际处理函数。
        self._routes = {
            "get_action": lambda req: self._policy.get_action(req["obs"]),
            "policy_reset": lambda req: self._policy.reset(),
        }
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def run_forever(self) -> None:
        """在当前线程启动服务，持续运行直到进程被中断。"""
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        async with _open_server(
            self._on_connection,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            # 两端都关闭 keepalive ping。一次模型推理可能比默认 ping 间隔更久，
            # 不希望慢请求被误判成连接断开。
            ping_interval=None,
            ping_timeout=None,
            process_request=_liveness_probe,
        ) as server:
            await server.serve_forever()

    async def _on_connection(self, socket) -> None:
        who = socket.remote_address
        log.info("client %s connected", who)
        channel = _Channel()

        # 连接建立后先发 metadata，让客户端知道 obs/action 的数据契约。
        await socket.send(channel.freeze(self._metadata))

        try:
            async for frame in socket:
                # 每个客户端请求都是一个 msgpack 二进制帧，解码后根据 type 分发。
                request = channel.thaw(frame)
                kind = request.get("type")
                route = self._routes.get(kind)
                if route is None:
                    raise ValueError(f"unrecognized request type: {kind!r}")
                # handler 返回值同样用 msgpack 编码为二进制帧发回客户端。
                await socket.send(channel.freeze(route(request)))
        except ConnectionClosed:
            log.info("client %s disconnected", who)
        except Exception:
            # 把服务端 traceback 发回客户端，便于远端调用者定位错误；随后关闭连接并重新抛出。
            await socket.send(traceback.format_exc())
            await socket.close(
                code=CloseCode.INTERNAL_ERROR,
                reason="server-side failure; traceback in the previous frame",
            )
            raise


def _liveness_probe(connection, request):
    """响应健康检查 HTTP GET；其他请求继续走正常 WebSocket upgrade 流程。"""
    if request.path == "/healthz":
        # 允许外部探针用 http://host:port/healthz 判断服务是否存活。
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
