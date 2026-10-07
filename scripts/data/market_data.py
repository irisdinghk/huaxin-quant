"""
数据源抽象层 — 日线数据获取的统一入口

当前实现:
  - TDXSource: 通达信 mootdx（主数据源）
  - MiaoxiangSource: 东方财富妙想 API（备用数据源）

设计原则:
  - DataSource 抽象基类定义 fetch_bars(code, name) → DataFrame | None 接口
  - 各实现封装自己的 API 调用、字段解析、重试逻辑
  - 返回统一的 OHLCV DataFrame (date, open, high, low, close, volume, turnover)
"""

import abc
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

from scripts.shared import PROJECT_ROOT, RateLimiter

# ===================== 常量 =====================

MX_BASE_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/query"
MAX_RETRIES = 3

TDX_DECODED_ZERO = 2.0 ** -127


def normalize_tdx_decoded_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    """tdxpy decodes a packed zero quantity as 2**-127 rather than zero."""
    data = frame.copy()
    for column in ("vol", "volume", "amount"):
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce")
            data.loc[values == TDX_DECODED_ZERO, column] = 0.0
    return data

# 字段别名：整合 quant_filter 的精确匹配 + tracker 的模糊匹配
# 按优先级排列，先精确后模糊
FIELD_GROUPS = [
    ("收盘价", ["收盘价", "收盘", "当日收盘价", "最新价"]),
    ("开盘价", ["开盘价", "开盘", "当日开盘价"]),
    ("最高价", ["最高价", "最高", "当日最高价"]),
    ("最低价", ["最低价", "最低", "当日最低价"]),
    ("成交量", ["成交量", "成交数量", "成交股数"]),
    ("换手率", ["换手率", "换手"]),
]


# ===================== 抽象基类 =====================

class DataSource(abc.ABC):
    """日线数据源抽象基类。

    子类只需实现 fetch_bars() 和 display_name。
    """

    @abc.abstractmethod
    def fetch_bars(self, code: str, name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """拉取近约 200 个交易日日线数据。

        Args:
            code: 6 位股票代码，如 '300604'
            name: 股票名称，如 '长川科技'

        Returns:
            (DataFrame, None) — 成功，DataFrame 列: date, open, high, low, close, volume, turnover
            (None, error_message) — 失败，error_message 为人类可读的错误原因
        """
        ...

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """人类可读的数据源名称，如 'mx-api'、'tdx'"""
        ...


# ===================== 妙想 API 数据源（备用） =====================

class MiaoxiangSource(DataSource):
    """东方财富妙想 API 数据源。

    整合了 quant_filter.py 和 tracker.py 两套 fetch_daily() 的优点:
      - 表选择: 遍历所有表索引查找完整 OHLCV 表（quant_filter 方式，更稳健）
      - 字段解析: 别名列表 + 模糊子串匹配（tracker 方式，支持 ETF 字段变体）
      - 错误消息: quant_filter 的详细版本
    """

    display_name = "mx-api"

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        self._rl = rate_limiter or RateLimiter()

    # ── 公共接口 ──

    def query_tool(self, query: str) -> Tuple[Optional[dict], Optional[str]]:
        """执行通用妙想查数请求，供其他数据适配器复用。"""
        api_key = os.environ.get("MX_APIKEY")
        if not api_key:
            return None, "MX_APIKEY 未设置"
        headers = {"Content-Type": "application/json", "apikey": api_key}
        payload = {"toolQuery": query, "toolType": "query_tool"}
        return self._do_request(headers, payload)

    def fetch_bars(self, code: str, name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """通过妙想 API 获取日线数据。返回 (DataFrame, None) 或 (None, error_msg)。

        查询策略：以代码为主锚点，避免 NLP 对简称（尤其是 ETF）匹配到残缺数据表。
        1. 主查询："{code} {name}近200个交易日..."
        2. 兜底：  "{code}近200个交易日..."  （纯代码，消除歧义）
        """
        api_key = os.environ.get("MX_APIKEY")
        if not api_key:
            return None, "MX_APIKEY 未设置"

        queries = [
            f"{code} {name}近200个交易日每日开盘价、最高价、最低价、收盘价、成交量、换手率",
            f"{code}近200个交易日每日开盘价、最高价、最低价、收盘价、成交量、换手率",
        ]

        headers = {"Content-Type": "application/json", "apikey": api_key}
        last_error = None

        for qi, query in enumerate(queries):
            payload = {"toolQuery": query, "toolType": "query_tool"}
            result, req_err = self._do_request(headers, payload)
            if result is None:
                last_error = req_err or "网络请求失败"
                continue

            raw, resolved, err = self._extract_table(result)
            if raw is not None:
                df = self._parse_rows(raw, resolved)
                if df is not None:
                    return df, None
                last_error = "无有效交易日数据(可能长期停牌)"
                continue

            last_error = err or "未找到完整历史行情表(字段缺失或结构异常)"

        return None, last_error

    def _extract_table(self, result: dict) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
        """从 API 响应中提取完整 OHLCV 表。

        返回 (raw_table, resolved_fields, error_msg)。
        表不完整或结构异常时 raw_table 为 None。
        """
        try:
            inner_msg = result.get("data", {}).get("data", {}).get("message", "")
        except (KeyError, TypeError, AttributeError):
            inner_msg = ""

        if inner_msg and ("上限" in str(inner_msg) or "额度" in str(inner_msg)):
            return None, None, str(inner_msg)

        try:
            tables = result["data"]["data"]["searchDataResultDTO"]["dataTableDTOList"]
        except (KeyError, TypeError):
            return None, None, "数据结构异常(缺 dataTableDTOList)"

        raw, resolved = self._select_table(tables)
        if raw is None:
            return None, None, str(inner_msg) if inner_msg else None

        return raw, resolved, None

    # ── HTTP 请求 + 重试 ──

    def _do_request(self, headers, payload):
        """执行 HTTP 请求，含重试和错误码处理。

        Returns (data_dict, None) 成功，或 (None, error_message) 失败。
        error_message 保留具体错误码供上层做重试/fatal stop 决策。
        """
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(MX_BASE_URL, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.Timeout:
                if attempt < 1:
                    self._rl.wait(is_fail=True)
                    continue
                return None, "网络超时(重试3次仍失败)"
            except requests.exceptions.ConnectionError:
                if attempt < 1:
                    self._rl.wait(is_fail=True)
                    continue
                return None, "网络连接失败(重试3次仍失败)"
            except Exception:
                return None, "未知网络错误"

            code = data.get("code", -1)
            if code == 0:
                self._rl.reset_fails()
                return data, None
            if code == 112:  # 频率限制
                if attempt < 2:
                    self._rl.wait(is_fail=True)
                    time.sleep(5)
                    continue
                return None, "112 频率限制(重试耗尽)"
            if code == 113:  # 调用上限 — 致命
                return None, "FATAL:113 本周调用额度已用完"
            if code == 114:  # Key 失效 — 致命
                return None, "FATAL:114 API Key 无效或已过期"
            if code == 115:  # 无数据
                return None, "115 无数据"
            if attempt == 0:
                self._rl.wait(is_fail=True)
                continue
            return None, f"API 错误码 {code}"

        return None, "未知错误(重试耗尽)"

    # ── 表选择（quant_filter 的遍历方式）──

    def _select_table(self, tables):
        """遍历所有返回的表，选择第一个包含全部 OHLCV 字段的历史行情表。"""
        for idx, table in enumerate(tables or []):
            try:
                raw = table["rawTable"]
                name_map = table["nameMap"]
            except (KeyError, TypeError):
                continue

            ind_map = {v: k for k, v in name_map.items() if k != "headNameSub"}
            resolved = self._resolve_fields(ind_map)
            if resolved is None:
                continue
            if not raw.get("headName"):
                continue
            return raw, resolved

        return None, None

    # ── 字段解析（tracker 的模糊匹配方式）──

    def _resolve_fields(self, ind_map):
        """精确匹配 → 模糊子串匹配，返回 {canonical_name: raw_key}。"""
        resolved = {}
        for canonical, aliases in FIELD_GROUPS:
            # 第一轮：精确匹配
            for alias in aliases:
                if alias in ind_map:
                    resolved[canonical] = ind_map[alias]
                    break
            if canonical in resolved:
                continue
            # 第二轮：模糊子串匹配
            for alias in aliases:
                for field_name in ind_map:
                    if alias in field_name:
                        resolved[canonical] = ind_map[field_name]
                        break
                if canonical in resolved:
                    break
            if canonical not in resolved:
                return None
        return resolved

    # ── 行解析 ──

    def _parse_rows(self, raw, resolved):
        """将 rawTable 的行转换为标准化 DataFrame。"""
        rows = []
        for i, d in enumerate(raw["headName"]):
            vol = raw[resolved["成交量"]][i]
            turn = raw[resolved["换手率"]][i]
            opn = raw[resolved["开盘价"]][i]
            high = raw[resolved["最高价"]][i]
            low = raw[resolved["最低价"]][i]
            close = raw[resolved["收盘价"]][i]

            # 停牌日任一字段为 '-' 则跳过
            if "-" in (vol, turn, opn, high, low, close):
                continue

            rows.append({
                "date": d,
                "open": float(opn),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(vol),
                "turnover": float(turn),
            })

        if not rows:
            return None

        return pd.DataFrame(rows)


# ===================== 通达信数据源（mootdx，主） =====================

class TdxPytdxClient:
    """pytdx 客户端适配器 —— 暴露与 mootdx client 兼容的 xdxr() 接口。

    背景（2026-09-16）：mootdx 的 bars() 对所有服务器返回 0 行，
    导致 _get_client() 判定 10 台服务器全部不可用，
    backtest.py 取除权数据时必然抛 "通达信连接失败"。
    pytdx 直连同一批服务器完全正常（前端筛选 01-screen1.py 一直在用），
    故此处提供适配器，字段与 mootdx 保持一致：
      category / fenhong / songzhuangu / peigu / peigujia / year / month / day
    """

    def __init__(self, api, ip, port):
        self._api = api
        self.ip = ip
        self.port = port

    def xdxr(self, symbol=None, code=None):
        """返回该股除权除息事件的 DataFrame（与 mootdx 结构兼容）。"""
        target = symbol or code
        market = 1 if str(target).startswith("6") else 0
        records = self._api.get_xdxr_info(market, target)
        return pd.DataFrame(records or [])


class TDXSource(DataSource):
    """通达信数据源，基于 mootdx 库连接公共行情服务器。

    作为日线主数据源，免费、无额度限制。
    每次初始化时用 sync=False 轻量探测（与 CLI 行为一致），
    取延迟最低的服务器直连。不同于 factory(bestip=True)，
    后者用 sync=True 探测后立即建连，会被服务器限流。
    依赖: pip install 'mootdx[all]'

    注意：_get_client() 优先返回 pytdx 适配器，仅当 pytdx 不可用时
    回退 mootdx（见 _get_client 注释）。
    """

    display_name = "tdx"

    _CACHE_FILE = Path(PROJECT_ROOT) / "cache" / "tdx_servers.json"

    def __init__(self):
        self._client = None

    @classmethod
    def _load_cached_servers(cls):
        """读取上次探测成功的服务器缓存。"""
        try:
            if cls._CACHE_FILE.exists():
                data = json.loads(cls._CACHE_FILE.read_text())
                if isinstance(data, list) and data:
                    return [(item[0], item[1]) for item in data]
        except Exception:
            pass
        return []

    @classmethod
    def _save_cached_servers(cls, servers):
        """保存成功探测的服务器列表（最多保留 10 台）。"""
        try:
            cls._CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            cls._CACHE_FILE.write_text(json.dumps(servers[:10]))
        except Exception:
            pass

    def _get_client(self):
        """获取或创建行情客户端。

        优先级（2026-09-16 调整）：
        1. **pytdx 适配器** — 直连 TDX 服务器，已验证稳定可靠。
           mootdx 的 bars() 坏了之后，这是唯一能拿到 xdxr 的路径。
        2. mootdx — 仅当 pytdx 不可用时回退（保留原逻辑以备上游修复）。

        pytdx 是同步阻塞库，用锁串行化访问。
        """
        if self._client is not None:
            return self._client

        client = self._build_pytdx_client()
        if client is not None:
            self._client = client
            return client

        return self._build_mootdx_client()

    def _build_pytdx_client(self):
        """用 pytdx 建连，成功返回适配器，失败返回 None。"""
        try:
            from pytdx.hq import TdxHq_API
        except ImportError:
            return None

        candidates = self._load_cached_servers() or [
            ("180.153.18.170", 7709),
            ("60.12.136.250", 7709),
            ("115.238.56.198", 7709),
        ]
        for ip, port in candidates:
            api = TdxHq_API()
            try:
                if not api.connect(ip, port):
                    continue
                # 用真实请求验证：拿 000001 的除权数据（xdxr 是我们要用的接口）
                probe = api.get_xdxr_info(0, "000001")
                if probe:
                    self._save_cached_servers([(ip, port)])
                    return TdxPytdxClient(api, ip, port)
                api.disconnect()
            except Exception:
                try:
                    api.disconnect()
                except Exception:
                    pass
                continue
        return None

    def _build_mootdx_client(self):
        """回退路径：原 mootdx 逻辑（服务器探测 → 逐个试连）。"""
        from mootdx.server import server as probe_servers
        from mootdx.quotes import Quotes

        candidates = []  # [(ip, port), ...]

        # 尝试实时探测
        try:
            results = probe_servers(index='HQ', limit=5, sync=False)
            if results:
                self._save_cached_servers(results)
                candidates = results[:5]
        except Exception:
            pass

        # 探测失败，回退到缓存
        if not candidates:
            candidates = self._load_cached_servers()

        if not candidates:
            raise RuntimeError("通达信服务器探测失败且无缓存可用")

        # 逐个尝试，取第一个成功的
        errors = []
        for ip, port in candidates:
            try:
                client = Quotes.factory(market='std', server=(ip, port), timeout=10)
                raw = client.bars(symbol='000001', frequency=9, offset=1)
                if raw is not None and not raw.empty:
                    self._client = client
                    return client
            except Exception as exc:
                errors.append(f"{ip}:{port} {exc}")
                continue

        raise RuntimeError(f"通达信连接失败: {'; '.join(errors[:3])}")

    def fetch_bars(self, code: str, name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """通过 mootdx 获取日线数据。

        mootdx 自动根据 code 前缀识别沪深市场（6→SH, 0/2/3→SZ）。
        """
        try:
            client = self._get_client()
        except ImportError:
            return None, "mootdx 未安装，运行: pip install 'mootdx[all]'"
        except Exception as exc:
            return None, f"TDX 服务器探测失败: {exc}"

        try:
            raw = client.bars(symbol=code, frequency=9, offset=200)
        except Exception as exc:
            self._client = None
            self._server = None
            return None, f"TDX 请求失败: {exc}"

        if raw is None or raw.empty:
            return None, "TDX 返回空数据"

        raw = normalize_tdx_decoded_zeros(raw)
        # Zero-activity placeholders are not effective trading sessions.
        normal = raw[raw["volume"] > 0].copy()
        if normal.empty:
            return None, "TDX 无有效交易日（可能长期停牌）"

        # 标准化为统一的 OHLCV DataFrame
        result = pd.DataFrame({
            "date":     normal["datetime"].astype(str).str[:10],
            "open":     normal["open"].astype(float),
            "high":     normal["high"].astype(float),
            "low":      normal["low"].astype(float),
            "close":    normal["close"].astype(float),
            "volume":   normal["volume"].astype(float),
            "turnover": 0.0,
        })

        # mootdx 返回最新在前，转为升序
        result = result.sort_values("date").reset_index(drop=True)
        return result, None
