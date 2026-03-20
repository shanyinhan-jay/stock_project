import os
import time
import random
from glob import glob
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd
import tushare as ts
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN:
    raise RuntimeError("TUSHARE_TOKEN environment variable is not set")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

app = FastAPI(title="Stock Warehouse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.1.148:8010",
        "http://127.0.0.1:8010",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchDailySyncRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    sync_adj_factor: bool = True


class BatchMinuteSyncRequest(BaseModel):
    symbols: List[str]
    date: str
    freq: Optional[str] = "1min"


class BatchAdjFactorSyncRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str


BASE_DIR = Path("/data")
DAILY_DIR = BASE_DIR / "daily"
MINUTE_DIR = BASE_DIR / "minute"
ADJ_FACTOR_DIR = BASE_DIR / "adj_factor"
META_DIR = BASE_DIR / "meta"

DAILY_DIR.mkdir(parents=True, exist_ok=True)
MINUTE_DIR.mkdir(parents=True, exist_ok=True)
ADJ_FACTOR_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

DUCKDB_FILE = str(META_DIR / "stock.duckdb")
con = duckdb.connect(DUCKDB_FILE)


@app.get("/")
def root():
    return {
        "message": "stock warehouse api is running",
        "endpoints": [
            "/sync/daily/{symbol}",
            "/sync/adj-factor/{symbol}",
            "/sync/minute/{symbol}",
            "/query/daily/{symbol}",
            "/query/minute/{symbol}",
            "/files/daily/{symbol}",
            "/files/adj-factor/{symbol}",
            "/files/minute/{symbol}",
            "/duckdb/daily/{symbol}",
            "/duckdb/minute/{symbol}",
            "/duckdb/scan/daily",
            "/health/a-share-status",
            "/screen/new-high",
            "/query/adj-factor/{symbol}",
            "/screen/three-day-pattern",
            "/batch/sync/daily",
            "/batch/sync/adj-factor",
            "/batch/sync/minute",
        ],
    }


def normalize_date_str(s: str) -> str:
    return pd.to_datetime(s).strftime("%Y-%m-%d")


def save_daily_partitioned(df: pd.DataFrame, symbol: str):
    if df is None or df.empty:
        return []
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["year"] = df["trade_date"].dt.year
    saved_files = []

    for year, group in df.groupby("year"):
        symbol_dir = DAILY_DIR / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        file_path = symbol_dir / f"year={year}.parquet"

        group = group.drop(columns=["year"]).sort_values("trade_date")

        if file_path.exists():
            old_df = pd.read_parquet(file_path)
            old_df["trade_date"] = pd.to_datetime(old_df["trade_date"])
            merged = pd.concat([old_df, group], ignore_index=True)
            merged = merged.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            merged = merged.sort_values("trade_date")
        else:
            merged = group.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            merged = merged.sort_values("trade_date")

        merged.to_parquet(file_path, index=False)
        saved_files.append(str(file_path))

    return saved_files


def save_adj_factor_partitioned(df: pd.DataFrame, symbol: str):
    if df is None or df.empty:
        return []
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["year"] = df["trade_date"].dt.year
    saved_files = []

    for year, group in df.groupby("year"):
        symbol_dir = ADJ_FACTOR_DIR / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        file_path = symbol_dir / f"year={year}.parquet"

        group = group.drop(columns=["year"]).sort_values("trade_date")

        if file_path.exists():
            old_df = pd.read_parquet(file_path)
            old_df["trade_date"] = pd.to_datetime(old_df["trade_date"])
            merged = pd.concat([old_df, group], ignore_index=True)
            merged = merged.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            merged = merged.sort_values("trade_date")
        else:
            merged = group.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
            merged = merged.sort_values("trade_date")

        merged.to_parquet(file_path, index=False)
        saved_files.append(str(file_path))

    return saved_files


def save_minute_partitioned(df: pd.DataFrame, symbol: str, date: str, freq: str):
    if df is None or df.empty:
        return None, 0
    df = df.copy()

    if "trade_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["trade_time"])
        sort_col = "trade_time"
        dedup_cols = ["ts_code", "trade_time"] if "ts_code" in df.columns else ["trade_time"]
    else:
        sort_col = df.columns[0]
        dedup_cols = [sort_col]

    interval_dir = MINUTE_DIR / f"interval={freq}" / f"symbol={symbol}"
    interval_dir.mkdir(parents=True, exist_ok=True)
    file_path = interval_dir / f"date={date}.parquet"

    if file_path.exists():
        old_df = pd.read_parquet(file_path)
        merged = pd.concat([old_df, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
        merged = merged.sort_values(sort_col)
    else:
        merged = df.drop_duplicates(subset=dedup_cols, keep="last")
        merged = merged.sort_values(sort_col)

    merged.to_parquet(file_path, index=False)
    return str(file_path), len(merged)


def load_daily_raw_local(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbol_dir = DAILY_DIR / f"symbol={symbol}"
    files = sorted(glob(str(symbol_dir / "year=*.parquet")))
    if not files:
        raise HTTPException(status_code=404, detail="no local daily files found")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
    return df.sort_values("trade_date")


def load_adj_factor_local(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbol_dir = ADJ_FACTOR_DIR / f"symbol={symbol}"
    files = sorted(glob(str(symbol_dir / "year=*.parquet")))
    if not files:
        raise HTTPException(status_code=404, detail="no local adj_factor files found")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
    return df.sort_values("trade_date")


def load_minute_local(symbol: str, date: str, freq: str) -> pd.DataFrame:
    file_path = MINUTE_DIR / f"interval={freq}" / f"symbol={symbol}" / f"date={date}.parquet"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="no local minute file found")

    df = pd.read_parquet(file_path)
    if "trade_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["trade_time"])
        df = df.sort_values("trade_time")
    return df


def apply_adjustment(daily_df: pd.DataFrame, adj_df: pd.DataFrame, adj: str) -> pd.DataFrame:
    df = daily_df.copy()
    af = adj_df.copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    af["trade_date"] = pd.to_datetime(af["trade_date"])

    merged = pd.merge(df, af[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    merged = merged.sort_values("trade_date")

    if merged["adj_factor"].isna().all():
        raise HTTPException(status_code=404, detail="adj_factor data not found for requested range")

    price_cols = [c for c in ["open", "high", "low", "close", "pre_close"] if c in merged.columns]

    if adj == "qfq":
        latest_factor = merged["adj_factor"].dropna().iloc[-1]
        for col in price_cols:
            merged[col] = merged[col] * merged["adj_factor"] / latest_factor
    elif adj == "hfq":
        for col in price_cols:
            merged[col] = merged[col] * merged["adj_factor"]
    elif adj != "none":
        raise HTTPException(status_code=400, detail="adj must be one of: none, qfq, hfq")

    return merged


def query_daily_duckdb(symbol: str, start_date: str, end_date: str):
    pattern = f"/data/daily/symbol={symbol}/year=*.parquet"
    sql = """
    SELECT *
    FROM read_parquet(?)
    WHERE CAST(trade_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
    ORDER BY trade_date
    """
    return con.execute(sql, [pattern, normalize_date_str(start_date), normalize_date_str(end_date)]).fetchdf()


def query_adj_factor_duckdb(symbol: str, start_date: str, end_date: str):
    pattern = f"/data/adj_factor/symbol={symbol}/year=*.parquet"
    sql = """
    SELECT *
    FROM read_parquet(?)
    WHERE CAST(trade_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
    ORDER BY trade_date
    """
    return con.execute(sql, [pattern, normalize_date_str(start_date), normalize_date_str(end_date)]).fetchdf()


def query_minute_duckdb(symbol: str, date: str, freq: str = "1min"):
    pattern = f"/data/minute/interval={freq}/symbol={symbol}/date={date}.parquet"
    sql = """
    SELECT *
    FROM read_parquet(?)
    ORDER BY trade_time
    """
    return con.execute(sql, [pattern]).fetchdf()


def scan_daily_by_date_duckdb(trade_date: str):
    pattern = "/data/daily/symbol=*/year=*.parquet"
    trade_date_fmt = pd.to_datetime(trade_date, format="%Y%m%d").strftime("%Y-%m-%d")
    sql = """
    SELECT *
    FROM read_parquet(?)
    WHERE CAST(trade_date AS DATE) = CAST(? AS DATE)
    ORDER BY ts_code
    """
    return con.execute(sql, [pattern, trade_date_fmt]).fetchdf()


def load_name_mapping():
    candidates = [
        "/data/stock_basic_min.csv",
        "/data/meta/stock_basic_min.csv",
        "/data/stock_basic.csv",
        "/data/meta/stock_basic.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "ts_code" in df.columns and "name" in df.columns:
                    return df[["ts_code", "name"]].drop_duplicates(subset=["ts_code"])
                if "代码" in df.columns and "名称" in df.columns:
                    tmp = df.copy()
                    tmp["代码"] = tmp["代码"].astype(str).str.zfill(6)

                    def to_ts_code(code: str) -> Optional[str]:
                        if code.startswith(("600", "601", "603", "605", "688")):
                            return f"{code}.SH"
                        if code.startswith(("000", "001", "002", "003", "300", "301")):
                            return f"{code}.SZ"
                        return None

                    tmp["ts_code"] = tmp["代码"].apply(to_ts_code)
                    tmp["name"] = tmp["名称"]
                    return tmp[["ts_code", "name"]].dropna().drop_duplicates(subset=["ts_code"])
            except Exception:
                pass
    return None


def screen_new_high(
    trade_date: str,
    trading_days: int = 250,
    recent_days: int = 5,
    adj: str = "qfq",
    price_field: str = "high",
    include_name: bool = True,
    upper_shadow_filter: str = "any",
):
    if adj not in ("qfq", "hfq", "none"):
        raise HTTPException(status_code=400, detail="adj must be one of: none, qfq, hfq")
    if price_field not in ("high", "close"):
        raise HTTPException(status_code=400, detail="price_field must be one of: high, close")
    if trading_days < 20 or trading_days > 1000:
        raise HTTPException(status_code=400, detail="trading_days must be between 20 and 1000")
    if recent_days < 1 or recent_days > 30:
        raise HTTPException(status_code=400, detail="recent_days must be between 1 and 30")
    if upper_shadow_filter not in ("any", "yes", "no"):
        raise HTTPException(status_code=400, detail="upper_shadow_filter must be one of: any, yes, no")

    target_dt = pd.to_datetime(trade_date, format="%Y%m%d")
    start_dt = target_dt - pd.Timedelta(days=max(400, trading_days * 2))
    start_date_sql = start_dt.strftime("%Y-%m-%d")
    target_date_sql = target_dt.strftime("%Y-%m-%d")

    daily_sql = """
    SELECT *
    FROM read_parquet('/data/daily/symbol=*/year=*.parquet')
    WHERE CAST(trade_date AS DATE)
          BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
    """
    daily = con.execute(daily_sql, [start_date_sql, target_date_sql]).fetchdf()
    if daily is None or daily.empty:
        raise HTTPException(status_code=404, detail="no daily data found in local warehouse")

    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    for col in ["open", "high", "low", "close", "pre_close"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    df = daily
    if adj != "none":
        adj_sql = """
        SELECT ts_code, trade_date, adj_factor
        FROM read_parquet('/data/adj_factor/symbol=*/year=*.parquet')
        WHERE CAST(trade_date AS DATE)
              BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        """
        adj_df = con.execute(adj_sql, [start_date_sql, target_date_sql]).fetchdf()
        if adj_df is None or adj_df.empty:
            raise HTTPException(status_code=404, detail="no adj_factor data found in local warehouse")
        adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
        adj_df["adj_factor"] = pd.to_numeric(adj_df["adj_factor"], errors="coerce")
        df = daily.merge(adj_df, on=["ts_code", "trade_date"], how="left")
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        if adj == "qfq":
            latest_factor = df.groupby("ts_code")["adj_factor"].transform("last")
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df.columns:
                    df[col] = df[col] * df["adj_factor"] / latest_factor
        elif adj == "hfq":
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df.columns:
                    df[col] = df[col] * df["adj_factor"]

    results = []
    for ts_code, g in df.groupby("ts_code"):
        g = g.sort_values("trade_date").reset_index(drop=True)
        if len(g) < trading_days:
            continue

        g = g.tail(trading_days).reset_index(drop=True)
        recent = g.tail(recent_days).copy()

        recent["upper_shadow"] = recent["high"] - recent[["open", "close"]].max(axis=1)
        recent["body"] = (recent["close"] - recent["open"]).abs()
        recent["is_long_upper_shadow"] = recent["upper_shadow"] >= (recent["body"] * 1.5)

        if upper_shadow_filter == "yes":
            recent = recent[recent["is_long_upper_shadow"]]
        elif upper_shadow_filter == "no":
            recent = recent[~recent["is_long_upper_shadow"]]

        max_high = g["high"].max()
        max_close = g["close"].max()

        hit_high_rows = recent[recent["high"] >= max_high]
        hit_close_rows = recent[recent["close"] >= max_close]

        hit_high = not hit_high_rows.empty
        hit_close = not hit_close_rows.empty

        target_rows = hit_high_rows if price_field == "high" else hit_close_rows
        target_hit = not target_rows.empty

        if target_hit:
            hit_dates = pd.concat([
                hit_high_rows[["trade_date"]],
                hit_close_rows[["trade_date"]],
            ]).drop_duplicates().sort_values("trade_date")

            results.append({
                "ts_code": ts_code,
                "last_trade_date": g.iloc[-1]["trade_date"].strftime("%Y-%m-%d"),
                "trading_days": trading_days,
                "recent_days": recent_days,
                "price_field": price_field,
                "hit_high": bool(hit_high),
                "hit_close": bool(hit_close),
                "high_break_dates": ",".join(hit_high_rows["trade_date"].dt.strftime("%Y-%m-%d").tolist()),
                "close_break_dates": ",".join(hit_close_rows["trade_date"].dt.strftime("%Y-%m-%d").tolist()),
                "hit_dates": ",".join(hit_dates["trade_date"].dt.strftime("%Y-%m-%d").tolist()),
                "latest_close": g.iloc[-1]["close"],
                "latest_high": g.iloc[-1]["high"],
                "latest_upper_shadow": g.iloc[-1]["high"] - max(g.iloc[-1]["open"], g.iloc[-1]["close"]),
                "latest_body": abs(g.iloc[-1]["close"] - g.iloc[-1]["open"]),
                "window_high_max": max_high,
                "window_close_max": max_close,
            })

    result_df = pd.DataFrame(results).sort_values("ts_code") if results else pd.DataFrame(columns=[
        "ts_code", "last_trade_date", "trading_days", "recent_days", "price_field",
        "hit_high", "hit_close", "high_break_dates", "close_break_dates", "hit_dates",
        "latest_close", "latest_high", "latest_upper_shadow", "latest_body", "window_high_max", "window_close_max"
    ])

    if include_name and not result_df.empty:
        name_df = load_name_mapping()
        if name_df is not None:
            result_df = result_df.merge(name_df, on="ts_code", how="left")

    preferred_cols = [
        "ts_code", "name", "last_trade_date", "trading_days", "recent_days", "price_field",
        "hit_high", "hit_close", "high_break_dates", "close_break_dates", "hit_dates",
        "latest_close", "latest_high", "latest_upper_shadow", "latest_body", "window_high_max", "window_close_max"
    ]
    result_df = result_df[[c for c in preferred_cols if c in result_df.columns]]

    return result_df


def screen_three_day_pattern(
    trade_date: str,
    adj: str = "qfq",
    include_name: bool = True,
):
    if adj not in ("qfq", "hfq", "none"):
        raise HTTPException(status_code=400, detail="adj must be one of: none, qfq, hfq")

    target_dt = pd.to_datetime(trade_date, format="%Y%m%d")
    start_dt = target_dt - pd.Timedelta(days=30)
    start_date_sql = start_dt.strftime("%Y-%m-%d")
    target_date_sql = target_dt.strftime("%Y-%m-%d")

    daily_sql = """
    SELECT *
    FROM read_parquet('/data/daily/symbol=*/year=*.parquet')
    WHERE CAST(trade_date AS DATE)
          BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
    """
    daily = con.execute(daily_sql, [start_date_sql, target_date_sql]).fetchdf()
    if daily is None or daily.empty:
        raise HTTPException(status_code=404, detail="no daily data found in local warehouse")

    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    for col in ["open", "high", "low", "close", "pre_close", "vol"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    df = daily
    if adj != "none":
        adj_sql = """
        SELECT ts_code, trade_date, adj_factor
        FROM read_parquet('/data/adj_factor/symbol=*/year=*.parquet')
        WHERE CAST(trade_date AS DATE)
              BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        """
        adj_df = con.execute(adj_sql, [start_date_sql, target_date_sql]).fetchdf()
        if adj_df is None or adj_df.empty:
            raise HTTPException(status_code=404, detail="no adj_factor data found in local warehouse")

        adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
        adj_df["adj_factor"] = pd.to_numeric(adj_df["adj_factor"], errors="coerce")

        df = daily.merge(adj_df, on=["ts_code", "trade_date"], how="left")
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        if adj == "qfq":
            latest_factor = df.groupby("ts_code")["adj_factor"].transform("last")
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df.columns:
                    df[col] = df[col] * df["adj_factor"] / latest_factor
        elif adj == "hfq":
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df.columns:
                    df[col] = df[col] * df["adj_factor"]

    results = []
    for ts_code, g in df.groupby("ts_code"):
        g = g.sort_values("trade_date").reset_index(drop=True)

        if len(g) < 4:
            continue

        g = g.tail(4).reset_index(drop=True)
        d0 = g.iloc[0]
        d1 = g.iloc[1]
        d2 = g.iloc[2]
        d3 = g.iloc[3]

        if d3["trade_date"].strftime("%Y%m%d") != trade_date:
            continue

        cond1 = (
            pd.notna(d1["open"]) and pd.notna(d1["close"]) and
            pd.notna(d1["vol"]) and pd.notna(d0["vol"]) and
            d1["close"] < d1["open"] and
            d1["vol"] > d0["vol"]
        )

        cond2 = (
            pd.notna(d2["open"]) and pd.notna(d2["close"]) and
            pd.notna(d2["vol"]) and pd.notna(d1["close"]) and pd.notna(d1["vol"]) and
            d2["open"] < d1["close"] and
            d2["close"] > d2["open"] and
            d2["vol"] < d1["vol"]
        )

        cond3 = (
            pd.notna(d3["open"]) and pd.notna(d3["close"]) and pd.notna(d2["close"]) and
            d3["open"] > d2["close"] and
            d3["close"] > d3["open"]
        )

        if cond1 and cond2 and cond3:
            results.append({
                "ts_code": ts_code,
                "pattern_trade_date": d3["trade_date"].strftime("%Y-%m-%d"),
                "d1_date": d1["trade_date"].strftime("%Y-%m-%d"),
                "d2_date": d2["trade_date"].strftime("%Y-%m-%d"),
                "d3_date": d3["trade_date"].strftime("%Y-%m-%d"),
                "d1_open": d1["open"],
                "d1_close": d1["close"],
                "d1_vol": d1["vol"],
                "d0_vol": d0["vol"],
                "d2_open": d2["open"],
                "d2_close": d2["close"],
                "d2_vol": d2["vol"],
                "d3_open": d3["open"],
                "d3_close": d3["close"],
            })

    result_df = pd.DataFrame(results).sort_values("ts_code") if results else pd.DataFrame(columns=[
        "ts_code", "pattern_trade_date", "d1_date", "d2_date", "d3_date",
        "d1_open", "d1_close", "d1_vol", "d0_vol",
        "d2_open", "d2_close", "d2_vol",
        "d3_open", "d3_close",
    ])

    if include_name and not result_df.empty:
        name_df = load_name_mapping()
        if name_df is not None:
            result_df = result_df.merge(name_df, on="ts_code", how="left")

    preferred_cols = [
        "ts_code", "name", "pattern_trade_date",
        "d1_date", "d2_date", "d3_date",
        "d1_open", "d1_close", "d1_vol", "d0_vol",
        "d2_open", "d2_close", "d2_vol",
        "d3_open", "d3_close",
    ]
    result_df = result_df[[c for c in preferred_cols if c in result_df.columns]]

    return result_df


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10) + wait_random(1, 3), reraise=True)
def sync_stock_data(symbol: str, start_date: str, end_date: str):
    df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="no daily data found from tushare")
    return df


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10) + wait_random(1, 3), reraise=True)
def sync_adj_factor_data(symbol: str, start_date: str, end_date: str):
    df = pro.adj_factor(ts_code=symbol, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="no adj_factor data found from tushare")
    return df


@app.get("/sync/daily/{symbol}")
def sync_daily(symbol: str, start_date: str, end_date: str, sync_adj_factor: bool = True):
    daily_df = None
    adj_df = None
    try:
        daily_df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        if daily_df is None or daily_df.empty:
            raise HTTPException(status_code=404, detail="no daily data found from tushare")

        daily_saved_files = save_daily_partitioned(daily_df, symbol)
        adj_saved_files = []
        adj_status = "skipped"

        if sync_adj_factor:
            try:
                adj_df = pro.adj_factor(ts_code=symbol, start_date=start_date, end_date=end_date)
                if adj_df is not None and not adj_df.empty:
                    adj_saved_files = save_adj_factor_partitioned(adj_df, symbol)
                    adj_status = "ok"
                else:
                    adj_status = "no_data"
            except Exception as e:
                adj_status = f"error: {str(e)}"

        return {
            "message": "daily data synced",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "daily": {
                "status": "ok",
                "rows": len(daily_df),
                "saved_files": daily_saved_files,
                "columns": list(daily_df.columns),
            },
            "adj_factor": {
                "enabled": sync_adj_factor,
                "status": adj_status,
                "rows": 0 if adj_df is None else len(adj_df),
                "saved_files": adj_saved_files,
                "columns": [] if adj_df is None else list(adj_df.columns),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sync/adj-factor/{symbol}")
def sync_adj_factor(symbol: str, start_date: str, end_date: str):
    try:
        df = pro.adj_factor(ts_code=symbol, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="no adj_factor data found from tushare")
        saved_files = save_adj_factor_partitioned(df, symbol)
        return {
            "message": "adj_factor data synced",
            "symbol": symbol,
            "rows": len(df),
            "saved_files": saved_files,
            "columns": list(df.columns),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sync/minute/{symbol}")
def sync_minute(symbol: str, date: str, freq: str = "1min"):
    try:
        start_time = f"{date} 09:30:00"
        end_time = f"{date} 15:00:00"
        df = pro.stk_mins(ts_code=symbol, start_time=start_time, end_time=end_time, freq=freq)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="no minute data found from tushare")
        file_path, rows = save_minute_partitioned(df, symbol, date, freq)
        return {
            "message": "minute data synced",
            "symbol": symbol,
            "date": date,
            "freq": freq,
            "rows": rows,
            "file": file_path,
            "columns": list(df.columns),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/daily/{symbol}")
def query_daily(symbol: str, start_date: str, end_date: str, adj: str = "none"):
    try:
        daily_df = load_daily_raw_local(symbol, start_date, end_date)
        if daily_df.empty:
            raise HTTPException(status_code=404, detail="no local daily data in range")
        result_df = daily_df if adj == "none" else apply_adjustment(daily_df, load_adj_factor_local(symbol, start_date, end_date), adj)
        return {
            "message": "local daily data loaded",
            "symbol": symbol,
            "adj": adj,
            "rows": len(result_df),
            "columns": list(result_df.columns),
            "data": result_df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/minute/{symbol}")
def query_minute(symbol: str, date: str, freq: str = "1min"):
    try:
        df = load_minute_local(symbol, date, freq)
        if df.empty:
            raise HTTPException(status_code=404, detail="no local minute data")
        return {
            "message": "local minute data loaded",
            "symbol": symbol,
            "date": date,
            "freq": freq,
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/daily/{symbol}")
def list_daily_files(symbol: str):
    symbol_dir = DAILY_DIR / f"symbol={symbol}"
    files = sorted(glob(str(symbol_dir / "year=*.parquet")))
    if not files:
        raise HTTPException(status_code=404, detail="no daily files found")
    return {"symbol": symbol, "count": len(files), "files": files}


@app.get("/files/adj-factor/{symbol}")
def list_adj_factor_files(symbol: str):
    symbol_dir = ADJ_FACTOR_DIR / f"symbol={symbol}"
    files = sorted(glob(str(symbol_dir / "year=*.parquet")))
    if not files:
        raise HTTPException(status_code=404, detail="no adj_factor files found")
    return {"symbol": symbol, "count": len(files), "files": files}


@app.get("/files/minute/{symbol}")
def list_minute_files(symbol: str, freq: str = "1min"):
    symbol_dir = MINUTE_DIR / f"interval={freq}" / f"symbol={symbol}"
    files = sorted(glob(str(symbol_dir / "date=*.parquet")))
    if not files:
        raise HTTPException(status_code=404, detail="no minute files found")
    return {"symbol": symbol, "freq": freq, "count": len(files), "files": files}


@app.get("/duckdb/daily/{symbol}")
def duckdb_daily(symbol: str, start_date: str, end_date: str, adj: str = "none"):
    try:
        daily_df = query_daily_duckdb(symbol, start_date, end_date)
        if daily_df is None or daily_df.empty:
            raise HTTPException(status_code=404, detail="no duckdb daily data found")
        result_df = daily_df if adj == "none" else apply_adjustment(daily_df, query_adj_factor_duckdb(symbol, start_date, end_date), adj)
        return {
            "message": "duckdb daily query ok",
            "symbol": symbol,
            "adj": adj,
            "rows": len(result_df),
            "columns": list(result_df.columns),
            "data": result_df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/duckdb/minute/{symbol}")
def duckdb_minute(symbol: str, date: str, freq: str = "1min"):
    try:
        df = query_minute_duckdb(symbol, date, freq)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="no duckdb minute data found")
        return {
            "message": "duckdb minute query ok",
            "symbol": symbol,
            "date": date,
            "freq": freq,
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/duckdb/scan/daily")
def duckdb_scan_daily(trade_date: str):
    try:
        df = scan_daily_by_date_duckdb(trade_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="no duckdb scan result")
        return {
            "message": "duckdb daily scan ok",
            "trade_date": trade_date,
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/a-share-status")
def a_share_status(trade_date: Optional[str] = None):
    try:
        info = {
            "api_ok": True,
            "message": "stock api ok",
            "trade_date": trade_date,
        }
        if trade_date:
            try:
                df = scan_daily_by_date_duckdb(trade_date)
                info["daily_scan_rows"] = 0 if df is None else len(df)
            except Exception as e:
                info["daily_scan_error"] = str(e)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/screen/new-high")
def api_screen_new_high(
    trade_date: str,
    trading_days: int = Query(250, description="回看交易日数量，例如 250 或 120"),
    recent_days: int = Query(5, description="最近多少个交易日内触发"),
    adj: str = Query("qfq", description="复权方式: qfq/hfq/none"),
    price_field: str = Query("high", description="新高口径: high/close"),
    include_name: bool = Query(True, description="是否合并股票名称"),
    upper_shadow_filter: str = Query("any", description="长上影线筛选: any/yes/no")
):
    result_df = screen_new_high(
        trade_date=trade_date,
        trading_days=trading_days,
        recent_days=recent_days,
        adj=adj,
        price_field=price_field,
        include_name=include_name,
        upper_shadow_filter=upper_shadow_filter,
    )
    return {
        "message": "screen new high ok",
        "trade_date": trade_date,
        "trading_days": trading_days,
        "recent_days": recent_days,
        "adj": adj,
        "price_field": price_field,
        "upper_shadow_filter": upper_shadow_filter,
        "rows": len(result_df),
        "columns": list(result_df.columns),
        "data": result_df.to_dict(orient="records"),
    }


@app.get("/screen/three-day-pattern")
def api_screen_three_day_pattern(
    trade_date: str,
    adj: str = Query("qfq", description="复权方式: qfq/hfq/none"),
    include_name: bool = Query(True, description="是否合并股票名称"),
):
    result_df = screen_three_day_pattern(
        trade_date=trade_date,
        adj=adj,
        include_name=include_name,
    )
    return {
        "message": "screen three day pattern ok",
        "trade_date": trade_date,
        "adj": adj,
        "rows": len(result_df),
        "columns": list(result_df.columns),
        "data": result_df.to_dict(orient="records"),
    }


@app.post("/batch/sync/daily")
def batch_sync_daily(req: BatchDailySyncRequest, max_batch_size: int = 50):
    results = []
    total_symbols = len(req.symbols)

    for i in range(0, total_symbols, max_batch_size):
        for symbol in req.symbols[i:i + max_batch_size]:
            try:
                time.sleep(random.uniform(0.5, 2))
                daily_df = sync_stock_data(symbol, req.start_date, req.end_date)
                daily_saved_files = save_daily_partitioned(daily_df, symbol)
                item = {
                    "symbol": symbol,
                    "status": "ok",
                    "daily_rows": len(daily_df),
                    "daily_saved_files": daily_saved_files,
                }
                if req.sync_adj_factor:
                    try:
                        adj_df = sync_adj_factor_data(symbol, req.start_date, req.end_date)
                        item["adj_factor"] = {
                            "status": "ok",
                            "rows": len(adj_df),
                            "saved_files": save_adj_factor_partitioned(adj_df, symbol),
                        }
                    except HTTPException as e:
                        item["adj_factor"] = {"status": "no_data" if e.status_code == 404 else "error", "error": e.detail}
                    except Exception as e:
                        item["adj_factor"] = {"status": "error", "error": str(e)}
                results.append(item)
            except HTTPException as e:
                results.append({"symbol": symbol, "status": "no_data" if e.status_code == 404 else "error", "error": e.detail})
            except Exception as e:
                results.append({"symbol": symbol, "status": "error", "error": str(e)})

    return {
        "message": "batch daily sync finished",
        "total": total_symbols,
        "success": sum(1 for r in results if r["status"] == "ok"),
        "no_data": sum(1 for r in results if r["status"] == "no_data"),
        "error": sum(1 for r in results if r["status"] == "error"),
        "sync_adj_factor": req.sync_adj_factor,
        "results": results,
    }


@app.post("/batch/sync/adj-factor")
def batch_sync_adj_factor(req: BatchAdjFactorSyncRequest, max_batch_size: int = 50):
    results = []
    total_symbols = len(req.symbols)

    for i in range(0, total_symbols, max_batch_size):
        for symbol in req.symbols[i:i + max_batch_size]:
            try:
                time.sleep(random.uniform(0.5, 2))
                df = sync_adj_factor_data(symbol, req.start_date, req.end_date)
                results.append({
                    "symbol": symbol,
                    "status": "ok",
                    "rows": len(df),
                    "saved_files": save_adj_factor_partitioned(df, symbol),
                })
            except HTTPException as e:
                results.append({"symbol": symbol, "status": "no_data" if e.status_code == 404 else "error", "error": e.detail})
            except Exception as e:
                results.append({"symbol": symbol, "status": "error", "error": str(e)})

    return {
        "message": "batch adj_factor sync finished",
        "total": total_symbols,
        "success": sum(1 for r in results if r["status"] == "ok"),
        "no_data": sum(1 for r in results if r["status"] == "no_data"),
        "error": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }


@app.post("/batch/sync/minute")
def batch_sync_minute(req: BatchMinuteSyncRequest):
    results = []
    for symbol in req.symbols:
        try:
            time.sleep(random.uniform(0.5, 2))
            start_time = f"{req.date} 09:30:00"
            end_time = f"{req.date} 15:00:00"
            df = pro.stk_mins(ts_code=symbol, start_time=start_time, end_time=end_time, freq=req.freq)

            if df is None or df.empty:
                results.append({"symbol": symbol, "status": "no_data", "rows": 0, "file": None})
                continue

            file_path, rows = save_minute_partitioned(df, symbol, req.date, req.freq)
            results.append({"symbol": symbol, "status": "ok", "rows": rows, "file": file_path})
        except Exception as e:
            results.append({"symbol": symbol, "status": "error", "error": str(e)})

    return {
        "message": "batch minute sync finished",
        "date": req.date,
        "freq": req.freq,
        "total": len(req.symbols),
        "success": sum(1 for r in results if r["status"] == "ok"),
        "no_data": sum(1 for r in results if r["status"] == "no_data"),
        "error": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }



@app.get("/query/adj-factor/{symbol}")
def api_query_adj_factor(
    symbol: str,
    start_date: str = Query(..., description="开始日期，格式 YYYYMMDD"),
    end_date: str = Query(..., description="结束日期，格式 YYYYMMDD"),
):
    try:
        df = query_adj_factor_duckdb(symbol, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query adj_factor failed: {e}")

    if df is None or df.empty:
        return {
            "message": "adj_factor query ok",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "rows": 0,
            "columns": [],
            "data": [],
        }

    if "trade_date" in df.columns:
        try:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    if "adj_factor" in df.columns:
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")

    preferred_cols = [c for c in ["ts_code", "trade_date", "adj_factor"] if c in df.columns]
    if preferred_cols:
        other_cols = [c for c in df.columns if c not in preferred_cols]
        df = df[preferred_cols + other_cols]

    return {
        "message": "adj_factor query ok",
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "rows": len(df),
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
    }


def screen_five_day_bullish_volume(
    trade_date: str,
    adj: str = "qfq",
    include_name: bool = True,
):
    if adj not in ("qfq", "hfq", "none"):
        raise HTTPException(status_code=400, detail="adj must be one of: none, qfq, hfq")

    target_dt = pd.to_datetime(trade_date, format="%Y%m%d")
    start_dt = target_dt - pd.Timedelta(days=30)
    start_date_sql = start_dt.strftime("%Y-%m-%d")
    target_date_sql = target_dt.strftime("%Y-%m-%d")

    daily_sql = """
    SELECT *
    FROM read_parquet('/data/daily/symbol=*/year=*.parquet')
    WHERE CAST(trade_date AS DATE)
          BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
    """
    daily = con.execute(daily_sql, [start_date_sql, target_date_sql]).fetchdf()
    if daily is None or daily.empty:
        raise HTTPException(status_code=404, detail="no daily data found in local warehouse")

    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    for col in ["open", "high", "low", "close", "pre_close", "vol"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    df = daily
    if adj != "none":
        adj_sql = """
        SELECT ts_code, trade_date, adj_factor
        FROM read_parquet('/data/adj_factor/symbol=*/year=*.parquet')
        WHERE CAST(trade_date AS DATE)
              BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        """
        adj_df = con.execute(adj_sql, [start_date_sql, target_date_sql]).fetchdf()
        if adj_df is None or adj_df.empty:
            raise HTTPException(status_code=404, detail="no adj_factor data found in local warehouse")

        adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
        adj_df["adj_factor"] = pd.to_numeric(adj_df["adj_factor"], errors="coerce")

        df = daily.merge(adj_df, on=["ts_code", "trade_date"], how="left")
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        if adj == "qfq":
            latest_factor = df.groupby("ts_code")["adj_factor"].transform("last")
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df.columns:
                    df[col] = df[col] * df["adj_factor"] / latest_factor
        elif adj == "hfq":
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df.columns:
                    df[col] = df[col] * df["adj_factor"]

    results = []
    for ts_code, g in df.groupby("ts_code"):
        g = g.sort_values("trade_date").reset_index(drop=True)
        if len(g) < 5:
            continue

        g = g.tail(5).reset_index(drop=True)
        d5 = g.iloc[-1]
        if d5["trade_date"].strftime("%Y%m%d") != trade_date:
            continue

        bullish_ok = bool((g["close"] > g["open"]).all())
        vol_ok = bool((g["vol"].diff().iloc[1:] > 0).all())

        if bullish_ok and vol_ok:
            results.append({
                "ts_code": ts_code,
                "pattern_trade_date": d5["trade_date"].strftime("%Y-%m-%d"),
                "start_date": g.iloc[0]["trade_date"].strftime("%Y-%m-%d"),
                "end_date": g.iloc[-1]["trade_date"].strftime("%Y-%m-%d"),
                "day1_vol": g.iloc[0]["vol"],
                "day2_vol": g.iloc[1]["vol"],
                "day3_vol": g.iloc[2]["vol"],
                "day4_vol": g.iloc[3]["vol"],
                "day5_vol": g.iloc[4]["vol"],
                "day1_close": g.iloc[0]["close"],
                "day5_close": g.iloc[4]["close"],
            })

    result_df = pd.DataFrame(results).sort_values("ts_code") if results else pd.DataFrame(columns=[
        "ts_code", "pattern_trade_date", "start_date", "end_date",
        "day1_vol", "day2_vol", "day3_vol", "day4_vol", "day5_vol",
        "day1_close", "day5_close",
    ])

    if include_name and not result_df.empty:
        name_df = load_name_mapping()
        if name_df is not None:
            result_df = result_df.merge(name_df, on="ts_code", how="left")

    preferred_cols = [
        "ts_code", "name", "pattern_trade_date", "start_date", "end_date",
        "day1_vol", "day2_vol", "day3_vol", "day4_vol", "day5_vol",
        "day1_close", "day5_close",
    ]
    result_df = result_df[[c for c in preferred_cols if c in result_df.columns]]
    return result_df


@app.get("/screen/five-day-bullish-volume")
def api_screen_five_day_bullish_volume(
    trade_date: str,
    adj: str = Query("qfq", description="复权方式: qfq/hfq/none"),
    include_name: bool = Query(True, description="是否合并股票名称"),
):
    result_df = screen_five_day_bullish_volume(
        trade_date=trade_date,
        adj=adj,
        include_name=include_name,
    )
    return {
        "message": "screen five day bullish volume ok",
        "trade_date": trade_date,
        "adj": adj,
        "rows": len(result_df),
        "columns": list(result_df.columns),
        "data": result_df.to_dict(orient="records"),
    }


def screen_bullish_engulfing_volume(
    trade_date: str,
    adj: str = "qfq",
    include_name: bool = True,
    volume_mode: str = "any",
    min_body_pct: float = 0.01,
):
    if adj not in ("qfq", "hfq", "none"):
        raise HTTPException(status_code=400, detail="adj must be one of: none, qfq, hfq")
    if volume_mode not in ("any", "shrink", "expand"):
        raise HTTPException(status_code=400, detail="volume_mode must be one of: any, shrink, expand")
    if min_body_pct < 0 or min_body_pct > 0.2:
        raise HTTPException(status_code=400, detail="min_body_pct must be between 0 and 0.2")

    target_dt = pd.to_datetime(trade_date, format="%Y%m%d")
    start_dt = target_dt - pd.Timedelta(days=30)
    start_date_sql = start_dt.strftime("%Y-%m-%d")
    target_date_sql = target_dt.strftime("%Y-%m-%d")

    daily_sql = """
    SELECT *
    FROM read_parquet('/data/daily/symbol=*/year=*.parquet')
    WHERE CAST(trade_date AS DATE)
          BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
    """
    daily = con.execute(daily_sql, [start_date_sql, target_date_sql]).fetchdf()
    if daily is None or daily.empty:
        raise HTTPException(status_code=404, detail="no daily data found in local warehouse")

    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    for col in ["open", "high", "low", "close", "pre_close", "vol"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    df = daily
    if adj != "none":
        adj_sql = """
        SELECT ts_code, trade_date, adj_factor
        FROM read_parquet('/data/adj_factor/symbol=*/year=*.parquet')
        WHERE CAST(trade_date AS DATE)
              BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        """
        adj_df = con.execute(adj_sql, [start_date_sql, target_date_sql]).fetchdf()
        if adj_df is None or adj_df.empty:
            raise HTTPException(status_code=404, detail="no adj_factor data found in local warehouse")

        adj_df["trade_date"] = pd.to_datetime(adj_df["trade_date"])
        adj_df["adj_factor"] = pd.to_numeric(adj_df["adj_factor"], errors="coerce")

        df = daily.merge(adj_df, on=["ts_code", "trade_date"], how="left")
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        if adj == "qfq":
            latest_factor = df.groupby("ts_code")["adj_factor"].transform("last")
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df.columns:
                    df[col] = df[col] * df["adj_factor"] / latest_factor
        elif adj == "hfq":
            for col in ["open", "high", "low", "close", "pre_close"]:
                if col in df.columns:
                    df[col] = df[col] * df["adj_factor"]

    results = []

    for ts_code, g in df.groupby("ts_code"):
        g = g.sort_values("trade_date").reset_index(drop=True)
        if len(g) < 3:
            continue

        g = g.tail(3).reset_index(drop=True)
        d0 = g.iloc[0]
        d1 = g.iloc[1]
        d2 = g.iloc[2]

        if d2["trade_date"].strftime("%Y%m%d") != trade_date:
            continue
        if pd.isna(d1["pre_close"]) or d1["pre_close"] == 0 or pd.isna(d2["pre_close"]) or d2["pre_close"] == 0:
            continue

        d1_body_pct = abs(d1["close"] - d1["open"]) / abs(d1["pre_close"])
        d2_body_pct = abs(d2["close"] - d2["open"]) / abs(d2["pre_close"])

        cond_prev_bear = (
            pd.notna(d1["open"]) and pd.notna(d1["close"]) and pd.notna(d1["vol"]) and pd.notna(d0["vol"])
            and d1["close"] < d1["open"]
            and d1["vol"] > d0["vol"]
            and d1_body_pct >= min_body_pct
        )
        cond_curr_engulf = (
            pd.notna(d2["open"]) and pd.notna(d2["close"]) and pd.notna(d2["vol"])
            and d2["close"] > d2["open"]
            and d2["open"] <= d1["close"]
            and d2["close"] >= d1["open"]
            and d2_body_pct >= min_body_pct
        )

        cond_volume = True
        if volume_mode == "shrink":
            cond_volume = pd.notna(d2["vol"]) and pd.notna(d1["vol"]) and d2["vol"] < d1["vol"]
        elif volume_mode == "expand":
            cond_volume = pd.notna(d2["vol"]) and pd.notna(d1["vol"]) and d2["vol"] > d1["vol"]

        if cond_prev_bear and cond_curr_engulf and cond_volume:
            results.append({
                "ts_code": ts_code,
                "pattern_trade_date": d2["trade_date"].strftime("%Y-%m-%d"),
                "prev_date": d1["trade_date"].strftime("%Y-%m-%d"),
                "curr_date": d2["trade_date"].strftime("%Y-%m-%d"),
                "volume_mode": volume_mode,
                "min_body_pct": min_body_pct,
                "prev_open": d1["open"],
                "prev_close": d1["close"],
                "prev_vol": d1["vol"],
                "prev_prev_vol": d0["vol"],
                "prev_body_pct": d1_body_pct,
                "curr_open": d2["open"],
                "curr_close": d2["close"],
                "curr_vol": d2["vol"],
                "curr_body_pct": d2_body_pct,
            })

    result_df = pd.DataFrame(results).sort_values("ts_code") if results else pd.DataFrame(columns=[
        "ts_code", "pattern_trade_date", "prev_date", "curr_date", "volume_mode", "min_body_pct",
        "prev_open", "prev_close", "prev_vol", "prev_prev_vol", "prev_body_pct",
        "curr_open", "curr_close", "curr_vol", "curr_body_pct",
    ])

    if include_name and not result_df.empty:
        name_df = load_name_mapping()
        if name_df is not None:
            result_df = result_df.merge(name_df, on="ts_code", how="left")

    preferred_cols = [
        "ts_code", "name", "pattern_trade_date", "prev_date", "curr_date", "volume_mode", "min_body_pct",
        "prev_open", "prev_close", "prev_vol", "prev_prev_vol", "prev_body_pct",
        "curr_open", "curr_close", "curr_vol", "curr_body_pct",
    ]
    result_df = result_df[[c for c in preferred_cols if c in result_df.columns]]
    return result_df


@app.get("/screen/bullish-engulfing-volume")
def api_screen_bullish_engulfing_volume(
    trade_date: str,
    adj: str = Query("qfq", description="复权方式: qfq/hfq/none"),
    include_name: bool = Query(True, description="是否合并股票名称"),
    volume_mode: str = Query("any", description="成交量要求: any/shrink/expand"),
    min_body_pct: float = Query(0.01, description="最小实体比例，例如 0.01 表示 1%"),
):
    result_df = screen_bullish_engulfing_volume(
        trade_date=trade_date,
        adj=adj,
        include_name=include_name,
        volume_mode=volume_mode,
        min_body_pct=min_body_pct,
    )
    return {
        "message": "screen bullish engulfing volume ok",
        "trade_date": trade_date,
        "adj": adj,
        "volume_mode": volume_mode,
        "min_body_pct": min_body_pct,
        "rows": len(result_df),
        "columns": list(result_df.columns),
        "data": result_df.to_dict(orient="records"),
    }
