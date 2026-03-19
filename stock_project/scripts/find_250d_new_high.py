import duckdb
import pandas as pd

TARGET_DATE = "20260319"   # 查询日期
LOOKBACK_DAYS = 400        # 多取一些自然日，保证覆盖 250 个交易日
WINDOW_TRADING_DAYS = 250
LAST_N_DAYS = 5

# 本地文件路径（宿主机）
DAILY_PARQUET = "/opt/stock_project/data/daily/symbol=*/year=*.parquet"
ADJ_PARQUET = "/opt/stock_project/data/adj_factor/symbol=*/year=*.parquet"
NAME_FILE = "/opt/stock_project/data/stock_basic_min.csv"

con = duckdb.connect()

daily_sql = f"""
SELECT *
FROM read_parquet('{DAILY_PARQUET}')
WHERE CAST(trade_date AS DATE)
      BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
"""

adj_sql = f"""
SELECT ts_code, trade_date, adj_factor
FROM read_parquet('{ADJ_PARQUET}')
WHERE CAST(trade_date AS DATE)
      BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
"""

target_dt = pd.to_datetime(TARGET_DATE, format="%Y%m%d")
target_date_sql = target_dt.strftime("%Y-%m-%d")
start_dt = target_dt - pd.Timedelta(days=LOOKBACK_DAYS)
start_date_sql = start_dt.strftime("%Y-%m-%d")

# 读取数据
daily = con.execute(daily_sql, [start_date_sql, target_date_sql]).fetchdf()
adj = con.execute(adj_sql, [start_date_sql, target_date_sql]).fetchdf()

daily["trade_date"] = pd.to_datetime(daily["trade_date"])
adj["trade_date"] = pd.to_datetime(adj["trade_date"])

for col in ["open", "high", "low", "close", "pre_close"]:
    if col in daily.columns:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")

# 合并复权因子
df = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

# 前复权
latest_factor = df.groupby("ts_code")["adj_factor"].transform("last")
for col in ["open", "high", "low", "close", "pre_close"]:
    if col in df.columns:
        df[col] = df[col] * df["adj_factor"] / latest_factor

results = []

for ts_code, g in df.groupby("ts_code"):
    g = g.sort_values("trade_date").reset_index(drop=True)

    # 至少有 250 个交易日
    if len(g) < WINDOW_TRADING_DAYS:
        continue

    # 只保留最近 250 个交易日
    g = g.tail(WINDOW_TRADING_DAYS).reset_index(drop=True)

    # 最近 5 个交易日
    last_5 = g.tail(LAST_N_DAYS).copy()

    # 250 交易日新高
    rolling_high_max = g["high"].max()
    rolling_close_max = g["close"].max()

    hit_high_rows = last_5[last_5["high"] >= rolling_high_max]
    hit_close_rows = last_5[last_5["close"] >= rolling_close_max]

    hit_high = not hit_high_rows.empty
    hit_close = not hit_close_rows.empty

    if hit_high or hit_close:
        hit_dates = pd.concat([
            hit_high_rows[["trade_date"]],
            hit_close_rows[["trade_date"]],
        ]).drop_duplicates().sort_values("trade_date")

        results.append({
            "ts_code": ts_code,
            "last_trade_date": g.iloc[-1]["trade_date"].strftime("%Y-%m-%d"),
            "window_trading_days": WINDOW_TRADING_DAYS,
            "hit_high": bool(hit_high),
            "hit_close": bool(hit_close),
            "high_break_dates": ",".join(hit_high_rows["trade_date"].dt.strftime("%Y-%m-%d").tolist()),
            "close_break_dates": ",".join(hit_close_rows["trade_date"].dt.strftime("%Y-%m-%d").tolist()),
            "hit_dates": ",".join(hit_dates["trade_date"].dt.strftime("%Y-%m-%d").tolist()),
            "latest_close_qfq": g.iloc[-1]["close"],
            "latest_high_qfq": g.iloc[-1]["high"],
            "rolling_250_high_max": rolling_high_max,
            "rolling_250_close_max": rolling_close_max,
        })

result_df = pd.DataFrame(results).sort_values("ts_code")

# ===== 合并股票名称 =====
name_df = pd.read_csv(NAME_FILE).drop_duplicates(subset=["ts_code"])
result_df = result_df.merge(name_df, on="ts_code", how="left")

# 调整列顺序
preferred_cols = [
    "ts_code", "name",
    "last_trade_date",
    "hit_high", "hit_close",
    "high_break_dates", "close_break_dates", "hit_dates",
    "latest_close_qfq", "latest_high_qfq",
    "rolling_250_high_max", "rolling_250_close_max",
    "window_trading_days"
]
result_df = result_df[[c for c in preferred_cols if c in result_df.columns]]

output_file = f"week_250d_new_high_qfq_{TARGET_DATE}.csv"
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("数量:", len(result_df))
if not result_df.empty:
    print(result_df.head(30).to_string(index=False))
print(f"结果已保存到 {output_file}")