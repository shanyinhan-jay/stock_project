import argparse
import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

BASE_DIR = Path("/data")
META_DIR = BASE_DIR / "meta"
DUCKDB_FILE = str(META_DIR / "stock.duckdb")


def load_name_mapping() -> Optional[pd.DataFrame]:
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


def screen_three_day_pattern(
    trade_date: str,
    adj: str = "qfq",
    include_name: bool = True,
) -> pd.DataFrame:
    if adj not in ("qfq", "hfq", "none"):
        raise ValueError("adj must be one of: none, qfq, hfq")

    con = duckdb.connect(DUCKDB_FILE)

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
        raise RuntimeError("no daily data found in local warehouse")

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
            raise RuntimeError("no adj_factor data found in local warehouse")

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
            pd.notna(d2["vol"]) and pd.notna(d1["close"]) and pd.notna(d1["vol"]) and pd.notna(d1["open"]) and
            d2["open"] < d1["close"] and
            d2["close"] > d2["open"] and
            d2["close"] < d1["open"] and
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

    result_df = (
        pd.DataFrame(results).sort_values("ts_code")
        if results else
        pd.DataFrame(columns=[
            "ts_code", "pattern_trade_date", "d1_date", "d2_date", "d3_date",
            "d1_open", "d1_close", "d1_vol", "d0_vol",
            "d2_open", "d2_close", "d2_vol",
            "d3_open", "d3_close",
        ])
    )

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


def main():
    parser = argparse.ArgumentParser(description="筛选最近三个交易日的日K形态")
    parser.add_argument("--trade-date", required=True, help="交易日，例如 20260319")
    parser.add_argument("--adj", default="qfq", choices=["qfq", "hfq", "none"], help="复权方式")
    parser.add_argument("--no-name", action="store_true", help="不合并股票名称")
    parser.add_argument("--output", default="", help="输出 CSV 文件路径")
    args = parser.parse_args()

    result_df = screen_three_day_pattern(
        trade_date=args.trade_date,
        adj=args.adj,
        include_name=not args.no_name,
    )

    print(f"trade_date={args.trade_date}, adj={args.adj}, rows={len(result_df)}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved: {out_path}")
    else:
        if result_df.empty:
            print("no result")
        else:
            print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
