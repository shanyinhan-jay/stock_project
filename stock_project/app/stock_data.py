import tushare as ts
import pandas as pd

# 设置 Tushare Token
ts.set_token("193480744983c3a025f82f729f02f38ead3982d448e305700f10d606")
pro = ts.pro_api()

def fetch_data(symbol, start_date, end_date):
    # 获取指定股票的日线数据
    df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
    
    # 打印返回的数据列名和前几行数据（调试）
    print("Columns:", df.columns)
    print(df.head())
    
    # 保存为 Parquet 格式
    df.to_parquet(f"/data/{symbol}.parquet", index=False)
    print(f"数据已保存到 /data/{symbol}.parquet")

if __name__ == "__main__":
    fetch_data("600519.SH", "20230101", "20230301")  # 示例：获取贵州茅台（600519.SH）数据