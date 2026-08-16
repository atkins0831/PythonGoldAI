import yfinance as yf

# 直接替換變數
symbol = "GC=F"         # 股票代號 / 商品代碼
start_date = "2024-08-17"  # 開始日期 (YYYY-MM-DD)
end_date = "2026-08-17"    # 結束日期 (YYYY-MM-DD)
interval = "1d"         # 資料頻率 (1d/1w/1mo)

# 自動下載並匯出 CSV
df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
df.to_csv(f"{symbol}_history.csv")
print("CSV 下載成功！")