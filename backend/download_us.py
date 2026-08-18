import pandas as pd
import requests
import yfinance as yf

symbol = "DX-Y.NYB"  # 美元指數代碼
start_date = "2024-08-11"  # 開始日期 (建議拉長至 2 年，利於計算 MA60)
end_date = "2026-08-18"  # 結束日期 (需比目標日期多 1 天，才能包含 08-17 當天)
interval = "1d"  # 資料頻率 (1d/1w/1mo)

print(f"⏳ 正在下載 {symbol} (美元指數) 歷史數據...")

# 建立模擬瀏覽器的 Header 以防被擋
session = requests.Session()
session.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,'
        ' like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
})

# 下載數據
df = yf.download(
    symbol,
    start=start_date,
    end=end_date,
    interval=interval,
    session=session,
    progress=False,
)

# 1. 檢查是否抓到資料
if df.empty:
  print("❌ 下載失敗：未抓取到任何資料，請檢查日期設定或網路連線。")
else:
  # 2. 如果欄位是 MultiIndex，進行平坦化處理
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  # 3. 整理欄位與日期
  df = df.reset_index()
  date_col = "Date" if "Date" in df.columns else df.columns[0]
  df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)

  # 重新命名欄位 (Price 代表日期，DXY_Close 代表美元收盤價)
  df = df.rename(columns={date_col: "Price", "Close": "DXY_Close"})

  # 留下核心欄位並保留小數位數 2 位
  df["DXY_Close"] = df["DXY_Close"].round(2)
  output_df = df[["Price", "DXY_Close", "Open", "High", "Low", "Volume"]]

  # 4. 匯出 CSV
  csv_filename = "DX-Y.NYB_history.csv"
  output_df.to_csv(csv_filename, index=False)

  print(f"✅ 美元指數 CSV 下載成功！已儲存至: {csv_filename}")
  print(f"📊 總筆數: {len(output_df)} 筆，最新 3 筆數據：")
  print(output_df.tail(3))