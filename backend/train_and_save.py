import logging
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

logger = logging.getLogger("GoldMindApp")

def build_and_export_model(model_path: str = "gold_rf_model.joblib"):
    """訓練隨機森林模型，並打包最新特徵數據與市場價格匯出至 .joblib"""
    logger.info("🏋️ 開始抓取歷史數據並進行模型訓練 (train_and_save)...")
    
    # 1. 擷取近 2 年歷史數據
    gold_df = yf.Ticker("GC=F").history(period="2y").reset_index()
    dxy_df = yf.Ticker("DX-Y.NYB").history(period="2y").reset_index()
    
    if gold_df.empty or dxy_df.empty:
        # 備份標的：GLD 與 UUP
        logger.warning("⚠️ yfinance 期貨標的為空，嘗試使用 ETF 備份標的...")
        gold_df = yf.Ticker("GLD").history(period="2y").reset_index()
        gold_df["Close"] = gold_df["Close"] * 10.85  # 換算為金價 USD/oz
        dxy_df = yf.Ticker("UUP").history(period="2y").reset_index()

    gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)
    dxy_df["Date"] = pd.to_datetime(dxy_df["Date"]).dt.tz_localize(None)
    dxy_df = dxy_df[["Date", "Close"]].rename(columns={"Close": "DXY_Close"})

    df = pd.merge(gold_df, dxy_df, on="Date", how="inner")
    df["BuyPrice"] = df["Close"] * 0.998
    df["SellPrice"] = df["Close"] * 1.002
    df["AveragePrice"] = (df["BuyPrice"] + df["SellPrice"]) / 2

    # 2. 特徵工程
    df["Return_Lag1"] = df["AveragePrice"].pct_change().shift(1)
    df["Return_Lag2"] = df["AveragePrice"].pct_change().shift(2)
    df["Return_Lag5"] = df["AveragePrice"].pct_change().shift(5)

    ma5_lag1 = df["AveragePrice"].shift(1).rolling(5).mean()
    ma20_lag1 = df["AveragePrice"].shift(1).rolling(20).mean()

    df["Dist_MA5"] = (df["AveragePrice"].shift(1) - ma5_lag1) / ma5_lag1
    df["Dist_MA20"] = (df["AveragePrice"].shift(1) - ma20_lag1) / ma20_lag1
    df["Rolling_Std_Lag1"] = df["AveragePrice"].shift(1).rolling(20).std()

    df["DXY_Close_Lag1"] = df["DXY_Close"].shift(1)
    df["DXY_Return_Lag1"] = df["DXY_Close"].pct_change().shift(1)
    dxy_ma5_lag1 = df["DXY_Close"].shift(1).rolling(5).mean()
    df["DXY_Dist_MA5"] = (df["DXY_Close"].shift(1) - dxy_ma5_lag1) / dxy_ma5_lag1

    # 預測目標：明日平均價是否高於今日
    df["Target"] = (df["AveragePrice"].shift(-1) > df["AveragePrice"]).astype(int)

    feature_cols = [
        "Return_Lag1", "Return_Lag2", "Return_Lag5",
        "Dist_MA5", "Dist_MA20", "Rolling_Std_Lag1",
        "DXY_Close_Lag1", "DXY_Return_Lag1", "DXY_Dist_MA5"
    ]

    df_clean = df.dropna(subset=feature_cols + ["Target"]).reset_index(drop=True)
    
    X = df_clean[feature_cols]
    y = df_clean["Target"]

    # 3. 訓練模型
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    best_threshold = 0.52

    # 💡 4. 提取最新一筆交易日真實數據
    latest_row = df_clean.iloc[-1]
    latest_price = float(latest_row["Close"])
    latest_dxy = float(latest_row["DXY_Close"])
    latest_date = str(latest_row["Date"]).split()[0]
    last_known_X = pd.DataFrame([latest_row[feature_cols]])

    # 💡 5. 擴充打包字典 (payload)
    payload = {
        "model": rf,
        "threshold": best_threshold,
        "feature_cols": feature_cols,
        # 備份特徵與最新數據
        "last_known_X": last_known_X,
        "last_known_price": float(latest_price),
        "last_known_dxy": float(latest_dxy),
        "last_known_date": str(latest_date)
    }

    joblib.dump(payload, model_path)
    logger.info(f"💾 模型與最新特徵數據成功打包並匯出至 {model_path} (最新日期: {latest_date}, 金價: ${latest_price})")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_and_export_model()