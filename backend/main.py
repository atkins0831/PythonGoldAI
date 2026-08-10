import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Tuple
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import gradio as gr
import plotly.express as px

# 引用訓練腳本中的建構函式
from train_and_save import build_and_export_model

# ======================================================
# 1. Logging 日誌設定
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GoldMindApp")

# ======================================================
# 2. 全域變數與 Pydantic Data Models
# ======================================================
MODEL_PATH = "gold_rf_model.joblib"
MAX_MODEL_AGE_DAYS = 1  # 模型過期天數門檻
ml_artifacts: Dict[str, Any] = {}

class HealthCheckResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    version: str = Field(..., example="1.0.0")

class GoldPredictionResponse(BaseModel):
    date: str = Field(..., example="2026-08-05")
    latest_price: float = Field(..., example=2450.50)
    latest_dxy: float = Field(..., example=104.25)
    prob_up: float = Field(..., example=68.5)
    threshold: float = Field(..., example=0.52)
    direction: str = Field(..., example="漲 📈 (Bullish)")
    ai_report: str = Field(..., example="### 🤖 GoldMind AI 語意診斷報告...")

# ======================================================
# 3. 檢查模型檔是否存在或過期的邏輯
# ======================================================
def ensure_model_exists_and_fresh(model_path: str, max_days: int = 7):
    """檢查 .joblib 是否存在，若不存在或修改時間超過 max_days 則自動重新訓練"""
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ 找不到模型檔 ({model_path})，準備自動執行模型訓練...")
        build_and_export_model(model_path)
        return

    # 檢查檔案最後修改時間
    file_mod_time = os.path.getmtime(model_path)
    age_in_days = (time.time() - file_mod_time) / (24 * 3600)
    
    if age_in_days > max_days:
        logger.warning(f"⚠️ 模型檔已過期 ({age_in_days:.1f} 天 > {max_days} 天)，自動重練新模型...")
        build_and_export_model(model_path)
    else:
        logger.info(f"✅ 模型檔存在且新鮮 (已使用 {age_in_days:.1f} 天)。")

# ======================================================
# 4. Lifespan 上下文管理器 (自動偵測/訓練 + 載入模型)
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 啟動 GoldMind 整合型服務...")
    
    # 步驟 1: 檢查模型檔是否存在或過期
    ensure_model_exists_and_fresh(MODEL_PATH, max_days=MAX_MODEL_AGE_DAYS)
    
    # 步驟 2: 載入模型 artifact
    try:
        pack = joblib.load(MODEL_PATH)
        ml_artifacts["model"] = pack["model"]
        ml_artifacts["threshold"] = pack["threshold"]
        ml_artifacts["feature_cols"] = pack["feature_cols"]
        logger.info(f"✅ 成功載入模型：{MODEL_PATH} (決策門檻: {pack['threshold']:.2f})")
    except Exception as e:
        logger.error(f"❌ 載入模型失敗 ({MODEL_PATH}): {e}")
        ml_artifacts["model"] = None

    yield
    
    ml_artifacts.clear()
    logger.info("🛑 服務已安全關閉。")

# ======================================================
# 5. 初始化 FastAPI App 與 CORS
# ======================================================
app = FastAPI(
    title="GoldMind ML Predict Service",
    description="智慧金價預測與 AI 投資診斷 API 與 Gradio UI 整合服務",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 6. Helper 業務邏輯函式
# ======================================================
def fetch_and_prepare_features(feature_cols: list) -> Tuple[pd.DataFrame, float, float, str]:
    logger.info("📡 正在透過 yfinance 擷取黃金與美元指數即時數據 (含多標的備份機制)...")
    
    # 1. 嘗試擷取黃金標的 (GC=F -> GLD -> XAUUSD=X)
    gold_df = pd.DataFrame()
    gold_symbols = ["GC=F", "GLD", "XAUUSD=X"]
    for sym in gold_symbols:
        try:
            df_temp = yf.Ticker(sym).history(period="2mo").reset_index()
            if not df_temp.empty and len(df_temp) >= 5:
                gold_df = df_temp
                logger.info(f"✅ 成功擷取黃金標的數據: {sym}")
                break
        except Exception as e:
            logger.warning(f"⚠️ 標的 {sym} 擷取失敗，嘗試下一個備份標的...")

    # 2. 嘗試擷取美元指數標的 (DX-Y.NYB -> UUP -> DX=F)
    dxy_df = pd.DataFrame()
    dxy_symbols = ["DX-Y.NYB", "UUP", "DX=F"]
    for sym in dxy_symbols:
        try:
            df_temp = yf.Ticker(sym).history(period="2mo").reset_index()
            if not df_temp.empty and len(df_temp) >= 5:
                dxy_df = df_temp
                logger.info(f"✅ 成功擷取美元指數標的數據: {sym}")
                break
        except Exception as e:
            logger.warning(f"⚠️ 標的 {sym} 擷取失敗，嘗試下一個備份標的...")

    # 防護機制：萬一全部標的都被封鎖，拋出明確例外
    if gold_df.empty or dxy_df.empty:
        raise ValueError("Yahoo Finance 暫時限制連線，無法順利取得黃金或美元指數數據")

    gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)
    dxy_df["Date"] = pd.to_datetime(dxy_df["Date"]).dt.tz_localize(None)
    dxy_df = dxy_df[["Date", "Close"]].rename(columns={"Close": "DXY_Close"})

    df = pd.merge(gold_df, dxy_df, on="Date", how="inner")
    df["BuyPrice"] = df["Close"] * 0.998
    df["SellPrice"] = df["Close"] * 1.002
    df["AveragePrice"] = (df["BuyPrice"] + df["SellPrice"]) / 2
    df["Spread"] = df["SellPrice"] - df["BuyPrice"]

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

    latest_row = df.dropna(subset=feature_cols).iloc[-1]
    latest_price = float(latest_row["Close"])
    latest_dxy = float(latest_row["DXY_Close"])
    latest_date = str(latest_row["Date"]).split()[0]
    
    X_latest = pd.DataFrame([latest_row[feature_cols]])
    return X_latest, latest_price, latest_dxy, latest_date



# 核心推論函式 (供 REST API 與 Gradio 共同呼叫)
def run_prediction_logic():
    model = ml_artifacts.get("model")
    threshold = ml_artifacts.get("threshold", 0.5)
    feature_cols = ml_artifacts.get("feature_cols", [])

    if not model:
        raise ValueError("Machine learning model is not initialized.")

    X_latest, latest_price, latest_dxy, latest_date = fetch_and_prepare_features(feature_cols)
    prob_up = float(model.predict_proba(X_latest)[:, 1][0])
    is_up = prob_up >= threshold
    direction = "漲 📈 (Bullish)" if is_up else "跌 📉 (Bearish)"

    ai_report = f"""
### 🤖 GoldMind AI 語意診斷報告 ({latest_date})

* **即時市場觀察**：最新黃金收盤價為 **${latest_price:,.2f} USD**，美元指數 (DXY) 落在 **{latest_dxy:.2f}**。
* **隨機森林 (Random Forest) 評估**：模型預測明日看多勝率為 **{prob_up*100:.1f}%** (最佳判決門檻值為 {threshold:.2f})。
* **技術指標解讀**：結合與 5日/20日均線之相對距離與 Lag 變數，模型給出明日走勢為 **【{direction}】**。
* **投資操作建議**：短期市場波動加劇，建議投資者控制資金倉位，避免過度槓桿，並關注聯準會最新動態。
    """

    return {
        "date": latest_date,
        "latest_price": round(latest_price, 2),
        "latest_dxy": round(latest_dxy, 2),
        "prob_up": round(prob_up * 100, 2),
        "threshold": round(threshold, 2),
        "direction": direction,
        "ai_report": ai_report
    }

def draw_gold_forecast(days: int = 30):
    """抓取黃金歷史資料，並預測未來 N 天趨勢（含多標的備份機制）"""
    logger.info(f"📡 正在抓取黃金資料並計算未來 {days} 天趨勢...")
    
    try:
        gold_df = pd.DataFrame()
        for sym in ["GC=F", "GLD", "XAUUSD=X"]:
            try:
                df_temp = yf.Ticker(sym).history(period="6mo").reset_index()
                if not df_temp.empty and len(df_temp) >= 10:
                    gold_df = df_temp
                    break
            except Exception:
                continue

        if gold_df.empty:
            raise ValueError("無法取得黃金市場數據 (所有備份標的皆無回應)")

        if "Date" in gold_df.columns:
            gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)
        
        gold_df = gold_df[["Date", "Close"]].dropna()

        # 時間序列 Holt-Winters 模型
        model = ExponentialSmoothing(
            gold_df["Close"].values,
            trend="add", 
            seasonal=None
        ).fit()
        
        future_days = int(days)
        forecast_values = model.forecast(future_days)
        
        last_date = gold_df["Date"].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        
        df_forecast = pd.DataFrame({
            "日期": future_dates,
            "預測金價 (USD)": [round(val, 2) for val in forecast_values]
        })
        
        fig = px.line(
            df_forecast, x="日期", y="預測金價 (USD)",
            title=f"🥇 黃金未來 {future_days} 天趨勢推演 (純預測區間)",
            markers=True,
            template="plotly_dark"
        )
        fig.update_traces(line_color="#ffd700", line_width=3, marker_size=7)
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="未來預測日期",
            yaxis_title="預估金價 (USD/盎司)"
        )
        return fig

    except Exception as e:
        logger.error(f"❌ 黃金趨勢繪圖失敗: {e}")
        fig = px.line(title=f"⚠️ 黃金趨勢暫時無法載入 ({str(e)})", template="plotly_dark")
        return fig


# ======================================================
# 7. FastAPI REST API 端點
# ======================================================
@app.get("/", response_model=HealthCheckResponse, summary="健康檢查")
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        model_loaded=ml_artifacts.get("model") is not None,
        version="1.0.0"
    )

@app.post("/api/v1/predict", response_model=GoldPredictionResponse, summary="執行預測")
async def predict_gold():
    try:
        res = run_prediction_logic()
        return GoldPredictionResponse(**res)
    except Exception as e:
        logger.error(f"❌ 預測失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/api/v1/gold-forecast", summary="黃金未來 N 天趨勢預測")
async def get_gold_forecast_api(days: int = 30):
    try:
        gold_df = yf.Ticker("GC=F").history(period="3mo").reset_index()
        gold_df = gold_df[["Date", "Close"]].dropna()
        
        model = ExponentialSmoothing(gold_df["Close"].values, trend="add").fit()
        forecast = model.forecast(days)
        
        last_date = pd.to_datetime(gold_df["Date"].iloc[-1])
        future_dates = [str((last_date + pd.Timedelta(days=i)).date()) for i in range(1, days + 1)]
        
        return {
            "status": "success",
            "forecast_days": days,
            "base_date": str(last_date.date()),
            "forecast_dates": future_dates,
            "forecast_prices": [round(p, 2) for p in forecast]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"金價預測失敗: {str(e)}")

@app.get("/api/v1/gold-history", summary="黃金近期歷史走勢")
async def get_gold_history_api(days: int = 60):
    try:
        period = "1mo" if days <= 30 else "3mo"
        gold_df = yf.Ticker("GC=F").history(period=period).reset_index()
        if gold_df.empty:
            raise ValueError("無法取得黃金市場數據 (GC=F)")
        gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)
        history = [
            {"date": str(row["Date"]).split()[0], "price": round(float(row["Close"]), 2)}
            for _, row in gold_df.tail(days).iterrows()
        ]
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"歷史走勢載入失敗: {str(e)}")

# ======================================================
# 8. 建立 Gradio UI 並掛載至 FastAPI
# ======================================================
def gradio_handle_predict():
    try:
        data = run_prediction_logic()
        return (
            f"${data['latest_price']:,} USD",
            f"{data['latest_dxy']}",
            f"{data['prob_up']}% (門檻 {data['threshold']})",
            data['direction'],
            data['ai_report']
        )
    except Exception as e:
        return "Error", "Error", "Error", "執行失敗", f"預測錯誤: {str(e)}"

def draw_gold_chart():
    df = yf.Ticker("GC=F").history(period="2mo").reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    fig = px.line(
        df, x="Date", y="Close",
        title="📈 黃金 (Gold Futures) 近 60 日歷史走勢圖",
        labels={"Close": "金價 (USD/盎司)", "Date": "日期"},
        template="plotly_dark"
    )
    fig.update_traces(line_color="#ffd700", line_width=2.5)
    return fig

with gr.Blocks(theme=gr.themes.Monochrome(), title="GoldMind 智慧金融診斷") as gradio_ui:
    gr.Markdown("# 🏆 GoldMind 智慧金價預測與投資助理 (MVP)")
    
    with gr.Tabs():
        # 分頁 1：黃金每日預測與 AI 診斷
        with gr.TabItem("🥇 黃金每日預測與 AI 診斷"):
            with gr.Row():
                with gr.Column(scale=2):
                    chart_output = gr.Plot(value=draw_gold_chart())
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 即時 ML 預測開關")
                    btn_predict = gr.Button("🚀 抓取即時數據並執行 AI 分析", variant="primary")
                    out_price = gr.Textbox(label="當前黃金收盤價 (Close)", interactive=False)
                    out_dxy = gr.Textbox(label="當前美元指數 (DXY)", interactive=False)
                    out_prob = gr.Textbox(label="模型看多機率", interactive=False)
                    out_dir = gr.Textbox(label="明日預測走勢", interactive=False)
            gr.Markdown("---")
            out_report = gr.Markdown("💡 **點擊【執行 AI 分析】按鈕，以取得最新智慧診斷報告。**")

            btn_predict.click(
                fn=gradio_handle_predict,
                inputs=[],
                outputs=[out_price, out_dxy, out_prob, out_dir, out_report]
            )
            
        # 分頁 2：黃金未來 N 天趨勢推演 (純預測)
        with gr.TabItem("📈 黃金未來走勢推演"):
            gr.Markdown("### 📊 基於時間序列模型 (Holt-Winters) 之未來價格推演")
            
            with gr.Row():
                radio_days = gr.Radio(
                    choices=[1, 7, 14, 30],
                    value=30,
                    label="🗓️ 請選擇預測天數 (Days)",
                    info="可切換選擇未來 1 天、7 天、14 天或 30 天之預估走勢"
                )
            
            # 初始化預設顯示未來 30 天純預測圖表
            gold_forecast_chart = gr.Plot(value=draw_gold_forecast(30))
            
            # 當使用者選擇不同的 Radio 選項時，自動觸發重新繪圖
            radio_days.change(
                fn=draw_gold_forecast,
                inputs=[radio_days],
                outputs=[gold_forecast_chart]
            )

# 關鍵：將 Gradio 掛載到 FastAPI 的 `/dashboard` 子路徑
app = gr.mount_gradio_app(app, gradio_ui, path="/dashboard")

# ======================================================
# 9. 本機直接執行入口
# ======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)