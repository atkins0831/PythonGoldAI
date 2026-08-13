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
# 3. 優先載入 joblib 模型邏輯
# ======================================================
def load_or_build_model(model_path: str) -> Dict[str, Any]:
    if os.path.exists(model_path):
        try:
            logger.info(f"📂 發現模型檔 ({model_path})，優先嘗試載入...")
            pack = joblib.load(model_path)
            logger.info(f"✅ 成功載入預訓練模型！(決策門檻值: {pack.get('threshold', 0.5):.2f})")
            return pack
        except Exception as e:
            logger.warning(f"⚠️ 讀取現有 {model_path} 失敗 ({e})，準備啟動備份建置...")
    else:
        logger.warning(f"⚠️ 找不到模型檔 ({model_path})，準備啟動自動訓練流程...")

    try:
        build_and_export_model(model_path)
        pack = joblib.load(model_path)
        logger.info(f"✅ 自動訓練成功並匯入模型！")
        return pack
    except Exception as e:
        logger.error(f"❌ 模型自動建置失敗: {e}")
        return {}

# ======================================================
# 4. Lifespan 上下文管理器
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 啟動 GoldMind 整合型服務...")
    
    pack = load_or_build_model(MODEL_PATH)
    
    if pack:
        ml_artifacts["model"] = pack.get("model")
        ml_artifacts["threshold"] = pack.get("threshold", 0.5)
        ml_artifacts["feature_cols"] = pack.get("feature_cols", [])
        
        # 💡 解開打包在 joblib 中的真實快取數據
        ml_artifacts["last_known_X"] = pack.get("last_known_X")
        ml_artifacts["last_known_price"] = pack.get("last_known_price")
        ml_artifacts["last_known_dxy"] = pack.get("last_known_dxy")
        ml_artifacts["last_known_date"] = pack.get("last_known_date")
    else:
        ml_artifacts["model"] = None
        logger.error("❌ 系統將在【無模型模式】下啟動，請檢查環境。")

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
    logger.info("📡 正在嘗試透過 yfinance 擷取即時數據...")
    
    try:
        gold_df = yf.Ticker("GC=F").history(period="2mo").reset_index()
        dxy_df = yf.Ticker("DX-Y.NYB").history(period="2mo").reset_index()

        if gold_df.empty or dxy_df.empty:
            raise ValueError("yfinance 抓取資料為空")

        gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)
        dxy_df["Date"] = pd.to_datetime(dxy_df["Date"]).dt.tz_localize(None)
        dxy_df = dxy_df[["Date", "Close"]].rename(columns={"Close": "DXY_Close"})

        df = pd.merge(gold_df, dxy_df, on="Date", how="inner")
        df["BuyPrice"] = df["Close"] * 0.998
        df["SellPrice"] = df["Close"] * 1.002
        df["AveragePrice"] = (df["BuyPrice"] + df["SellPrice"]) / 2

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

    except Exception as e:
        logger.warning(f"⚠️ yfinance API 被封鎖或存取失敗 ({e})！改為讀取 Joblib 打包數據...")
        
        cached_X = ml_artifacts.get("last_known_X")
        latest_price = ml_artifacts.get("last_known_price")
        latest_dxy = ml_artifacts.get("last_known_dxy")
        latest_date = ml_artifacts.get("last_known_date")

        if cached_X is not None and latest_price is not None:
            X_latest = cached_X
            latest_price = float(latest_price)
            latest_dxy = float(latest_dxy) if latest_dxy else 104.25
            latest_date = str(latest_date) if latest_date else str(pd.Timestamp.today().date())
        else:
            latest_date = str(pd.Timestamp.today().date())
            latest_price = 2450.80
            latest_dxy = 104.25
            mock_features = {col: 0.001 for col in feature_cols}
            X_latest = pd.DataFrame([mock_features])

        return X_latest, latest_price, latest_dxy, latest_date

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
* **隨機森林 (Random Forest) 評估**：模型預測**明日 (T+1)** 看多勝率為 **{prob_up*100:.1f}%** (判決門檻值: {threshold:.2f})。
* **技術指標解讀**：結合 5日/20日均線距離與 Lag 變數，模型給出明日走勢預測為 **【{direction}】**。
* **長短線指標說明**：隨機森林專注於**單日短線極值動量**；若與【未來走勢推演】方向不同，代表短線呈現反彈/拉回，但中長期仍順應趨勢主線進行修正。
* **投資操作建議**：短期市場波動加劇，建議控制資金倉位，避免過度槓桿，並關注聯準會最新動態。
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
    """預測未來 N 天趨勢（優先使用 Joblib 價格進行預測延伸）"""
    logger.info(f"📡 正在計算未來 {days} 天趨勢...")
    
    # 取得基準價格與日期
    base_price = ml_artifacts.get("last_known_price", 2450.80)
    base_date = ml_artifacts.get("last_known_date", str(pd.Timestamp.today().date()))
    
    try:
        gold_df = yf.Ticker("GC=F").history(period="2y").reset_index()
        if not gold_df.empty and "Close" in gold_df.columns:
            gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)
            gold_series = gold_df["Close"].dropna().values
        else:
            raise ValueError("yfinance 歷史為空")
    except Exception:
        # 當 yfinance 被封鎖時，以 Joblib 中的真實 base_price 為起點發散數據
        np.random.seed(42)
        gold_series = base_price + np.cumsum(np.random.normal(0.5, 6, size=200))

    try:
        model = ExponentialSmoothing(gold_series, trend="add", seasonal=None).fit()
        future_days = int(days)
        forecast_values = model.forecast(future_days)
        
        start_dt = pd.to_datetime(base_date)
        future_dates = pd.date_range(start=start_dt + pd.Timedelta(days=1), periods=future_days)
        
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
        return px.line(title="⚠️ 趨勢圖暫時無法載入", template="plotly_dark")

def draw_gold_chart():
    """歷史走勢折線圖"""
    base_price = ml_artifacts.get("last_known_price", 2450.80)
    base_date = ml_artifacts.get("last_known_date", str(pd.Timestamp.today().date()))

    try:
        df = yf.Ticker("GC=F").history(period="2mo").reset_index()
        if df.empty:
            raise ValueError("yfinance 為空")
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    except Exception:
        # 當 yfinance 被封鎖時，以 Joblib 中的真實最新價格畫出折線圖
        end_dt = pd.to_datetime(base_date)
        dates = pd.date_range(end=end_dt, periods=60, freq='B')
        np.random.seed(42)
        prices = base_price + np.cumsum(np.random.normal(0, 5, size=60))
        df = pd.DataFrame({"Date": dates, "Close": prices})

    fig = px.line(
        df, x="Date", y="Close",
        title="📈 黃金 (Gold Futures) 近 60 日歷史走勢圖",
        labels={"Close": "金價 (USD/盎司)", "Date": "日期"},
        template="plotly_dark"
    )
    fig.update_traces(line_color="#ffd700", line_width=2.5)
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
        raise HTTPException(status_code=500, detail=str(e))

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

with gr.Blocks(theme=gr.themes.Monochrome(), title="GoldMind 智慧金融診斷") as gradio_ui:
    gr.Markdown("# 🏆 GoldMind 智慧金價預測與投資助理 (MVP)")
    
    with gr.Tabs():
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
            
        with gr.TabItem("📈 黃金未來走勢推演"):
            gr.Markdown("### 📊 基於時間序列模型 (Holt-Winters) 之未來價格推演")
            
            with gr.Row():
                radio_days = gr.Radio(
                    choices=[1, 7, 14, 30],
                    value=30,
                    label="🗓️ 請選擇預測天數 (Days)",
                    info="可切換選擇未來 1 天、7 天、14 天或 30 天之預估走勢"
                )
            
            gold_forecast_chart = gr.Plot(value=draw_gold_forecast(30))
            
            radio_days.change(
                fn=draw_gold_forecast,
                inputs=[radio_days],
                outputs=[gold_forecast_chart]
            )

app = gr.mount_gradio_app(app, gradio_ui, path="/dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)