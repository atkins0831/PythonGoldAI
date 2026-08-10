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
    logger.info("📡 正在透過 yfinance 擷取 GC=F 與 DX-Y.NYB 即時數據...")
    gold_df = yf.Ticker("GC=F").history(period="1mo").reset_index()
    gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)

    dxy_df = yf.Ticker("DX-Y.NYB").history(period="1mo").reset_index()
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

def draw_oil_30day_forecast():
    """抓取原油歷史資料，並預測未來 30 天趨勢 (含防空值備份機制)"""
    logger.info("📡 正在抓取原油 (CL=F) 資料並計算未來 30 天趨勢...")
    
    try:
        # 1. 抓取原油期貨歷史數據
        oil_ticker = yf.Ticker("CL=F")
        oil_df = oil_ticker.history(period="6mo")
        
        # 檢查抓到的資料是否為空
        if oil_df.empty:
            logger.warning("⚠️ yfinance 未能取得 CL=F 數據，嘗試使用 USO 替代...")
            oil_df = yf.Ticker("USO").history(period="6mo")

        if oil_df.empty:
            raise ValueError("無法取得原油市場數據 (CL=F 與 USO 皆為空)")

        oil_df = oil_df.reset_index()
        
        # 處理時區問題並保留 Date 與 Close 欄位
        if "Date" in oil_df.columns:
            oil_df["Date"] = pd.to_datetime(oil_df["Date"]).dt.tz_localize(None)
        
        oil_df = oil_df[["Date", "Close"]].dropna()

        # 再次確認清理後的資料筆數是否足夠做時間序列分析
        if len(oil_df) < 10:
            raise ValueError(f"原油數據筆數過少 (僅 {len(oil_df)} 筆)，無法建立 Holt-Winters 模型")

        # 2. 建立 ExponentialSmoothing 模型
        model = ExponentialSmoothing(
            oil_df["Close"].values,  # 傳入 numpy array 避免索引錯位
            trend="add", 
            seasonal=None
        ).fit()
        
        future_days = 30
        forecast_values = model.forecast(future_days)
        
        # 3. 建立未來 30 天日期
        last_date = oil_df["Date"].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        
        # 4. 合併歷史與預測
        df_history = pd.DataFrame({
            "Date": oil_df["Date"],
            "Price": oil_df["Close"],
            "Type": "歷史實際價格"
        })
        
        df_forecast = pd.DataFrame({
            "Date": future_dates,
            "Price": forecast_values,
            "Type": "未來 30 天預測趨勢"
        })
        
        df_combined = pd.concat([df_history, df_forecast], ignore_index=True)
        
        # 5. 繪製 Plotly 圖表
        fig = px.line(
            df_combined, x="Date", y="Price", color="Type",
            title="🛢️ 原油 (WTI Crude Oil) 歷史走勢與未來 30 天趨勢預測",
            labels={"Price": "原油價格 (USD/桶)", "Date": "日期"},
            color_discrete_map={"歷史實際價格": "#1f77b4", "未來 30 天預測趨勢": "#ff7f0e"},
            template="plotly_dark"
        )
        fig.update_traces(line_width=2.5)
        return fig

    except Exception as e:
        logger.error(f"❌ 原油繪圖失敗: {e}")
        # 備份機制：回傳一張寫有錯誤提示的空圖表，防止整隻 FastAPI 服務崩潰 Exit 1
        fig = px.line(title=f"⚠️ 原油趨勢暫時無法載入 ({str(e)})", template="plotly_dark")
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

@app.get("/api/v1/oil-30day-forecast", summary="原油未來 30 天趨勢預測")
async def get_oil_forecast():
    try:
        oil_df = yf.Ticker("CL=F").history(period="3mo").reset_index()
        oil_df = oil_df[["Date", "Close"]].dropna()
        
        model = ExponentialSmoothing(oil_df["Close"], trend="add").fit()
        forecast = model.forecast(30)
        
        last_date = pd.to_datetime(oil_df["Date"].iloc[-1])
        future_dates = [str((last_date + pd.Timedelta(days=i)).date()) for i in range(1, 31)]
        
        return {
            "status": "success",
            "base_date": str(last_date.date()),
            "forecast_dates": future_dates,
            "forecast_prices": [round(p, 2) for p in forecast]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"油價預測失敗: {str(e)}")

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

# with gr.Blocks(theme=gr.themes.Monochrome(), title="GoldMind 金價預測與 AI 投資助理") as gradio_ui:
#     gr.Markdown("# 🏆 GoldMind 智慧金價預測與投資助理 (MVP)")
#     gr.Markdown("整合 **FastAPI API + Gradio UI + 隨機森林自動化重練** 之金融儀表板")
    
#     with gr.Row():
#         with gr.Column(scale=2):
#             chart_output = gr.Plot(value=draw_gold_chart())
        
#         with gr.Column(scale=1):
#             gr.Markdown("### ⚙️ 即時 ML 預測開關")
#             btn_predict = gr.Button("🚀 抓取即時數據並執行 AI 分析", variant="primary")
            
#             out_price = gr.Textbox(label="當前黃金收盤價 (Close)", interactive=False)
#             out_dxy = gr.Textbox(label="當前美元指數 (DXY)", interactive=False)
#             out_prob = gr.Textbox(label="模型看多機率 (Probability)", interactive=False)
#             out_dir = gr.Textbox(label="明日預測走勢 (Direction)", interactive=False)

#     gr.Markdown("---")
#     out_report = gr.Markdown("💡 **點擊【執行 AI 分析】按鈕，以取得最新智慧診斷報告。**")

#     btn_predict.click(
#         fn=gradio_handle_predict,
#         inputs=[],
#         outputs=[out_price, out_dxy, out_prob, out_dir, out_report]
#     )

with gr.Blocks(theme=gr.themes.Monochrome(), title="GoldMind 智慧金融診斷") as gradio_ui:
    gr.Markdown("# 🏆 GoldMind 智慧金價與原油市場助理 (MVP)")
    
    with gr.Tabs():
        # 分頁 1：黃金每日預測與 AI 診斷
        with gr.TabItem("🥇 黃金預測與 AI 診斷"):
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
            
        # 分頁 2：原油未來 30 天趨勢分析 (延伸功能)
        with gr.TabItem("🛢️ 原油未來 30 天趨勢預測"):
            gr.Markdown("### 📊 基於時間序列模型 (Holt-Winters) 之原油 30 天價格推演")
            oil_chart = gr.Plot(value=draw_oil_30day_forecast())
            btn_refresh_oil = gr.Button("🔄 重新整理原油預測線", variant="secondary")
            btn_refresh_oil.click(fn=draw_oil_30day_forecast, inputs=[], outputs=[oil_chart])

# 關鍵：將 Gradio 掛載到 FastAPI 的 `/dashboard` 子路徑
app = gr.mount_gradio_app(app, gradio_ui, path="/dashboard")

# ======================================================
# 9. 本機直接執行入口
# ======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)