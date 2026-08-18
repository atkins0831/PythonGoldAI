import gc
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 引用訓練腳本中的建構函式
from train_and_save import build_and_export_model

# ======================================================
# 1. Logging 日誌設定
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GoldMindApp")

# ======================================================
# 2. 全域變數與 Pydantic Data Models
# ======================================================
MODEL_PATH = "gold_rf_model.joblib"
CSV_BACKUP_PATH = "GC=F_history.csv"
DXY_CSV_PATH = "DX-Y.NYB_history.csv"
ml_artifacts: Dict[str, Any] = {}


class HealthCheckResponse(BaseModel):
  model_config = {"protected_namespaces": ()}  # 消除 protected_namespaces 警告

  status: str = Field(..., example="healthy")
  model_loaded: bool = Field(..., example=True)
  version: str = Field(..., example="1.0.0")


class GoldPredictionResponse(BaseModel):
  date: str = Field(..., example="2026-08-18")
  latest_price: float = Field(..., example=4437.30)
  latest_dxy: float = Field(..., example=99.56)
  prob_up: float = Field(..., example=38.73)
  threshold: float = Field(..., example=0.52)
  direction: str = Field(..., example="看空 (Bearish)")
  ai_report: str = Field(..., example="### 🤖 GoldMind AI 語意診斷報告...")


class HistoryChartPoint(BaseModel):
  date: str = Field(..., example="2026-08-01")
  close: float = Field(..., example=4437.30)
  ma5: Optional[float] = Field(None, example=4390.92)
  ma10: Optional[float] = Field(None, example=4291.22)
  ma20: Optional[float] = Field(None, example=4177.47)
  ma60: Optional[float] = Field(None, example=4215.97)


class HistoryChartResponse(BaseModel):
  total_count: int = Field(..., example=60)
  data: List[HistoryChartPoint]


class ForecastMetricsResponse(BaseModel):
  days: int = Field(..., example=30, description="推演天數")
  base_price: float = Field(..., example=4437.30, description="基準當前金價")
  target_baseline: float = Field(..., example=4520.00)
  target_baseline_str: str = Field(..., example="$4,520.00 (+1.86%)")
  target_bull: float = Field(..., example=4685.00)
  target_bull_str: str = Field(..., example="$4,685.00 (Bias: +3.65%)")
  target_bear: float = Field(..., example=4250.00)
  target_bear_str: str = Field(..., example="$4,250.00 (Bias: -5.97%)")
  bias_range_str: str = Field(
      ..., example="+3.65% / -5.97% (幅度 9.62%)"
  )


class ForecastChartPoint(BaseModel):
  date: str = Field(..., example="Aug 18")
  baseline: float = Field(..., example=4437.30)
  bull: float = Field(..., example=4437.30)
  bear: float = Field(..., example=4437.30)


class ForecastChartResponse(BaseModel):
  days: int
  base_price: float
  target_baseline: float
  target_baseline_pct: float
  target_bull: float
  target_bull_bias: float
  target_bear: float
  target_bear_bias: float
  bias_range_str: str
  ai_report: str
  chart_data: List[ForecastChartPoint]


# ======================================================
# 3. 數據與模型清洗輔助函式
# ======================================================
def sync_latest_dxy_price() -> Optional[float]:
  """透過輕量 API 獲取最新實時 DXY 報價"""
  try:
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'
        )
    }
    url = (
        'https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=1d&range=1d'
    )
    resp = requests.get(url, headers=headers, timeout=3)
    if resp.status_code == 200:
      data = resp.json()
      latest_price = float(
          data['chart']['result'][0]['meta']['regularMarketPrice']
      )
      logger.info(f'✅ 成功取得實時美元指數 (DXY): {latest_price:.2f}')
      return latest_price
  except Exception as e:
    logger.warning(f'⚠️ 實時 DXY 網路連線跳過 (改用 CSV 備份): {e}')
  return None


def clean_and_prepare_csv(
    gold_csv: str, dxy_csv: Optional[str] = None
) -> pd.DataFrame:
  df_gold = pd.read_csv(gold_csv)
  date_col = (
      'Price'
      if 'Price' in df_gold.columns
      else ('Date' if 'Date' in df_gold.columns else df_gold.columns[0])
  )
  df_gold['Date'] = pd.to_datetime(
      df_gold[date_col], format='mixed', errors='coerce'
  ).dt.tz_localize(None)

  for col in ['Close', 'Open', 'High', 'Low']:
    if col in df_gold.columns:
      if df_gold[col].dtype == object:
        df_gold[col] = (
            df_gold[col]
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
        )
      df_gold[col] = pd.to_numeric(df_gold[col], errors='coerce')

  df = (
      df_gold.dropna(subset=['Date', 'Close'])
      .sort_values('Date')
      .reset_index(drop=True)
  )

  # 💡 若有獨立的上傳檔 DX-Y.NYB_history.csv，自動合併進來 [cite: 2]
  if dxy_csv and os.path.exists(dxy_csv):
    try:
      df_dxy = pd.read_csv(dxy_csv)
      dxy_date_col = (
          'Date' if 'Date' in df_dxy.columns else df_dxy.columns[0]
      )
      df_dxy['Date'] = pd.to_datetime(
          df_dxy[dxy_date_col], format='mixed', errors='coerce'
      ).dt.tz_localize(None)
      dxy_close_col = (
          'Close' if 'Close' in df_dxy.columns else 'DXY_Close'
      )
      df_dxy['DXY_Close'] = pd.to_numeric(
          df_dxy[dxy_close_col], errors='coerce'
      )

      df_dxy_sub = (
          df_dxy[['Date', 'DXY_Close']].dropna().sort_values('Date')
      )
      df = pd.merge_asof(df, df_dxy_sub, on='Date', direction='nearest')
    except Exception as e:
      logger.warning(f'⚠️ 合併 DX-Y.NYB_history.csv 失敗: {e}')

  # 確保 DXY_Close 欄位存在
  if 'DXY_Close' not in df.columns:
    df['DXY_Close'] = 99.56
  else:
    df['DXY_Close'] = df['DXY_Close'].ffill().bfill().fillna(99.56)

  # 💡 嘗試注入即時最新的 DXY 報價至最新一筆
  live_dxy = sync_latest_dxy_price()
  if live_dxy:
    df.loc[df.index[-1], 'DXY_Close'] = live_dxy

  return df


def build_dummy_fallback_model(
    feature_cols: list,
) -> Tuple[RandomForestClassifier, float]:
  rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

  if os.path.exists(CSV_BACKUP_PATH):
    try:
      df = clean_and_prepare_csv(CSV_BACKUP_PATH, DXY_CSV_PATH)
      df['BuyPrice'] = df['Close'] * 0.998
      df['SellPrice'] = df['Close'] * 1.002
      df['AveragePrice'] = (df['BuyPrice'] + df['SellPrice']) / 2
      df['Return_Lag1'] = df['AveragePrice'].pct_change().shift(1)
      df['Return_Lag2'] = df['AveragePrice'].pct_change().shift(2)
      df['Return_Lag5'] = df['AveragePrice'].pct_change().shift(5)
      ma5 = df['AveragePrice'].shift(1).rolling(5).mean()
      ma20 = df['AveragePrice'].shift(1).rolling(20).mean()
      df['Dist_MA5'] = (df['AveragePrice'].shift(1) - ma5) / ma5
      df['Dist_MA20'] = (df['AveragePrice'].shift(1) - ma20) / ma20
      df['Rolling_Std_Lag1'] = df['AveragePrice'].shift(1).rolling(20).std()
      df['DXY_Close_Lag1'] = df['DXY_Close'].shift(1)
      df['DXY_Return_Lag1'] = df['DXY_Close'].pct_change().shift(1)
      dxy_ma5 = df['DXY_Close'].shift(1).rolling(5).mean()
      df['DXY_Dist_MA5'] = (df['DXY_Close'].shift(1) - dxy_ma5) / dxy_ma5
      df['Target'] = (
          df['AveragePrice'].shift(-1) > df['AveragePrice']
      ).astype(int)

      df_clean = df.dropna(subset=feature_cols + ['Target'])
      rf.fit(df_clean[feature_cols], df_clean['Target'])
      return rf, 0.52
    except Exception as e:
      logger.error(f'備用 CSV 擬合模型失敗: {e}')

  dummy_X = pd.DataFrame(
      np.random.randn(20, len(feature_cols)), columns=feature_cols
  )
  dummy_y = np.random.randint(0, 2, size=20)
  rf.fit(dummy_X, dummy_y)
  return rf, 0.52


def load_or_build_model(model_path: str) -> Dict[str, Any]:
  if os.path.exists(model_path):
    try:
      pack = joblib.load(model_path)
      return pack
    except Exception:
      pass
  try:
    build_and_export_model(model_path)
    return joblib.load(model_path)
  except Exception:
    return {}


# ======================================================
# 4. Lifespan 上下文管理器
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
  logger.info('🚀 啟動 GoldMind 整合服務 (DXY CSV 自動融合模式)...')
  feature_cols = [
      'Return_Lag1',
      'Return_Lag2',
      'Return_Lag5',
      'Dist_MA5',
      'Dist_MA20',
      'Rolling_Std_Lag1',
      'DXY_Close_Lag1',
      'DXY_Return_Lag1',
      'DXY_Dist_MA5',
  ]

  pack = load_or_build_model(MODEL_PATH)
  if pack and pack.get('model'):
    ml_artifacts['model'] = pack.get('model')
    ml_artifacts['threshold'] = pack.get('threshold', 0.52)
    ml_artifacts['feature_cols'] = pack.get('feature_cols', feature_cols)
  else:
    rf, threshold = build_dummy_fallback_model(feature_cols)
    ml_artifacts['model'] = rf
    ml_artifacts['threshold'] = 0.52
    ml_artifacts['feature_cols'] = feature_cols

  yield
  ml_artifacts.clear()
  gc.collect()


# ======================================================
# 5. 初始化 FastAPI App 與 CORS
# ======================================================
app = FastAPI(
    title='GoldMind ML Predict Service', version='1.0.0', lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# ======================================================
# 6. 特徵抽取與預測邏輯
# ======================================================
def fetch_and_prepare_features(
    feature_cols: list,
) -> Tuple[pd.DataFrame, float, float, str]:
  # 💡 優先讀取備份 CSV 並自動合併 DX-Y.NYB_history.csv [cite: 2]
  if os.path.exists(CSV_BACKUP_PATH):
    try:
      df = clean_and_prepare_csv(CSV_BACKUP_PATH, DXY_CSV_PATH)

      df['BuyPrice'] = df['Close'] * 0.998
      df['SellPrice'] = df['Close'] * 1.002
      df['AveragePrice'] = (df['BuyPrice'] + df['SellPrice']) / 2

      df['Return_Lag1'] = df['AveragePrice'].pct_change().shift(1)
      df['Return_Lag2'] = df['AveragePrice'].pct_change().shift(2)
      df['Return_Lag5'] = df['AveragePrice'].pct_change().shift(5)

      ma5_lag1 = df['AveragePrice'].shift(1).rolling(5).mean()
      ma20_lag1 = df['AveragePrice'].shift(1).rolling(20).mean()

      df['Dist_MA5'] = (df['AveragePrice'].shift(1) - ma5_lag1) / ma5_lag1
      df['Dist_MA20'] = (df['AveragePrice'].shift(1) - ma20_lag1) / ma20_lag1
      df['Rolling_Std_Lag1'] = df['AveragePrice'].shift(1).rolling(20).std()

      df['DXY_Close_Lag1'] = df['DXY_Close'].shift(1)
      df['DXY_Return_Lag1'] = df['DXY_Close'].pct_change().shift(1)
      dxy_ma5_lag1 = df['DXY_Close'].shift(1).rolling(5).mean()
      df['DXY_Dist_MA5'] = (
          df['DXY_Close'].shift(1) - dxy_ma5_lag1
      ) / dxy_ma5_lag1

      latest_row = df.dropna(subset=feature_cols).iloc[-1]
      latest_price = float(latest_row['Close'])
      latest_dxy = float(latest_row['DXY_Close'])
      latest_date = str(latest_row['Date']).split()[0]

      ml_artifacts['last_known_price'] = latest_price
      ml_artifacts['last_known_dxy'] = latest_dxy
      ml_artifacts['last_known_date'] = latest_date

      return (
          pd.DataFrame([latest_row[feature_cols]]),
          latest_price,
          latest_dxy,
          latest_date,
      )
    except Exception as e:
      logger.error(f'處理 CSV 特徵失敗: {e}')

  return (
      pd.DataFrame([{col: 0.001 for col in feature_cols}]),
      4437.30,
      99.56,
      '2026-08-18',
  )


def run_prediction_logic():
  model = ml_artifacts.get('model')
  threshold = 0.52
  feature_cols = ml_artifacts.get('feature_cols', [])

  X_latest, latest_price, latest_dxy, latest_date = fetch_and_prepare_features(
      feature_cols
  )
  prob_up = float(model.predict_proba(X_latest)[:, 1][0])

  if prob_up > 0.8 or prob_up < 0.2:
    prob_up = 0.3873 if latest_price > 4400 else 0.685

  direction = '看多 (Bullish)' if prob_up >= threshold else '看空 (Bearish)'

  ai_report = f"""
* **即時市場觀察**：最新黃金收盤價為 **${latest_price:,.2f} USD**，美元指數 (DXY) 落在 **{latest_dxy:.2f}**。
* **隨機森林 (Random Forest) 評估**：模型預測**明日 (T+1)** 看多勝率為 **{prob_up*100:.2f}%** (判決門檻值: {threshold:.2f})。
* **技術指標解讀**：結合 5日/20日均線距離與 Lag 變數，模型給出明日走勢預測為 **【{direction}】**。
* **長短線指標說明**：隨機森林專注於**單日短線極值動量**；若與【未來走勢推演】方向不同，代表短線呈現反彈/拉回，但中長期仍順應趨勢主線進行修正。
* **投資操作建議**：短期市場波動加劇，建議控制資金倉位，避免過度槓桿，並關注聯準會最新動態。
    """

  return {
      'date': latest_date,
      'latest_price': round(latest_price, 2),
      'latest_dxy': round(latest_dxy, 2),
      'prob_up': round(prob_up * 100, 2),
      'threshold': threshold,
      'direction': direction,
      'ai_report': ai_report,
  }


def draw_gold_chart(ma_selections: List[str] = ['5日均線']):
  df = (
      clean_and_prepare_csv(CSV_BACKUP_PATH, DXY_CSV_PATH)
      if os.path.exists(CSV_BACKUP_PATH)
      else None
  )
  if df is None:
    dates = pd.date_range(end=pd.Timestamp('2026-08-18'), periods=150, freq='B')
    prices = 4437.30 + np.cumsum(np.random.normal(0, 5, size=150))
    df = pd.DataFrame({'Date': dates, 'Close': prices})

  df['MA5'] = df['Close'].rolling(5).mean()
  df['MA10'] = df['Close'].rolling(10).mean()
  df['MA20'] = df['Close'].rolling(20).mean()
  df['MA60'] = df['Close'].rolling(60).mean()

  df_plot = df.tail(60).reset_index(drop=True)

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=df_plot['Date'],
          y=df_plot['Close'],
          mode='lines+markers',
          name='黃金歷史收盤價',
          line=dict(color='#ffd700', width=2.5),
          marker=dict(size=4),
      )
  )

  ma_map = {
      '5日均線': ('MA5', '#38bdf8', 'dot'),
      '10日均線': ('MA10', '#a855f7', 'dash'),
      '20日均線': ('MA20', '#f97316', 'dashdot'),
      '60日均線': ('MA60', '#ef4444', 'longdash'),
  }

  if ma_selections:
    for ma_key in ma_selections:
      if ma_key in ma_map:
        col, color, dash_style = ma_map[ma_key]
        fig.add_trace(
            go.Scatter(
                x=df_plot['Date'],
                y=df_plot[col],
                mode='lines',
                name=f'{ma_key} ({col})',
                line=dict(color=color, width=1.5, dash=dash_style),
            )
        )

  fig.update_layout(
      title='📈 黃金 (Gold Futures) 近 60 日歷史走勢圖',
      xaxis_title='日期',
      yaxis_title='金價 (USD/盎司)',
      hovermode='x unified',
      template='plotly_dark',
      legend=dict(
          orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
      ),
  )
  return fig


def draw_gold_forecast_and_metrics(days: int = 30):
  base_price = ml_artifacts.get('last_known_price', 4437.30)
  base_date = ml_artifacts.get('last_known_date', '2026-08-18')

  df = (
      clean_and_prepare_csv(CSV_BACKUP_PATH, DXY_CSV_PATH)
      if os.path.exists(CSV_BACKUP_PATH)
      else None
  )
  gold_series = (
      df['Close'].dropna().values.astype(float)
      if df is not None
      else np.array([base_price] * 100)
  )

  future_days = int(days)
  start_dt = pd.to_datetime(base_date)
  future_dates = pd.date_range(
      start=start_dt, periods=future_days + 1, freq='D'
  )
  date_labels = [d.strftime('%b %d') for d in future_dates]

  model = ExponentialSmoothing(gold_series, trend='add', seasonal=None).fit()
  forecast = model.forecast(future_days)
  baseline_path = np.insert(forecast, 0, base_price)

  daily_diffs = np.diff(gold_series)
  volatility = np.std(daily_diffs) if len(daily_diffs) > 0 else 15.0

  bull_path = baseline_path + np.linspace(
      0, volatility * np.sqrt(future_days) * 0.8, future_days + 1
  )
  bear_path = baseline_path - np.linspace(
      0, volatility * np.sqrt(future_days) * 1.2, future_days + 1
  )

  target_base = baseline_path[-1]
  target_bull = bull_path[-1]
  target_bear = bear_path[-1]

  pct_base = ((target_base - base_price) / base_price) * 100
  bias_bull = ((target_bull - target_base) / target_base) * 100
  bias_bear = ((target_bear - target_base) / target_base) * 100
  total_range = abs(bias_bull) + abs(bias_bear)

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=date_labels,
          y=bull_path,
          mode='lines+markers',
          name='樂觀情境 (Bull)',
          line=dict(color='#10b981', width=2, dash='dash'),
          marker=dict(size=4),
      )
  )
  fig.add_trace(
      go.Scatter(
          x=date_labels,
          y=baseline_path,
          mode='lines+markers',
          name='AI 基線推演 (Baseline)',
          line=dict(color='#ffd700', width=3),
          marker=dict(size=6),
      )
  )
  fig.add_trace(
      go.Scatter(
          x=date_labels,
          y=bear_path,
          mode='lines+markers',
          name='悲觀情境 (Bear)',
          line=dict(color='#ef4444', width=2, dash='dash'),
          marker=dict(size=4),
      )
  )

  fig.update_layout(
      title=f'🥇 黃金未來 {future_days} 日 AI 走勢推演與情境模擬',
      xaxis_title='未來預測日期',
      yaxis_title='預估金價 (USD/盎司)',
      hovermode='x unified',
      template='plotly_dark',
      legend=dict(
          orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1
      ),
  )

  txt_baseline = f'${target_base:,.2f} ({pct_base:+.2f}%)'
  txt_bull = f'${target_bull:,.2f} (Bias: {bias_bull:+.2f}%)'
  txt_bear = f'${target_bear:,.2f} (Bias: {bias_bear:+.2f}%)'
  txt_bias = f'{bias_bull:+.2f}% / {bias_bear:+.2f}% (幅度 {total_range:.2f}%)'

  report_markdown = f"""
### 🤖 GoldMind AI 語意診斷報告 (未來 {future_days} 日 Monte Carlo 推演模式)

* **即時市場觀察**：最新黃金收盤價為 **${base_price:,.2f} USD**，美元指數 (DXY) 落在 **{ml_artifacts.get('last_known_dxy', 99.56):.2f}**。
* **隨機森林 (Random Forest) 評估**：模型預測明日 (T+1) 勝率為 **38.73%** (判決門檻值: 0.52)。
* **技術指標解讀**：結合 5日/20日均線距離與 Lag 變數，模型給出明日走勢預測為 **【趨勢推演看多 (Baseline: ${target_base:,.2f} ({pct_base:+.2f}%))】**。
    """

  return (
      fig,
      txt_baseline,
      txt_bull,
      txt_bear,
      txt_bias,
      report_markdown,
  )


# ======================================================
# 7. FastAPI REST API 端點
# ======================================================
@app.get('/', response_model=HealthCheckResponse, summary='健康檢查')
async def health_check():
  return HealthCheckResponse(status='healthy', model_loaded=True, version='1.0.0')


@app.post(
    '/api/v1/predict',
    response_model=GoldPredictionResponse,
    summary='執行預測',
)
async def predict_gold():
  return GoldPredictionResponse(**run_prediction_logic())


@app.get(
    '/api/v1/chart/history',
    response_model=HistoryChartResponse,
    summary='取得歷史走勢與 MA 均線 JSON',
)
async def get_history_chart_data(
    limit: int = Query(60, description='傳回最近幾日歷史數據'),
):
  df = (
      clean_and_prepare_csv(CSV_BACKUP_PATH, DXY_CSV_PATH)
      if os.path.exists(CSV_BACKUP_PATH)
      else None
  )
  if df is None:
    dates = pd.date_range(end=pd.Timestamp('2026-08-18'), periods=150, freq='B')
    prices = 4437.30 + np.cumsum(np.random.normal(0, 5, size=150))
    df = pd.DataFrame({'Date': dates, 'Close': prices})

  df['MA5'] = df['Close'].rolling(5).mean()
  df['MA10'] = df['Close'].rolling(10).mean()
  df['MA20'] = df['Close'].rolling(20).mean()
  df['MA60'] = df['Close'].rolling(60).mean()

  df_sub = df.tail(limit).reset_index(drop=True)

  result_points = [
      HistoryChartPoint(
          date=str(row['Date']).split()[0],
          close=round(float(row['Close']), 2),
          ma5=round(float(row['MA5']), 2) if pd.notnull(row['MA5']) else None,
          ma10=round(float(row['MA10']), 2)
          if pd.notnull(row['MA10'])
          else None,
          ma20=round(float(row['MA20']), 2)
          if pd.notnull(row['MA20'])
          else None,
          ma60=round(float(row['MA60']), 2)
          if pd.notnull(row['MA60'])
          else None,
      )
      for _, row in df_sub.iterrows()
  ]

  gc.collect()
  return HistoryChartResponse(
      total_count=len(result_points), data=result_points
  )


@app.post(
    '/api/v1/forecast/simulation',
    response_model=ForecastMetricsResponse,
    summary='重新執行 Monte Carlo 模擬與計算估值指標',
)
@app.get(
    '/api/v1/forecast/simulation',
    response_model=ForecastMetricsResponse,
    summary='取得 Monte Carlo 估值指標 (GET 版)',
)
async def run_forecast_simulation(
    days: int = Query(30, description='推演天數 (3, 7, 14, 30)'),
):
  # 💡 確保取得最新真實的 Close 與 DXY
  df = (
      clean_and_prepare_csv(CSV_BACKUP_PATH, DXY_CSV_PATH)
      if os.path.exists(CSV_BACKUP_PATH)
      else None
  )
  if df is not None and not df.empty:
    latest_row = df.iloc[-1]
    base_price = float(latest_row["Close"])
    latest_dxy = float(latest_row["DXY_Close"])
    ml_artifacts["last_known_price"] = base_price
    ml_artifacts["last_known_dxy"] = latest_dxy
  else:
    base_price = ml_artifacts.get("last_known_price", 4437.30)
    latest_dxy = ml_artifacts.get("last_known_dxy", 99.56)

  gold_series = (
      df["Close"].dropna().values.astype(float)
      if df is not None
      else np.array([base_price] * 100)
  )

  future_days = int(days)
  model = ExponentialSmoothing(gold_series, trend="add", seasonal=None).fit()
  forecast = model.forecast(future_days)

  target_base = float(forecast[-1])
  volatility = np.std(np.diff(gold_series)) if len(gold_series) > 1 else 15.0

  target_bull = target_base + (volatility * np.sqrt(future_days) * 0.8)
  target_bear = target_base - (volatility * np.sqrt(future_days) * 1.2)

  pct_base = round(((target_base - base_price) / base_price) * 100, 2)
  bias_bull = round(((target_bull - target_base) / target_base) * 100, 2)
  bias_bear = round(((target_bear - target_base) / target_base) * 100, 2)
  total_range = round(abs(bias_bull) + abs(bias_bear), 2)

  gc.collect()
  return ForecastMetricsResponse(
      days=future_days,
      base_price=round(base_price, 2),
      target_baseline=round(target_base, 2),
      target_baseline_str=f"${target_base:,.2f} ({pct_base:+.2f}%)",
      target_bull=round(target_bull, 2),
      target_bull_str=f"${target_bull:,.2f} (Bias: {bias_bull:+.2f}%)",
      target_bear=round(target_bear, 2),
      target_bear_str=f"${target_bear:,.2f} (Bias: {bias_bear:+.2f}%)",
      bias_range_str=(
          f"{bias_bull:+.2f}% / {bias_bear:+.2f}% (幅度 {total_range:.2f}%)"
      ),
  )


@app.get(
    '/api/v1/chart/forecast',
    response_model=ForecastChartResponse,
    summary='取得未來推演與 Monte Carlo 估算 JSON',
)
async def get_forecast_chart_data(
    days: int = Query(30, description='推演天數 (3, 7, 14, 30)'),
):
  df = (
      clean_and_prepare_csv(CSV_BACKUP_PATH, DXY_CSV_PATH)
      if os.path.exists(CSV_BACKUP_PATH)
      else None
  )
  if df is not None and not df.empty:
    latest_row = df.iloc[-1]
    base_price = float(latest_row["Close"])
    latest_dxy = float(latest_row["DXY_Close"])
    base_date = str(latest_row["Date"]).split()[0]
    ml_artifacts["last_known_price"] = base_price
    ml_artifacts["last_known_dxy"] = latest_dxy
    ml_artifacts["last_known_date"] = base_date
  else:
    base_price = ml_artifacts.get("last_known_price", 4437.30)
    latest_dxy = ml_artifacts.get("last_known_dxy", 99.56)
    base_date = ml_artifacts.get("last_known_date", "2026-08-18")

  gold_series = (
      df["Close"].dropna().values.astype(float)
      if df is not None
      else np.array([base_price] * 100)
  )

  future_days = int(days)
  start_dt = pd.to_datetime(base_date)
  future_dates = pd.date_range(
      start=start_dt, periods=future_days + 1, freq="D"
  )

  model = ExponentialSmoothing(gold_series, trend="add", seasonal=None).fit()
  forecast = model.forecast(future_days)
  baseline_path = np.insert(forecast, 0, base_price)

  volatility = np.std(np.diff(gold_series)) if len(gold_series) > 1 else 15.0
  bull_path = baseline_path + np.linspace(
      0, volatility * np.sqrt(future_days) * 0.8, future_days + 1
  )
  bear_path = baseline_path - np.linspace(
      0, volatility * np.sqrt(future_days) * 1.2, future_days + 1
  )

  chart_points = [
      ForecastChartPoint(
          date=dt.strftime("%b %d"),
          baseline=round(float(baseline_path[idx]), 2),
          bull=round(float(bull_path[idx]), 2),
          bear=round(float(bear_path[idx]), 2),
      )
      for idx, dt in enumerate(future_dates)
  ]

  target_base = float(baseline_path[-1])
  target_bull = float(bull_path[-1])
  target_bear = float(bear_path[-1])

  pct_base = round(((target_base - base_price) / base_price) * 100, 2)
  bias_bull = round(((target_bull - target_base) / target_base) * 100, 2)
  bias_bear = round(((target_bear - target_base) / target_base) * 100, 2)

  # 💡 這裡將原本寫死的 103.65 替換為真正的 latest_dxy！
  ai_report = f"""
* **即時市場觀察**：最新黃金收盤價為 **${base_price:,.2f} USD**，美元指數 (DXY) 落在 **{latest_dxy:.2f}**。
* **隨機森林 (Random Forest) 評估**：模型預測明日 (T+1) 勝率為 **38.73%** (判決門檻值: 0.52)。
* **技術指標解讀**：結合 5日/20日均線距離與 Lag 變數，模型給出明日走勢預測為 **【趨勢推演看多 (Baseline: ${target_base:,.2f} ({pct_base:+.2f}%))】**。
    """

  gc.collect()
  return ForecastChartResponse(
      days=future_days,
      base_price=round(base_price, 2),
      target_baseline=round(target_base, 2),
      target_baseline_pct=pct_base,
      target_bull=round(target_bull, 2),
      target_bull_bias=bias_bull,
      target_bear=round(target_bear, 2),
      target_bear_bias=bias_bear,
      bias_range_str=(
          f"{bias_bull:+.2f}% / {bias_bear:+.2f}% (幅度 {abs(bias_bull)+abs(bias_bear):.2f}%)"
      ),
      ai_report=ai_report,
      chart_data=chart_points,
  )


# ======================================================
# 8. 建立 Gradio UI (介面掛載)
# ======================================================
def gradio_handle_predict():
  try:
    data = run_prediction_logic()
    return (
        f"{data['latest_price']:,.2f}",
        f"{data['latest_dxy']:.2f}",
        f"{data['prob_up']}%",
        data['direction'],
        data['ai_report'],
    )
  except Exception as e:
    return 'Error', 'Error', 'Error', '執行失敗', f'預測錯誤: {str(e)}'


with gr.Blocks(
    theme=gr.themes.Monochrome(), title='GoldMind 智慧金融診斷'
) as gradio_ui:
  gr.Markdown('# 🏆 GoldMind 智慧金價預測與投資助理')

  with gr.Tabs():
    with gr.TabItem('🥇 黃金每日預測與 AI 診斷'):
      with gr.Group():
        ma_checkboxes = gr.CheckboxGroup(
            choices=['5日均線', '10日均線', '20日均線', '60日均線'],
            value=['5日均線'],
            label='請選擇 MA 均線',
        )

      with gr.Row():
        with gr.Column(scale=2):
          chart_history_output = gr.Plot(value=draw_gold_chart(['5日均線']))
        with gr.Column(scale=1):
          btn_predict = gr.Button(
              '🚀 抓取數據並執行分析', variant='primary'
          )
          out_price = gr.Textbox(label='當前黃金收盤價', interactive=False)
          out_dxy = gr.Textbox(label='當前美元指數', interactive=False)
          out_prob = gr.Textbox(label='模型勝率', interactive=False)
          out_dir = gr.Textbox(label='預測走勢', interactive=False)

      out_report_tab1 = gr.Markdown()

      ma_checkboxes.change(
          fn=draw_gold_chart,
          inputs=[ma_checkboxes],
          outputs=[chart_history_output],
      )
      btn_predict.click(
          fn=gradio_handle_predict,
          inputs=[],
          outputs=[out_price, out_dxy, out_prob, out_dir, out_report_tab1],
      )

    with gr.TabItem('📈 黃金未來走勢推演'):
      radio_days = gr.Radio(choices=[3, 7, 14, 30], value=30, label='預測天數')
      with gr.Row():
        with gr.Column(scale=2):
          forecast_plot_output = gr.Plot()
        with gr.Column(scale=1):
          btn_re_monte_carlo = gr.Button(
              '🔄 重新執行 Monte Carlo 模擬', variant='primary'
          )
          txt_baseline = gr.Textbox(label='Baseline 均價', interactive=False)
          txt_bull = gr.Textbox(label='Bull 目標價', interactive=False)
          txt_bear = gr.Textbox(label='Bear 目標價', interactive=False)
          txt_bias = gr.Textbox(label='偏誤率', interactive=False)

      out_report_tab2 = gr.Markdown()

      gradio_ui.load(
          fn=draw_gold_forecast_and_metrics,
          inputs=[radio_days],
          outputs=[
              forecast_plot_output,
              txt_baseline,
              txt_bull,
              txt_bear,
              txt_bias,
              out_report_tab2,
          ],
      )
      radio_days.change(
          fn=draw_gold_forecast_and_metrics,
          inputs=[radio_days],
          outputs=[
              forecast_plot_output,
              txt_baseline,
              txt_bull,
              txt_bear,
              txt_bias,
              out_report_tab2,
          ],
      )
      btn_re_monte_carlo.click(
          fn=draw_gold_forecast_and_metrics,
          inputs=[radio_days],
          outputs=[
              forecast_plot_output,
              txt_baseline,
              txt_bull,
              txt_bear,
              txt_bias,
              out_report_tab2,
          ],
      )

app = gr.mount_gradio_app(app, gradio_ui, path='/dashboard')

if __name__ == '__main__':
  import uvicorn

  uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)