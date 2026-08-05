import logging
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ======================================================
# 1. 日誌設定
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GoldModelTrainer")

def build_and_export_model(export_path: str = "gold_rf_model.joblib"):
    # ======================================================
    # 2. 抓取 yfinance 資料並對齊合併
    # ======================================================
    logger.info("📡 正在從 yfinance 下載近 5 年黃金 (GC=F) 與美元指數 (DX-Y.NYB) 數據...")
    gold_df = yf.Ticker("GC=F").history(period="5y").reset_index()
    gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)

    dxy_df = yf.Ticker("DX-Y.NYB").history(period="5y").reset_index()
    dxy_df["Date"] = pd.to_datetime(dxy_df["Date"]).dt.tz_localize(None)
    dxy_df = dxy_df[["Date", "Close"]].rename(columns={"Close": "DXY_Close"})

    # 合併與清洗
    df = pd.merge(gold_df, dxy_df, on="Date", how="inner")
    df["BuyPrice"] = df["Close"] * 0.998
    df["SellPrice"] = df["Close"] * 1.002

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    logger.info(f"✅ 資料對齊成功，總共 {len(df)} 筆交易日紀錄。")

    # ======================================================
    # 3. 特徵工程 (嚴格消除 Look-ahead Bias)
    # ======================================================
    logger.info("⚙️ 正在進行無前瞻偏誤之特徵工程計算...")
    df["AveragePrice"] = (df["BuyPrice"] + df["SellPrice"]) / 2
    df["Spread"] = df["SellPrice"] - df["BuyPrice"]

    # Lag 特徵
    df["Return_Lag1"] = df["AveragePrice"].pct_change().shift(1)
    df["Return_Lag2"] = df["AveragePrice"].pct_change().shift(2)
    df["Return_Lag5"] = df["AveragePrice"].pct_change().shift(5)

    # 移動平均與相對距離指標
    ma5_lag1 = df["AveragePrice"].shift(1).rolling(5).mean()
    ma20_lag1 = df["AveragePrice"].shift(1).rolling(20).mean()

    df["Dist_MA5"] = (df["AveragePrice"].shift(1) - ma5_lag1) / ma5_lag1
    df["Dist_MA20"] = (df["AveragePrice"].shift(1) - ma20_lag1) / ma20_lag1
    df["Rolling_Std_Lag1"] = df["AveragePrice"].shift(1).rolling(20).std()

    # 美元指數 Lag 特徵
    df["DXY_Close_Lag1"] = df["DXY_Close"].shift(1)
    df["DXY_Return_Lag1"] = df["DXY_Close"].pct_change().shift(1)
    dxy_ma5_lag1 = df["DXY_Close"].shift(1).rolling(5).mean()
    df["DXY_Dist_MA5"] = (df["DXY_Close"].shift(1) - dxy_ma5_lag1) / dxy_ma5_lag1

    feature_cols = [
        "Spread", "Return_Lag1", "Return_Lag2", "Return_Lag5",
        "Dist_MA5", "Dist_MA20", "Rolling_Std_Lag1",
        "DXY_Close_Lag1", "DXY_Return_Lag1", "DXY_Dist_MA5"
    ]

    # 設定預測目標 (1: 明日漲, 0: 明日跌)
    df['Target'] = (df['AveragePrice'].pct_change().shift(-1) > 0).astype(int)
    model_df = df.dropna(subset=feature_cols + ['Target']).copy()

    X = model_df[feature_cols]
    y = model_df['Target']

    # ======================================================
    # 4. 三階時間序列切分 (Train / Val / Test)
    # ======================================================
    val_size = int(len(model_df) * 0.2)
    test_size = int(len(model_df) * 0.2)
    train_size = len(model_df) - val_size - test_size

    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size : train_size + val_size], y.iloc[train_size : train_size + val_size]
    X_test, y_test = X.iloc[train_size + val_size :], y.iloc[train_size + val_size :]

    logger.info(f"📊 資料切分比例 - 訓練集: {len(X_train)} | 驗證集: {len(X_val)} | 測試集: {len(X_test)}")

    # ======================================================
    # 5. 網格搜尋訓練隨機森林 (GridSearchCV + TimeSeriesSplit)
    # ======================================================
    logger.info("🔍 開始執行網格搜尋尋找隨機森林最佳超參數組合...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 5],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    logger.info(f"🏆 最佳超參數組合: {grid_search.best_params_}")

    # ======================================================
    # 6. 在 Validation 集尋找最佳門檻值 (Best Threshold)
    # ======================================================
    y_val_probs = best_rf.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_val_score = -1.0

    for threshold in np.linspace(0.3, 0.7, 41):
        preds = (y_val_probs >= threshold).astype(int)
        report = classification_report(y_val, preds, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        if macro_f1 > best_val_score:
            best_val_score = macro_f1
            best_threshold = float(threshold)

    logger.info(f"🎯 於 Validation 集找到之最佳機率門檻: {best_threshold:.2f} (Macro F1: {best_val_score:.4f})")

    # ======================================================
    # 7. 最終在 Test 集進行盲測評估
    # ======================================================
    y_test_probs = best_rf.predict_proba(X_test)[:, 1]
    final_preds = (y_test_probs >= best_threshold).astype(int)

    acc = accuracy_score(y_test, final_preds)
    auc = roc_auc_score(y_test, y_test_probs)
    
    logger.info("=== 最終盲測評估結果 ===")
    logger.info(f"測試集 Accuracy: {acc:.4f} | ROC AUC Score: {auc:.4f}")
    print("\n詳細分類報告:")
    print(classification_report(y_test, final_preds, target_names=['跌 (0)', '漲 (1)']))

    # ======================================================
    # 8. 打包與匯出成 joblib 檔案
    # ======================================================
    export_pack = {
        "model": best_rf,
        "threshold": best_threshold,
        "feature_cols": feature_cols,
        "metrics": {
            "accuracy": acc,
            "auc": auc
        }
    }

    joblib.dump(export_pack, export_path)
    logger.info(f"📦 成功將模型、門檻值與特徵欄位打包儲存至: {export_path}")

if __name__ == "__main__":
    build_and_export_model()