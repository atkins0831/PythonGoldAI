import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ======================================================
# 1. 常數與全域設定
# ======================================================
MODEL_FILENAME = "gold_rf_model.joblib"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME
FEATURE_COLS = [
    "Spread", "Return_Lag1", "Return_Lag2", "Return_Lag5",
    "Dist_MA5", "Dist_MA20", "Rolling_Std_Lag1",
    "DXY_Close_Lag1", "DXY_Return_Lag1", "DXY_Dist_MA5"
]

PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 5],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5]
}

sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False


def fetch_market_data(period: str = "5y") -> pd.DataFrame:
    """從 yfinance 擷取黃金與美元指數資料並對齊交易日。"""
    gold_df = yf.Ticker("GC=F").history(period=period).reset_index()
    gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.tz_localize(None)

    dxy_df = yf.Ticker("DX-Y.NYB").history(period=period).reset_index()
    dxy_df["Date"] = pd.to_datetime(dxy_df["Date"]).dt.tz_localize(None)
    dxy_df = dxy_df[["Date", "Close"]].rename(columns={"Close": "DXY_Close"})

    df = pd.merge(gold_df, dxy_df, on="Date", how="inner")
    df["BuyPrice"] = df["Close"] * 0.998
    df["SellPrice"] = df["Close"] * 1.002
    return df


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """清理缺失值與重複值。"""
    df = df.copy()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """新增無前視偏誤特徵。"""
    df = df.copy()
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

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """建立二元目標變數：隔日漲跌。"""
    df = df.copy()
    df['Target'] = (df['AveragePrice'].pct_change().shift(-1) > 0).astype(int)
    return df


def prepare_datasets(
    df: pd.DataFrame,
    feature_cols: list,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
):
    """依時間序列切分訓練 / 驗證 / 測試資料。"""
    model_df = df.dropna(subset=feature_cols + ['Target']).copy()
    X = model_df[feature_cols]
    y = model_df['Target']

    n_total = len(model_df)
    val_size = int(n_total * val_ratio)
    test_size = int(n_total * test_ratio)
    train_size = n_total - val_size - test_size

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size: train_size + val_size]
    y_val = y.iloc[train_size: train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict = None,
    cv_split: int = 3
) -> tuple[RandomForestClassifier, GridSearchCV]:
    """使用時間序列交叉驗證進行 Random Forest 超參數搜尋。"""
    if param_grid is None:
        param_grid = PARAM_GRID

    tscv = TimeSeriesSplit(n_splits=cv_split)
    base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search


def find_best_threshold(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: np.ndarray = None
) -> tuple[float, float]:
    """在驗證集上搜尋最佳預測機率門檻。"""
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.7, 41)

    y_val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_val_score = -1.0

    for threshold in thresholds:
        preds = (y_val_probs >= threshold).astype(int)
        report = classification_report(y_val, preds, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        if macro_f1 > best_val_score:
            best_val_score = macro_f1
            best_threshold = threshold

    return best_threshold, best_val_score


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float
) -> dict:
    """評估測試集表現並回傳評估內容。"""
    y_test_probs = model.predict_proba(X_test)[:, 1]
    final_preds = (y_test_probs >= threshold).astype(int)

    return {
        'accuracy': accuracy_score(y_test, final_preds),
        'roc_auc': roc_auc_score(y_test, y_test_probs),
        'confusion_matrix': confusion_matrix(y_test, final_preds),
        'classification_report': classification_report(y_test, final_preds, target_names=['跌 (0)', '漲 (1)'])
    }


def save_model_artifact(
    model: RandomForestClassifier,
    feature_cols: list,
    threshold: float,
    model_path: Path = MODEL_PATH
) -> None:
    """將訓練好的模型與參數儲存成 joblib 檔案。"""
    artifact = {
        'model': model,
        'feature_cols': feature_cols,
        'threshold': threshold
    }
    joblib.dump(artifact, model_path)
    print(f"✅ 已儲存模型檔案到：{model_path}")


def train_and_save_model(
    period: str = "5y",
    model_path: Path = MODEL_PATH
) -> dict:
    """整體訓練流程：擷取資料、清洗、特徵工程、訓練、選門檻、評估、存檔。"""
    df = fetch_market_data(period=period)
    df = clean_market_data(df)
    df = engineer_features(df)
    df = create_target(df)

    print("=== Data Overview ===")
    print(df.info())
    print("\n=== Selected Feature Statistics ===")
    print(df[FEATURE_COLS].describe())
    print("\n" + "=" * 50 + "\n")

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(df, FEATURE_COLS)
    print(f"訓練集筆數: {len(X_train)}, 驗證集筆數: {len(X_val)}, 測試集筆數: {len(X_test)}")

    best_rf, grid_search = train_random_forest(X_train, y_train)
    best_threshold, best_val_score = find_best_threshold(best_rf, X_val, y_val)

    print(f"最佳超參數組合: {grid_search.best_params_}")
    print(f"在驗證集上最佳門檻: {best_threshold:.2f} (macro F1={best_val_score:.4f})")

    evaluation = evaluate_model(best_rf, X_test, y_test, best_threshold)
    print(f"測試集準確率 (Accuracy): {evaluation['accuracy']:.4f}")
    print(f"測試集 ROC AUC: {evaluation['roc_auc']:.4f}")
    print("混淆矩陣：")
    print(evaluation['confusion_matrix'])
    print("\n分類報告：")
    print(evaluation['classification_report'])

    save_model_artifact(best_rf, FEATURE_COLS, best_threshold, model_path=model_path)

    return {
        'model': best_rf,
        'threshold': best_threshold,
        'feature_cols': FEATURE_COLS,
        'evaluation': evaluation,
        'grid_search': grid_search
    }


if __name__ == "__main__":
    train_and_save_model()

grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    cv=tscv,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# 在 Validation 集尋找最佳預測機率門檻 (避免 Test 集資料洩漏)
y_val_probs = best_rf.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_val_score = -1

for threshold in np.linspace(0.3, 0.7, 41):
    preds = (y_val_probs >= threshold).astype(int)
    report = classification_report(y_val, preds, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    if macro_f1 > best_val_score:
        best_val_score = macro_f1
        best_threshold = threshold

# 最終在全新的 Test 集上進行盲測評估
y_test_probs = best_rf.predict_proba(X_test)[:, 1]
final_preds = (y_test_probs >= best_threshold).astype(int)

print("\n=== 修復前瞻偏誤後的最佳化隨機森林模型評估結果 ===")
print(f"最佳超參數組合: {grid_search.best_params_}")
print(f"於Validation集確定的最佳預測門檻: {best_threshold:.2f}")
print(f"測試集準確率 (Accuracy): {accuracy_score(y_test, final_preds):.4f}")
print(f"測試集 ROC AUC Score: {roc_auc_score(y_test, y_test_probs):.4f}\n")
print("混淆矩陣 (Confusion Matrix):")
print(confusion_matrix(y_test, final_preds))
print("\n詳細分類報告:")
print(classification_report(y_test, final_preds, target_names=['跌 (0)', '漲 (1)']))

# ======================================================
# 11. 特徵重要性分析 (Feature Importances)
# ======================================================
importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='darkgoldenrod')
plt.title("Feature Importances in Random Forest (Leak-Free Model)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()