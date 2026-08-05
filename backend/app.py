import gradio as gr
import requests
import pandas as pd
import yfinance as yf
import plotly.express as px

# 呼叫 FastAPI 端點
def get_prediction():
    api_url = "http://127.0.0.1:8000/api/v1/predict"
    try:
        res = requests.post(api_url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return (
                f"${data['latest_price']:,} USD",
                f"{data['latest_dxy']}",
                f"{data['prob_up']}% (門檻 {data['threshold']})",
                data['direction'],
                data['ai_report']
            )
        else:
            return "Error", "Error", "Error", "伺服器異常", f"FastAPI 回傳錯誤: {res.text}"
    except Exception as e:
        return "連線失敗", "連線失敗", "連線失敗", "離線", f"無法連線至 FastAPI 後端 ({api_url})。請確認 main.py 已啟動。"

# 畫出近 60 日黃金趨勢折線圖
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

# Gradio Dashboard 介面
with gr.Blocks(theme=gr.themes.Monochrome(), title="GoldMind 金價預測與 AI 投資助理") as demo:
    gr.Markdown("# 🏆 GoldMind 智慧金價預測與投資助理 (MVP)")
    gr.Markdown("整合 **FastAPI + 隨機森林無偏誤模型 + 生成式 AI 診斷** 之金融儀表板")
    
    with gr.Row():
        with gr.Column(scale=2):
            chart_output = gr.Plot(value=draw_gold_chart())
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 即時 ML 預測開關")
            btn_predict = gr.Button("🚀 抓取即時數據並執行 AI 分析", variant="primary")
            
            out_price = gr.Textbox(label="當前黃金收盤價 (Close)", interactive=False)
            out_dxy = gr.Textbox(label="當前美元指數 (DXY)", interactive=False)
            out_prob = gr.Textbox(label="模型看多機率 (Probability)", interactive=False)
            out_dir = gr.Textbox(label="明日預測走勢 (Direction)", interactive=False)

    gr.Markdown("---")
    out_report = gr.Markdown("💡 **點擊【執行 AI 分析】按鈕，以取得最新智慧診斷報告。**")

    # 事件綁定
    btn_predict.click(
        fn=get_prediction,
        inputs=[],
        outputs=[out_price, out_dxy, out_prob, out_dir, out_report]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)