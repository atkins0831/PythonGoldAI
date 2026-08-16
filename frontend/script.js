const API_BASE_URL = 'https://pythongoldai.onrender.com';

const ctx = document.getElementById('goldChart').getContext('2d');

let currentMode = 'daily';
let currentDays = '30';
let latestDailyApiData = null;

// 初始化 Chart.js 圖表實例
const goldChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: []
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        display: true,
        labels: {
          color: '#cbd5e1',
          font: { size: 13, weight: '600' }
        }
      },
      tooltip: {
        backgroundColor: '#131c2e',
        titleColor: '#ffffff',
        bodyColor: '#cbd5e1',
        borderColor: '#f59e0b',
        borderWidth: 1,
        padding: 10
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#94a3b8' }
      },
      y: {
        grace: '5%',
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: {
          color: '#94a3b8',
          callback: function (value) {
            return '$' + value.toLocaleString();
          }
        }
      }
    }
  }
});

// DOM 元素引用
const tabDailyBtn = document.getElementById('tabDailyBtn');
const tabFutureBtn = document.getElementById('tabFutureBtn');
const maSelectorCard = document.getElementById('maSelectorCard');
const daysSelectorCard = document.getElementById('daysSelectorCard');
const chartTitle = document.getElementById('chartTitle');
const panelTitle = document.getElementById('panelTitle');
const fetchDataBtn = document.getElementById('fetchDataBtn');

const label1 = document.getElementById('label1');
const label2 = document.getElementById('label2');
const label3 = document.getElementById('label3');
const label4 = document.getElementById('label4');

const val1 = document.getElementById('val1');
const val2 = document.getElementById('val2');
const val3 = document.getElementById('val3');
const val4 = document.getElementById('val4');

const reportTitle = document.getElementById('reportTitle');
const rptPrice = document.getElementById('rptPrice');
const rptDxy = document.getElementById('rptDxy');
const rptProb = document.getElementById('rptProb');
const rptThreshold = document.getElementById('rptThreshold');
const rptDirection = document.getElementById('rptDirection');

// ======================================================
// 1. API 串接：取得歷史走勢與 MA 數據 (/api/v1/chart/history)
// ======================================================
async function fetchAndRenderHistoryChart() {
  try {
    const resp = await fetch(`${API_BASE_URL}/api/v1/chart/history?limit=60`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const resData = await resp.json();

    const labels = resData.data.map(item => item.date);
    const closePrices = resData.data.map(item => item.close);

    const datasets = [{
      label: '黃金歷史收盤價',
      data: closePrices,
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245, 158, 11, 0.1)',
      borderWidth: 2.5,
      pointRadius: 2,
      pointHoverRadius: 5,
      tension: 0.2,
      fill: false
    }];

    const maColorMap = {
      '5': { key: 'ma5', label: '5日均綫 (MA5)', color: '#38bdf8' },
      '10': { key: 'ma10', label: '10日均綫 (MA10)', color: '#a855f7' },
      '20': { key: 'ma20', label: '20日均綫 (MA20)', color: '#ec4899' },
      '60': { key: 'ma60', label: '60日均綫 (MA60)', color: '#22c55e' }
    };

    document.querySelectorAll('input[name="maOption"]').forEach(checkbox => {
      const parentLabel = checkbox.closest('.day-radio');
      if (checkbox.checked) {
        parentLabel.classList.add('active');
        const config = maColorMap[checkbox.value];
        if (config) {
          datasets.push({
            label: config.label,
            data: resData.data.map(item => item[config.key]),
            borderColor: config.color,
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.2,
            fill: false
          });
        }
      } else {
        parentLabel.classList.remove('active');
      }
    });

    goldChart.data.labels = labels;
    goldChart.data.datasets = datasets;
    goldChart.options.plugins.legend.display = true;
    goldChart.update();

  } catch (err) {
    console.error('取得歷史圖表資料失敗:', err);
  }
}

// ======================================================
// 2. API 串接：執行每日 ML 預測 (/api/v1/predict)
// ======================================================
async function fetchPrediction() {
  fetchDataBtn.disabled = true;
  const originalText = fetchDataBtn.innerText;
  fetchDataBtn.innerText = '正在連線至伺服器...';

  try {
    const resp = await fetch(`${API_BASE_URL}/api/v1/predict`, { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    latestDailyApiData = data;
    applyDailyData(data);
    fetchDataBtn.innerText = '資料已更新!';
  } catch (err) {
    console.error('每日預測 API 串接失敗:', err);
    fetchDataBtn.innerText = '連線失敗，請稍後重試';
  } finally {
    setTimeout(() => {
      fetchDataBtn.disabled = false;
      fetchDataBtn.innerText = originalText;
    }, 1200);
  }
}

function applyDailyData(data) {
  label1.innerText = '當前黃金收盤價 (Close)';
  val1.innerText = `$${data.latest_price.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;

  label2.innerText = '當前美元指數 (DXY)';
  val2.innerText = Number(data.latest_dxy).toFixed(2);

  label3.innerText = '模型看多機率';
  val3.innerText = `${data.prob_up}%`;

  label4.innerText = '明日預測走勢';
  val4.innerText = data.direction;

  reportTitle.innerText = `GoldMind AI 語意診斷報告 (${data.date})`;
  rptPrice.innerText = `$${data.latest_price.toLocaleString('en-US', { minimumFractionDigits: 2 })} USD`;
  rptDxy.innerText = Number(data.latest_dxy).toFixed(2);
  rptProb.innerText = `${data.prob_up}%`;
  rptThreshold.innerText = Number(data.threshold).toFixed(2);
  rptDirection.innerText = `【${data.direction}】`;
}

// ======================================================
// 3. API 串接：取得未來推演圖表與右側估值指標 (/api/v1/chart/forecast & /api/v1/forecast/simulation)
// ======================================================
async function fetchAndRenderForecastChart(days) {
  chartTitle.innerText = `黃金未來 ${days} 日 AI 走勢推演與情境模擬`;
  label1.innerText = `${days} 日目標預估均價 (Baseline)`;
  label2.innerText = '樂觀情境目標價 (Bull)';
  label3.innerText = '悲觀情境目標價 (Bear)';
  label4.innerText = '樂/悲觀區間偏誤率 (Bias)';

  fetchDataBtn.disabled = true;
  fetchDataBtn.innerText = '模擬計算中...';

  try {
    // 呼叫估值 API 取得右側面板數據
    const simResp = await fetch(`${API_BASE_URL}/api/v1/forecast/simulation?days=${days}`, { method: 'POST' });
    if (simResp.ok) {
      const simData = await simResp.json();
      val1.innerText = simData.target_baseline_str;
      val2.innerText = simData.target_bull_str;
      val3.innerText = simData.target_bear_str;
      val4.innerText = simData.bias_range_str;
    }

    // 呼叫圖表 API 取得推演折線數據
    const resp = await fetch(`${API_BASE_URL}/api/v1/chart/forecast?days=${days}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    reportTitle.innerText = `GoldMind AI 語意診斷報告 (未來 ${days} 日 Monte Carlo 推演模式)`;
    rptDirection.innerText = `【趨勢推演看多 (Baseline: ${val1.innerText})】`;

    const labels = data.chart_data.map(item => item.date);
    const baselineData = data.chart_data.map(item => item.baseline);
    const bullData = data.chart_data.map(item => item.bull);
    const bearData = data.chart_data.map(item => item.bear);

    goldChart.data.labels = labels;
    goldChart.data.datasets = [
      {
        label: '樂觀情境 (Bull)',
        data: bullData,
        borderColor: '#22c55e',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 3,
        pointBackgroundColor: '#22c55e',
        tension: 0.2,
        fill: false
      },
      {
        label: 'AI 基線推演 (Baseline)',
        data: baselineData,
        borderColor: '#f59e0b',
        borderWidth: 3,
        pointRadius: 4,
        pointBackgroundColor: '#f59e0b',
        tension: 0.2,
        fill: false
      },
      {
        label: '悲觀情境 (Bear)',
        data: bearData,
        borderColor: '#ef4444',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 3,
        pointBackgroundColor: '#ef4444',
        tension: 0.2,
        fill: false
      }
    ];
    goldChart.options.plugins.legend.display = true;
    goldChart.update();

  } catch (err) {
    console.error('取得未來推演圖表資料失敗:', err);
  } finally {
    fetchDataBtn.disabled = false;
    fetchDataBtn.innerText = '重新執行 Monte Carlo 模擬';
  }
}

// ======================================================
// 4. 事件監聽 (MA 勾選、天數選擇、頁籤切換)
// ======================================================
document.querySelectorAll('input[name="maOption"]').forEach(checkbox => {
  checkbox.addEventListener('change', () => {
    if (currentMode === 'daily') fetchAndRenderHistoryChart();
  });
});

document.querySelectorAll('input[name="forecastDays"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    currentDays = e.target.value;
    document.querySelectorAll('input[name="forecastDays"]').forEach(r => r.closest('.day-radio').classList.remove('active'));
    e.target.closest('.day-radio').classList.add('active');
    if (currentMode === 'future') fetchAndRenderForecastChart(currentDays);
  });
});

tabDailyBtn.addEventListener('click', () => {
  if (currentMode === 'daily') return;
  currentMode = 'daily';
  tabDailyBtn.classList.add('active');
  tabFutureBtn.classList.remove('active');
  maSelectorCard.style.display = 'flex';
  daysSelectorCard.style.display = 'none';
  chartTitle.innerText = '黃金 (Gold Futures) 近 60 日歷史走勢圖';
  panelTitle.innerText = '即時 ML 預測開關';
  fetchDataBtn.innerText = '抓取即時數據並執行 AI 分析';

  if (latestDailyApiData) {
    applyDailyData(latestDailyApiData);
  }
  fetchAndRenderHistoryChart();
});

tabFutureBtn.addEventListener('click', () => {
  if (currentMode === 'future') return;
  currentMode = 'future';
  tabFutureBtn.classList.add('active');
  tabDailyBtn.classList.remove('active');
  maSelectorCard.style.display = 'none';
  daysSelectorCard.style.display = 'flex';
  panelTitle.innerText = '未來推演參數與估值';
  fetchDataBtn.innerText = '重新執行 Monte Carlo 模擬';
  fetchAndRenderForecastChart(currentDays);
});

fetchDataBtn.addEventListener('click', () => {
  if (currentMode === 'daily') {
    fetchPrediction();
  } else {
    fetchAndRenderForecastChart(currentDays);
  }
});

// 頁面初次載入時連動 API 繪圖與即時預測
fetchAndRenderHistoryChart();
fetchPrediction();