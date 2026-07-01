// State Management
let priceChart = null;

// DOM Elements
const btcPriceEl = document.getElementById('btc-price');
const btcVolumeEl = document.getElementById('btc-volume');
const accuracyEl = document.getElementById('model-accuracy');
const modelStateEl = document.getElementById('model-state');
const alertContainer = document.getElementById('alert-container');
const logsConsole = document.getElementById('terminal-logs');

const btnIngest = document.getElementById('btn-ingest');
const btnTrain = document.getElementById('btn-train');
const btnDrift = document.getElementById('btn-drift');

// Tabs Controller
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
    });
});

// Logs helper
function logMessage(sender, text) {
    const timestamp = new Date().toLocaleTimeString();
    logsConsole.innerHTML += `\n[${timestamp}] [${sender}] ${text}`;
    logsConsole.scrollTop = logsConsole.scrollHeight;
}

// Format Currency Utility
function formatCurrency(value, fractionDigits = 2) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits
    }).format(value);
}

// ----------------- API CALLS & RENDERING -----------------

// Fetch and load dashboard telemetry
async function loadDashboard() {
    try {
        // 1. Fetch Metrics
        const responseMetrics = await fetch('/api/metrics');
        if (responseMetrics.ok) {
            const metrics = await responseMetrics.json();
            
            // Populate Metrics Cards
            btcPriceEl.textContent = formatCurrency(metrics.latest_price);
            btcVolumeEl.textContent = formatCurrency(metrics.latest_volume, 0);
            accuracyEl.textContent = `${(metrics.model_accuracy * 100).toFixed(2)}%`;
            
            // Format state color and title
            const action = metrics.action_taken;
            let statusText = "STABLE";
            let statusColor = "#00ffcc";
            
            if (action.includes("Retrained")) {
                statusText = "RETRAINED";
                statusColor = "#ff4b4b";
                modelStateEl.style.color = statusColor;
            } else if (action.includes("Initialized")) {
                statusText = "INITIALIZED";
                statusColor = "#45f3ff";
            }
            modelStateEl.textContent = statusText;
            modelStateEl.style.color = statusColor;
            
            // Render Status Alerts
            renderAlertBanner(metrics);
        }
        
        // 2. Fetch Data & Build Chart
        const responseData = await fetch('/api/data');
        if (responseData.ok) {
            const dataList = await responseData.json();
            renderPriceChart(dataList);
            populateIngestedLedger(dataList);
        }
        
        // 3. Fetch Prediction Inference
        const responsePredict = await fetch('/api/predict');
        if (responsePredict.ok) {
            const predictions = await responsePredict.json();
            renderPredictions(predictions);
        }
        
    } catch (err) {
        console.error("Error loading dashboard data:", err);
        logMessage("Error", `Failed to refresh dashboard. Server may be starting. Details: ${err.message}`);
    }
}

// Custom alert generator
function renderAlertBanner(metrics) {
    const action = metrics.action_taken;
    const accuracy = metrics.model_accuracy;
    const date = metrics.date;
    
    let html = '';
    
    if (action.includes("Retrained")) {
        html = `
        <div class="alert-card error">
            <div class="alert-icon"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div>
                <h4 class="alert-title">Auto-Retrain Event Triggered</h4>
                <p class="alert-body">
                    Model validation accuracy dropped below <b>65%</b> due to data drift. The pipeline completed an automated training run on updated data at <b>${date}</b>, saving updated weights to production.
                </p>
            </div>
        </div>
        `;
    } else if (accuracy < 0.65) {
        html = `
        <div class="alert-card warning">
            <div class="alert-icon"><i class="fa-solid fa-circle-exclamation"></i></div>
            <div>
                <h4 class="alert-title">Performance Margin Warning</h4>
                <p class="alert-body">
                    Validation accuracy has dropped close to the retraining trigger threshold. High probability of incoming data drift.
                </p>
            </div>
        </div>
        `;
    } else {
        html = `
        <div class="alert-card success">
            <div class="alert-icon"><i class="fa-solid fa-circle-check"></i></div>
            <div>
                <h4 class="alert-title">Pipeline Status Stable</h4>
                <p class="alert-body">
                    All systems operational. Validation accuracy is safely above the 65% limit. Automatic daily cron checks are passing.
                </p>
            </div>
        </div>
        `;
    }
    
    alertContainer.innerHTML = html;
}

// Render market chart using ApexCharts
function renderPriceChart(data) {
    const dates = data.map(item => item.date);
    const prices = data.map(item => item.price);
    const volumes = data.map(item => item.volume);
    
    const options = {
        series: [{
            name: 'Close Price (USD)',
            type: 'line',
            data: prices
        }, {
            name: 'Trading Volume (USD)',
            type: 'column',
            data: volumes
        }],
        chart: {
            height: 380,
            type: 'line',
            toolbar: { show: false },
            background: 'transparent'
        },
        stroke: {
            width: [4, 0],
            curve: 'smooth'
        },
        plotOptions: {
            bar: {
                columnWidth: '50%',
                opacity: 0.15
            }
        },
        colors: ['#66fcf1', 'rgba(102, 252, 241, 0.25)'],
        dataLabels: { enabled: false },
        theme: { mode: 'dark' },
        grid: {
            borderColor: 'rgba(255,255,255,0.05)'
        },
        xaxis: {
            categories: dates,
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: [{
            title: {
                text: 'Price (USD)',
                style: { color: '#66fcf1' }
            },
            labels: {
                style: { colors: '#66fcf1' }
            }
        }, {
            opposite: true,
            title: {
                text: 'Volume (USD)',
                style: { color: '#8a99ad' }
            },
            labels: {
                style: { colors: '#8a99ad' }
            }
        }],
        tooltip: {
            shared: true,
            intersect: false,
            y: {
                formatter: function (y, { seriesIndex }) {
                    if (typeof y !== "undefined") {
                        return seriesIndex === 0 ? formatCurrency(y) : formatCurrency(y, 0);
                    }
                    return y;
                }
            }
        }
    };
    
    if (priceChart) {
        priceChart.destroy();
    }
    
    priceChart = new ApexCharts(document.querySelector("#chart-price"), options);
    priceChart.render();
}

// Populate the ledger data grid
function populateIngestedLedger(data) {
    const tableBody = document.getElementById('data-table-body');
    tableBody.innerHTML = '';
    
    // Sort descending by date
    const sortedData = [...data].sort((a, b) => new Date(b.date) - new Date(a.date));
    
    sortedData.forEach(row => {
        const trend = row.target === 1 ? '<span style="color:#00ffcc;">🟢 Price Up</span>' : '<span style="color:#ff4b4b;">🔴 Price Down</span>';
        const volatility = row.volatility ? `${(row.volatility * 100).toFixed(2)}%` : '0.00%';
        
        tableBody.innerHTML += `
        <tr>
            <td>${row.date}</td>
            <td>${formatCurrency(row.price)}</td>
            <td>${formatCurrency(row.volume, 0)}</td>
            <td>${volatility}</td>
            <td>${trend}</td>
        </tr>
        `;
    });
}

// Populate predictions and tomorrow's forecast
function renderPredictions(predictions) {
    const inferBody = document.getElementById('infer-table-body');
    inferBody.innerHTML = '';
    
    // Tomorrow's Forecast Card
    const latest = predictions[predictions.length - 1];
    const prob = latest.probability;
    const forecastClass = prob > 0.5 ? 'UP TREND' : 'DOWN TREND';
    const forecastColor = prob > 0.5 ? '#00ffcc' : '#ff4b4b';
    const confidence = prob > 0.5 ? prob : (1.0 - prob);
    
    const forecastCard = document.getElementById('forecast-card');
    forecastCard.style.borderColor = forecastColor;
    
    document.getElementById('forecast-value').textContent = forecastClass;
    document.getElementById('forecast-value').style.color = forecastColor;
    document.getElementById('forecast-conf').innerHTML = `Confidence: <b>${(confidence * 100).toFixed(2)}%</b>`;
    
    // Populate Inference Grid Table (sort descending)
    const sortedPredict = [...predictions].sort((a, b) => new Date(b.date) - new Date(a.date));
    
    sortedPredict.forEach(row => {
        const inferred = row.prediction_class === 1 ? '<span style="color:#00ffcc;">🟢 Up</span>' : '<span style="color:#ff4b4b;">🔴 Down</span>';
        const actual = row.target === 1 ? '<span style="color:#00ffcc;">🟢 Up</span>' : '<span style="color:#ff4b4b;">🔴 Down</span>';
        const correct = row.prediction_class === row.target ? '<span class="validation-outcome outcome-correct">✅ Correct</span>' : '<span class="validation-outcome outcome-incorrect">❌ Incorrect</span>';
        const confidenceVal = row.probability > 0.5 ? row.probability : (1.0 - row.probability);
        
        inferBody.innerHTML += `
        <tr>
            <td>${row.date}</td>
            <td>${formatCurrency(row.price)}</td>
            <td>${inferred}</td>
            <td>${(confidenceVal * 100).toFixed(2)}%</td>
            <td>${correct}</td>
        </tr>
        `;
    });
}

// ----------------- CONTROLS HANDLERS -----------------

btnIngest.addEventListener('click', async () => {
    btnIngest.disabled = true;
    logMessage("Pipeline", "Executing data_ingestion.py to fetch CoinGecko market ticks...");
    
    try {
        const response = await fetch('/api/run-ingestion', { method: 'POST' });
        const result = await response.json();
        
        if (result.status === 'success') {
            logMessage("Pipeline", `Ingestion executed successfully.\nStdout:\n${result.stdout}`);
            await loadDashboard();
        } else {
            logMessage("Pipeline", `Ingestion failed.\nStderr:\n${result.stderr}`);
        }
    } catch (err) {
        logMessage("Error", `Ingestion failed to run: ${err.message}`);
    } finally {
        btnIngest.disabled = false;
    }
});

btnTrain.addEventListener('click', async () => {
    btnTrain.disabled = true;
    logMessage("Pipeline", "Executing model_training.py to retrain Keras model...");
    
    try {
        const response = await fetch('/api/run-training', { method: 'POST' });
        const result = await response.json();
        
        if (result.status === 'success') {
            logMessage("Pipeline", `Retraining completed successfully.\nStdout:\n${result.stdout}`);
            await loadDashboard();
        } else {
            logMessage("Pipeline", `Retraining failed.\nStderr:\n${result.stderr}`);
        }
    } catch (err) {
        logMessage("Error", `Retraining failed to run: ${err.message}`);
    } finally {
        btnTrain.disabled = false;
    }
});

btnDrift.addEventListener('click', async () => {
    btnDrift.disabled = true;
    logMessage("Pipeline", "Simulating Data Drift Event (inverting labels)...");
    
    try {
        const response = await fetch('/api/simulate-drift', { method: 'POST' });
        const result = await response.json();
        
        if (result.status === 'success') {
            logMessage("Pipeline", `Drift simulated successfully and retraining completed.\nStdout:\n${result.stdout}`);
            await loadDashboard();
        } else {
            logMessage("Pipeline", `Drift simulation failed.\nStderr:\n${result.stderr}`);
        }
    } catch (err) {
        logMessage("Error", `Drift simulation failed: ${err.message}`);
    } finally {
        btnDrift.disabled = false;
    }
});

// ----------------- BOOTSTRAP -----------------
logMessage("System", "Dashboard initialized. Fetching telemetry metrics...");
loadDashboard();
// Automatically poll metrics every 30 seconds
setInterval(loadDashboard, 30000);
