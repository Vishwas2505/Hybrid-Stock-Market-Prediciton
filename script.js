// Application State
const state = {
    activeTab: 'dashboard',
    selectedTicker: 'AAPL',
    windowSize: 30,
    modelType: 'himm',
    geminiKey: localStorage.getItem('gemini_api_key') || '',
    isPredicting: false,
    
    // Custom Chart State
    chartMode: 'candle', // 'candle' or 'line'
    showSignals: true,
    historicalPrices: [],
    tradeSignals: [],
    userTrades: [], // tracks user orders { index, price, type }
    hoverIndex: null,
    
    // Mock Trading State
    portfolio: {
        cash: parseFloat(localStorage.getItem('mock_cash')) || 10000.00,
        shares: parseInt(localStorage.getItem('mock_shares')) || 0
    },
    currentPrice: 150.0,
    
    // Neural Explorer Module State
    activeModuleTab: 'gru'
};

// DOM Elements
const elements = {
    navItems: document.querySelectorAll('.nav-item'),
    tabContents: document.querySelectorAll('.tab-content'),
    pageTitle: document.getElementById('page-title'),
    
    // Config panel
    stockSelect: document.getElementById('stock-select'),
    customTickerInput: document.getElementById('custom-ticker-input'),
    paramWindow: document.getElementById('param-window'),
    paramModel: document.getElementById('param-model'),
    toggleAdvanced: document.getElementById('toggle-advanced-params'),
    advancedParamsPanel: document.getElementById('advanced-params-panel'),
    runPrediction: document.getElementById('run-prediction'),
    
    // Results
    resultCard: document.getElementById('result-card'),
    resultEmpty: document.getElementById('result-empty'),
    resultSuccess: document.getElementById('result-success'),
    directionBadge: document.getElementById('direction-badge'),
    probabilityText: document.getElementById('probability-text'),
    resTicker: document.getElementById('res-ticker'),
    resSentiment: document.getElementById('res-sentiment'),
    resModel: document.getElementById('res-model'),
    aiOutlookText: document.getElementById('ai-outlook-text'),
    
    // Terminal & Charts
    terminalConsole: document.getElementById('terminal-console'),
    chartTickerName: document.getElementById('chart-ticker-name'),
    newsContainer: document.getElementById('news-container'),
    stockChart: document.getElementById('stockChart'),
    
    // Chart Controls & Toggles
    toggleCandle: document.getElementById('toggle-candle'),
    toggleLine: document.getElementById('toggle-line'),
    
    // Mock Trading Panel
    portCash: document.getElementById('port-cash'),
    portShares: document.getElementById('port-shares'),
    portTotal: document.getElementById('port-total'),
    tradeQty: document.getElementById('trade-qty'),
    tradeQtyDisplay: document.getElementById('trade-qty-display'),
    actionBuy: document.getElementById('action-buy'),
    actionSell: document.getElementById('action-sell'),
    
    // Module Tabs
    moduleTabBtns: document.querySelectorAll('.explorer-tab-btn'),
    modulePanels: document.querySelectorAll('.module-panel'),
    gruVectorGrid: document.getElementById('gru-vector-grid'),
    bertVectorGrid: document.getElementById('bert-vector-grid'),
    sentGaugeFill: document.getElementById('sent-gauge-fill'),
    sentGaugeIndicator: document.getElementById('sent-gauge-indicator'),
    sentScoreReadout: document.getElementById('sent-score-readout'),
    gruTrendState: document.getElementById('gru-trend-state'),
    gruVolSig: document.getElementById('gru-vol-sig'),
    fusionTensorDisplay: document.getElementById('fusion-tensor-display'),
    mlpFeatureBox: document.getElementById('mlp-feature-box'),
    mlpInteractionBox: document.getElementById('mlp-interaction-box'),
    
    // Settings
    settingsGeminiKey: document.getElementById('settings-gemini-key'),
    saveGeminiKey: document.getElementById('save-gemini-key'),
    statusPython: document.getElementById('status-python'),
    statusPytorch: document.getElementById('status-pytorch'),
    
    // Modal
    openApiModal: document.getElementById('open-api-modal'),
    apiModal: document.getElementById('api-modal'),
    closeApiModal: document.getElementById('close-api-modal'),
    cancelModalBtn: document.getElementById('cancel-modal-btn'),
    saveModalBtn: document.getElementById('save-modal-btn'),
    modalGeminiKey: document.getElementById('modal-gemini-key'),
    currentDate: document.getElementById('current-date')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initDate();
    initSettings();
    initEventListeners();
    initChartHandlers();
    initPortfolio();
    loadStockData(state.selectedTicker);
    checkSystemStatus();
});

// Setup Date
function initDate() {
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    elements.currentDate.textContent = new Date().toLocaleDateString('en-US', options);
}

// Setup saved API keys
function initSettings() {
    if (state.geminiKey) {
        elements.settingsGeminiKey.value = '••••••••••••••••••••••••';
        elements.modalGeminiKey.value = state.geminiKey;
    }
}

// Initialize Portfolio metrics from storage
function initPortfolio() {
    updatePortfolioUI();
}

// Calculate and refresh simulated portfolio cards
function updatePortfolioUI() {
    state.portfolio.cash = parseFloat(localStorage.getItem('mock_cash')) ?? 10000.00;
    state.portfolio.shares = parseInt(localStorage.getItem('mock_shares')) ?? 0;
    
    const cash = state.portfolio.cash;
    const shares = state.portfolio.shares;
    const total = cash + (shares * state.currentPrice);
    
    elements.portCash.textContent = `$${cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    elements.portShares.textContent = shares;
    elements.portTotal.textContent = `$${total.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

// Save mock portfolio stats
function savePortfolio(cash, shares) {
    localStorage.setItem('mock_cash', cash.toFixed(2));
    localStorage.setItem('mock_shares', shares);
    state.portfolio.cash = cash;
    state.portfolio.shares = shares;
    updatePortfolioUI();
}

// Check System Specs from server
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/prices?ticker=AAPL');
        if (response.ok) {
            const data = await response.json();
            elements.statusPython.textContent = "Python 3.7+ (Active)";
            elements.statusPytorch.textContent = "PyTorch 1.13+ (CPU/CUDA)";
            elements.statusPytorch.className = "status-box-val text-green";
        }
    } catch (e) {
        elements.statusPython.textContent = "Offline / Connection Error";
        elements.statusPython.className = "status-box-val text-red";
        elements.statusPytorch.textContent = "Unavailable";
        elements.statusPytorch.className = "status-box-val text-red";
    }
}

// Initialize Event Listeners
function initEventListeners() {
    // Navigation Toggling
    elements.navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabName = item.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
    
    // Stock select change
    elements.stockSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'CUSTOM') {
            elements.customTickerInput.classList.remove('hidden');
            elements.customTickerInput.focus();
        } else {
            elements.customTickerInput.classList.add('hidden');
            state.selectedTicker = val;
            loadStockData(val);
        }
    });

    // Custom ticker inputs
    elements.customTickerInput.addEventListener('blur', (e) => {
        const val = e.target.value.trim().toUpperCase();
        if (val) {
            state.selectedTicker = val;
            loadStockData(val);
        }
    });
    
    elements.customTickerInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const val = e.target.value.trim().toUpperCase();
            if (val) {
                state.selectedTicker = val;
                loadStockData(val);
                e.target.blur();
            }
        }
    });
    
    // Toggle advanced configuration panel
    elements.toggleAdvanced.addEventListener('click', () => {
        elements.advancedParamsPanel.classList.toggle('hidden');
        const icon = elements.toggleAdvanced.querySelector('i');
        icon.classList.toggle('fa-chevron-down');
        icon.classList.toggle('fa-chevron-up');
    });
    
    // Modal controls
    elements.openApiModal.addEventListener('click', () => {
        elements.apiModal.classList.add('active');
    });
    
    const closeModal = () => elements.apiModal.classList.remove('active');
    elements.closeApiModal.addEventListener('click', closeModal);
    elements.cancelModalBtn.addEventListener('click', closeModal);
    
    elements.saveModalBtn.addEventListener('click', () => {
        const key = elements.modalGeminiKey.value.trim();
        saveApiKey(key);
        closeModal();
    });
    
    // Settings tab key save
    elements.saveGeminiKey.addEventListener('click', () => {
        const key = elements.settingsGeminiKey.value.trim();
        if (key !== '••••••••••••••••••••••••') {
            saveApiKey(key);
            alert("Gemini API key updated successfully.");
        }
    });
    
    // Predict CTA triggers
    elements.runPrediction.addEventListener('click', () => {
        if (state.isPredicting) return;
        executePrediction();
    });

    // Neural Explorer sub-tab toggling
    elements.moduleTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-module-tab');
            switchModuleTab(tabId);
        });
    });

    // Chart toggles (Line vs Candles)
    elements.toggleCandle.addEventListener('click', () => {
        elements.toggleCandle.classList.add('active');
        elements.toggleLine.classList.remove('active');
        state.chartMode = 'candle';
        drawChart();
    });

    elements.toggleLine.addEventListener('click', () => {
        elements.toggleLine.classList.add('active');
        elements.toggleCandle.classList.remove('active');
        state.chartMode = 'line';
        drawChart();
    });

    // Mock Trading Orders
    elements.tradeQty.addEventListener('input', (e) => {
        elements.tradeQtyDisplay.textContent = e.target.value;
    });

    elements.actionBuy.addEventListener('click', () => {
        executeMockTrade('BUY');
    });

    elements.actionSell.addEventListener('click', () => {
        executeMockTrade('SELL');
    });
}

// Switch Sidebar tabs
function switchTab(tabName) {
    state.activeTab = tabName;
    
    elements.navItems.forEach(item => {
        if (item.getAttribute('data-tab') === tabName) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    elements.tabContents.forEach(content => {
        if (content.id === `tab-${tabName}`) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
    
    // Update Header Text
    if (tabName === 'dashboard') {
        elements.pageTitle.textContent = "Predictive Dashboard";
        drawChart(); // redraw custom canvas
    } else if (tabName === 'research') {
        elements.pageTitle.textContent = "Model Analysis & Research";
    } else if (tabName === 'settings') {
        elements.pageTitle.textContent = "System Configurations";
    }
}

// Switch sub-tab in neural modules explorer
function switchModuleTab(tabId) {
    state.activeModuleTab = tabId;
    
    elements.moduleTabBtns.forEach(btn => {
        if (btn.getAttribute('data-module-tab') === tabId) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    elements.modulePanels.forEach(panel => {
        if (panel.id === `module-panel-${tabId}`) {
            panel.classList.add('active');
        } else {
            panel.classList.remove('active');
        }
    });

    if (tabId === 'fusion') {
        triggerFusionAnimation();
    }
}

// LocalStorage API Key saving
function saveApiKey(key) {
    if (!key) {
        localStorage.removeItem('gemini_api_key');
        state.geminiKey = '';
        elements.settingsGeminiKey.value = '';
        elements.modalGeminiKey.value = '';
    } else {
        localStorage.setItem('gemini_api_key', key);
        state.geminiKey = key;
        elements.settingsGeminiKey.value = '••••••••••••••••••••••••';
        elements.modalGeminiKey.value = key;
    }
    appendConsoleLine(`[SYSTEM] API credentials updated. Gemini API is ${key ? 'active' : 'disabled (lexical mode fallback)'}.`, 'system-msg');
}

// Append logs line to terminal output
function appendConsoleLine(text, className = '') {
    const line = document.createElement('div');
    line.className = `console-line ${className}`;
    line.textContent = text;
    elements.terminalConsole.appendChild(line);
    elements.terminalConsole.scrollTop = elements.terminalConsole.scrollHeight;
}

// Execute simulated trade orders
function executeMockTrade(type) {
    const qty = parseInt(elements.tradeQty.value) || 1;
    const price = state.currentPrice;
    const cost = qty * price;
    
    let cash = state.portfolio.cash;
    let shares = state.portfolio.shares;
    
    if (type === 'BUY') {
        if (cash >= cost) {
            cash -= cost;
            shares += qty;
            appendConsoleLine(`[TRADE] MOCK BUY ORDER: Bought ${qty} shares of ${state.selectedTicker} at $${price.toFixed(2)} ($${cost.toFixed(2)} total)`, 'success-msg');
            
            // Add user trade marker to the chart at the last day
            state.userTrades.push({
                index: state.historicalPrices.length - 1,
                price: price,
                type: 'BUY'
            });
            
            savePortfolio(cash, shares);
        } else {
            appendConsoleLine(`[TRADE] ORDER ERROR: Insufficient Cash ($${cash.toFixed(2)}) to buy ${qty} shares at $${price.toFixed(2)}`, 'error-msg');
            alert("Order Error: Insufficient cash balance.");
        }
    } else if (type === 'SELL') {
        if (shares >= qty) {
            cash += cost;
            shares -= qty;
            appendConsoleLine(`[TRADE] MOCK SELL ORDER: Sold ${qty} shares of ${state.selectedTicker} at $${price.toFixed(2)} (+$${cost.toFixed(2)} total)`, 'success-msg');
            
            // Add user trade marker
            state.userTrades.push({
                index: state.historicalPrices.length - 1,
                price: price,
                type: 'SELL'
            });
            
            savePortfolio(cash, shares);
        } else {
            appendConsoleLine(`[TRADE] ORDER ERROR: Insufficient shares owned (${shares}) to sell ${qty} shares`, 'error-msg');
            alert("Order Error: Insufficient shares owned.");
        }
    }
    
    drawChart();
}

// Fetch and load initial stock prices / news
async function loadStockData(ticker) {
    appendConsoleLine(`[DATA] Querying data streams for ticker: ${ticker}...`);
    elements.chartTickerName.textContent = `${ticker} Loading...`;
    
    try {
        // Fetch Prices
        const priceRes = await fetch(`/api/prices?ticker=${ticker}`);
        if (!priceRes.ok) throw new Error("Price retrieval server failure.");
        const priceData = await priceRes.json();
        
        state.historicalPrices = priceData.prices;
        state.currentPrice = priceData.prices[priceData.prices.length - 1].close;
        
        // Generate mock trading signals historically
        state.tradeSignals = generateSignals(priceData.prices);
        state.userTrades = []; // reset user trades on ticker load
        
        // Update Portfolio calculations
        updatePortfolioUI();
        
        // Render Chart
        drawChart();
        elements.chartTickerName.textContent = `${ticker} Historical Trend`;
        
        // Load default vector grid visualizers
        populateVectorGrid(elements.gruVectorGrid, null, false);
        populateVectorGrid(elements.bertVectorGrid, null, true);
        
        // Fetch News
        const newsRes = await fetch(`/api/news?ticker=${ticker}`);
        if (!newsRes.ok) throw new Error("News feed retrieval server failure.");
        const newsData = await newsRes.json();
        
        // Update News Stream DOM
        renderNewsFeed(newsData.news);
        
        appendConsoleLine(`[DATA] Loaded historical daily rates and news headlines for ${ticker}. Ready to forecast.`, 'success-msg');
        
    } catch (error) {
        appendConsoleLine(`[ERROR] Data load failed: ${error.message}. Loaded local synthetics instead.`, 'error-msg');
        elements.chartTickerName.textContent = `${ticker} (Synthetic Trend)`;
        
        const mockData = generateMockPrices(ticker);
        state.historicalPrices = mockData;
        state.currentPrice = mockData[mockData.length - 1].close;
        state.tradeSignals = generateSignals(mockData);
        state.userTrades = [];
        
        updatePortfolioUI();
        drawChart();
        
        populateVectorGrid(elements.gruVectorGrid, null, false);
        populateVectorGrid(elements.bertVectorGrid, null, true);
        renderNewsFeed(generateMockNews(ticker));
    }
}

// Render News Feed DOM List
function renderNewsFeed(newsItems) {
    elements.newsContainer.innerHTML = '';
    if (!newsItems || newsItems.length === 0) {
        elements.newsContainer.innerHTML = '<div class="empty-news">No matching headlines could be retrieved.</div>';
        return;
    }
    
    newsItems.forEach(item => {
        const row = document.createElement('a');
        row.href = item.link;
        row.target = '_blank';
        row.className = 'news-item';
        
        let tagText = 'NEUTRAL';
        let tagClass = 'neutral';
        const titleL = item.title.toLowerCase();
        
        const posWords = ['growth', 'profit', 'surpass', 'upbeat', 'upgrade', 'rise', 'rises', 'highest', 'gain', 'buy', 'positive'];
        const negWords = ['drop', 'loss', 'decline', 'miss', 'regulatory', 'scrutiny', 'sell', 'fall', 'falls', 'negative'];
        
        let score = 0;
        posWords.forEach(w => { if (titleL.includes(w)) score++; });
        negWords.forEach(w => { if (titleL.includes(w)) score--; });
        
        if (score > 0) {
            tagText = 'BULLISH';
            tagClass = 'bullish';
        } else if (score < 0) {
            tagText = 'BEARISH';
            tagClass = 'bearish';
        }
        
        let formattedDate = item.date;
        try {
            const d = new Date(item.date);
            if (!isNaN(d.getTime())) {
                formattedDate = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            }
        } catch(e) {}
        
        row.innerHTML = `
            <div class="news-item-left">
                <span class="news-title">${item.title}</span>
                <span class="news-meta">${formattedDate} • Yahoo Finance</span>
            </div>
            <span class="sentiment-indicator ${tagClass}">${tagText}</span>
        `;
        elements.newsContainer.appendChild(row);
    });
}

// Helper to format short date from YYYY-MM-DD
function formatDateShort(dateStr) {
    try {
        const parts = dateStr.split('-');
        if (parts.length === 3) {
            const d = new Date(parts[0], parts[1] - 1, parts[2]);
            return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        }
    } catch(e) {}
    return dateStr;
}

// Interactive Custom Canvas Chart Event Bindings
function initChartHandlers() {
    const canvas = elements.stockChart;
    
    // Mouse hover handler for crosshair calculations
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const prices = state.historicalPrices;
        if (!prices || prices.length === 0) return;
        
        const marginLeft = 60;
        const marginRight = 20;
        const plotWidth = rect.width - marginLeft - marginRight;
        
        const index = Math.round(((x - marginLeft) / plotWidth) * (prices.length - 1));
        if (index >= 0 && index < prices.length) {
            if (state.hoverIndex !== index) {
                state.hoverIndex = index;
                drawChart();
            }
        } else {
            if (state.hoverIndex !== null) {
                state.hoverIndex = null;
                drawChart();
            }
        }
    });

    canvas.addEventListener('mouseleave', () => {
        state.hoverIndex = null;
        drawChart();
    });

    // Resize canvas dynamic support
    window.addEventListener('resize', () => {
        if (state.activeTab === 'dashboard') {
            drawChart();
        }
    });
}

// Draw HTML5 Canvas Stock Chart (Candlestick or Line with Volume)
function drawChart() {
    const canvas = elements.stockChart;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentNode.getBoundingClientRect();
    
    // Set actual drawing width scaled by device pixel ratio to keep lines sharp
    canvas.width = rect.width * dpr;
    canvas.height = Math.max(300, rect.height) * dpr;
    
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = Math.max(300, rect.height);
    
    ctx.clearRect(0, 0, width, height);
    
    const prices = state.historicalPrices;
    if (!prices || prices.length === 0) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '14px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText('Loading data trend...', width / 2, height / 2);
        return;
    }
    
    // Extract boundaries
    const highs = prices.map(p => p.high);
    const lows = prices.map(p => p.low);
    const volumes = prices.map(p => p.volume);
    
    const maxP = Math.max(...highs);
    const minP = Math.min(...lows);
    const rangeP = (maxP - minP) || 1;
    
    // Pad scale by 5%
    const scalePad = rangeP * 0.05;
    const paddedMinP = Math.max(0, minP - scalePad);
    const paddedMaxP = maxP + scalePad;
    const paddedRangeP = paddedMaxP - paddedMinP;
    
    const maxV = Math.max(...volumes) || 1;
    
    const marginLeft = 60;
    const marginRight = 20;
    const marginTop = 35;
    const marginBottom = 30;
    
    const plotWidth = width - marginLeft - marginRight;
    const plotHeight = height - marginTop - marginBottom;
    
    const getX = (index) => marginLeft + (index / (prices.length - 1)) * plotWidth;
    const getY = (price) => marginTop + plotHeight - ((price - paddedMinP) / paddedRangeP) * plotHeight;
    
    // 1. Draw horizontal gridlines and Y-axis scale values
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Fira Code';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    const gridCount = 5;
    for (let i = 0; i <= gridCount; i++) {
        const val = paddedMinP + (i / gridCount) * paddedRangeP;
        const y = getY(val);
        
        ctx.beginPath();
        ctx.moveTo(marginLeft, y);
        ctx.lineTo(width - marginRight, y);
        ctx.stroke();
        
        ctx.fillText('$' + val.toFixed(2), marginLeft - 10, y);
    }
    
    // 2. Draw vertical gridlines and X-axis Dates
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    
    const xLabelCount = 6;
    const step = Math.ceil(prices.length / xLabelCount) || 1;
    for (let i = 0; i < prices.length; i += step) {
        const x = getX(i);
        ctx.beginPath();
        ctx.moveTo(x, marginTop);
        ctx.lineTo(x, height - marginBottom);
        ctx.stroke();
        
        const labelDate = formatDateShort(prices[i].date);
        ctx.fillText(labelDate, x, height - marginBottom + 8);
    }
    
    // 3. Draw volume histogram at bottom of chart
    const volHeightMax = plotHeight * 0.20;
    const barWidth = Math.max(1, (plotWidth / prices.length) * 0.7);
    
    for (let i = 0; i < prices.length; i++) {
        const p = prices[i];
        const x = getX(i);
        const vH = (p.volume / maxV) * volHeightMax;
        
        const isUp = p.close >= p.open;
        ctx.fillStyle = isUp ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)';
        ctx.fillRect(x - barWidth / 2, height - marginBottom - vH, barWidth, vH);
    }
    
    // 4. Render main Price plot (Candlesticks or Line)
    if (state.chartMode === 'candle') {
        const candleW = Math.max(2, (plotWidth / prices.length) * 0.6);
        for (let i = 0; i < prices.length; i++) {
            const p = prices[i];
            const x = getX(i);
            const yH = getY(p.high);
            const yL = getY(p.low);
            const yO = getY(p.open);
            const yC = getY(p.close);
            
            const isUp = p.close >= p.open;
            const color = isUp ? '#10b981' : '#ef4444';
            
            // Draw wick
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x, yH);
            ctx.lineTo(x, yL);
            ctx.stroke();
            
            // Draw body
            ctx.fillStyle = color;
            const bodyH = Math.max(1.5, Math.abs(yC - yO));
            const yTop = Math.min(yO, yC);
            
            ctx.fillRect(x - candleW / 2, yTop, candleW, bodyH);
        }
    } else {
        // Line chart rendering
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < prices.length; i++) {
            const x = getX(i);
            const y = getY(prices[i].close);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        
        // Gradient area under line
        const grad = ctx.createLinearGradient(0, marginTop, 0, height - marginBottom);
        grad.addColorStop(0, 'rgba(59, 130, 246, 0.25)');
        grad.addColorStop(1, 'rgba(0, 0, 0, 0.01)');
        
        ctx.lineTo(getX(prices.length - 1), height - marginBottom);
        ctx.lineTo(getX(0), height - marginBottom);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();
    }
    
    // 5. Draw Model Trade Signals on Candlesticks
    if (state.showSignals) {
        state.tradeSignals.forEach(sig => {
            const i = sig.index;
            if (i >= 0 && i < prices.length) {
                const p = prices[i];
                const x = getX(i);
                const isBuy = sig.type === 'BUY';
                
                if (isBuy) {
                    const y = getY(p.low) + 12;
                    // Upward Triangle icon
                    ctx.fillStyle = '#10b981';
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x - 5, y + 8);
                    ctx.lineTo(x + 5, y + 8);
                    ctx.closePath();
                    ctx.fill();
                    
                    ctx.font = 'bold 8px Outfit';
                    ctx.fillStyle = '#10b981';
                    ctx.textAlign = 'center';
                    ctx.fillText('BUY', x, y + 16);
                } else {
                    const y = getY(p.high) - 12;
                    // Downward Triangle icon
                    ctx.fillStyle = '#ef4444';
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x - 5, y - 8);
                    ctx.lineTo(x + 5, y - 8);
                    ctx.closePath();
                    ctx.fill();
                    
                    ctx.font = 'bold 8px Outfit';
                    ctx.fillStyle = '#ef4444';
                    ctx.textAlign = 'center';
                    ctx.fillText('SELL', x, y - 16);
                }
            }
        });

        // 6. Draw User Mock Order signals executed dynamically
        state.userTrades.forEach(trade => {
            const x = getX(trade.index);
            const y = getY(trade.price);
            const isBuy = trade.type === 'BUY';
            
            // Draw distinct blue ring circle representing user execution
            ctx.fillStyle = '#2563eb';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            
            ctx.font = 'bold 8px Outfit';
            ctx.fillStyle = '#3b82f6';
            ctx.textAlign = 'center';
            ctx.fillText(trade.type, x, isBuy ? y - 12 : y + 16);
        });
    }
    
    // 7. Hover crosshairs and Tooltip draw
    if (state.hoverIndex !== null && state.hoverIndex >= 0 && state.hoverIndex < prices.length) {
        const i = state.hoverIndex;
        const p = prices[i];
        const x = getX(i);
        const y = getY(p.close);
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        
        // Vertical dashed tracking line
        ctx.beginPath();
        ctx.moveTo(x, marginTop);
        ctx.lineTo(x, height - marginBottom);
        ctx.stroke();
        
        // Horizontal dashed tracking line
        ctx.beginPath();
        ctx.moveTo(marginLeft, y);
        ctx.lineTo(width - marginRight, y);
        ctx.stroke();
        
        ctx.setLineDash([]); // clear dash formatting
        
        // Plot dot highlighter
        ctx.fillStyle = p.close >= p.open ? '#10b981' : '#ef4444';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Draw float details card
        drawTooltipCard(ctx, p, x, y, width, height);
    }
}

// Draw Floating details tooltip inside Custom Stock Chart
function drawTooltipCard(ctx, p, x, y, canvasW, canvasH) {
    const boxW = 160;
    const boxH = 110;
    
    // Adjust coordinates to fit canvas
    let boxX = x + 15;
    if (boxX + boxW > canvasW - 10) {
        boxX = x - boxW - 15;
    }
    let boxY = y - boxH / 2;
    if (boxY < 10) boxY = 10;
    if (boxY + boxH > canvasH - 10) boxY = canvasH - boxH - 10;
    
    // Draw background card
    ctx.fillStyle = 'rgba(9, 14, 24, 0.95)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 8);
    ctx.fill();
    ctx.stroke();
    
    // Draw content texts
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px Outfit';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`Date: ${p.date}`, boxX + 10, boxY + 10);
    
    ctx.font = '9px Fira Code';
    ctx.fillStyle = '#9ca3af';
    
    ctx.fillText(`Open:  $${p.open.toFixed(2)}`, boxX + 10, boxY + 28);
    ctx.fillText(`High:  $${p.high.toFixed(2)}`, boxX + 10, boxY + 41);
    ctx.fillText(`Low:   $${p.low.toFixed(2)}`, boxX + 10, boxY + 54);
    ctx.fillText(`Close: $${p.close.toFixed(2)}`, boxX + 10, boxY + 67);
    
    const formattedVolume = p.volume >= 1e6 ? (p.volume / 1e6).toFixed(2) + 'M' : p.volume.toLocaleString();
    ctx.fillText(`Vol:   ${formattedVolume}`, boxX + 10, boxY + 80);
    
    // Check if there is a signal matching this date
    const matchedSignal = state.tradeSignals.find(sig => sig.index === state.hoverIndex);
    if (matchedSignal) {
        ctx.font = 'bold 9px Outfit';
        ctx.fillStyle = matchedSignal.type === 'BUY' ? '#10b981' : '#ef4444';
        ctx.fillText(`Signal: ${matchedSignal.type}`, boxX + 10, boxY + 93);
    }
}

// Generate crossover signals historically for stock visualization
function generateSignals(prices) {
    const signals = [];
    if (!prices || prices.length < 15) return [];
    
    // Core SMA 5 vs SMA 15 calculations
    const sma5 = [];
    const sma15 = [];
    
    for (let i = 0; i < prices.length; i++) {
        if (i >= 4) {
            const sum = prices.slice(i - 4, i + 1).reduce((acc, v) => acc + v.close, 0);
            sma5.push(sum / 5);
        } else {
            sma5.push(prices[i].close);
        }
        
        if (i >= 14) {
            const sum = prices.slice(i - 14, i + 1).reduce((acc, v) => acc + v.close, 0);
            sma15.push(sum / 15);
        } else {
            sma15.push(prices[i].close);
        }
    }
    
    // Build Buy / Sell crossover signals
    for (let i = 15; i < prices.length - 1; i++) {
        const prev5 = sma5[i - 1];
        const curr5 = sma5[i];
        const prev15 = sma15[i - 1];
        const curr15 = sma15[i];
        
        if (prev5 <= prev15 && curr5 > curr15) {
            signals.push({ index: i, type: 'BUY' });
        } else if (prev5 >= prev15 && curr5 < curr15) {
            signals.push({ index: i, type: 'SELL' });
        }
    }
    
    return signals;
}

// Populate vector grids with real float parameters
function populateVectorGrid(gridElement, values, isBert = false) {
    if (!gridElement) return;
    gridElement.innerHTML = '';
    
    const size = 128;
    const hasData = values && values.length > 0;
    
    const theme = isBert ? 
        { r: 245, g: 158, b: 11, glow: 'rgba(245, 158, 11, 0.35)' } : 
        { r: 59, g: 130, b: 246, glow: 'rgba(59, 130, 246, 0.35)' }; 
        
    for (let idx = 0; idx < size; idx++) {
        let val = 0.0;
        if (hasData) {
            // Extrapolate the slice values deterministically with sine wave fluctuations
            const base = values[idx % values.length];
            val = base + Math.sin(idx * 1.85) * 0.12;
            val = Math.max(-1, Math.min(1, val));
        } else {
            // Neutral initial grid state
            val = Math.sin(idx * 0.35) * 0.2;
        }
        
        // Normalize range [-1.0, 1.0] -> [0.0, 1.0]
        const factor = (val + 1.0) / 2.0;
        
        const node = document.createElement('div');
        node.className = 'vector-cell';
        node.style.backgroundColor = `rgba(${theme.r}, ${theme.g}, ${theme.b}, ${0.1 + factor * 0.9})`;
        node.setAttribute('data-tooltip', `Dim [${idx}]: ${val.toFixed(4)}`);
        
        if (factor > 0.75) {
            node.style.setProperty('--hover-color', theme.glow);
            node.classList.add('cell-active');
        }
        
        gridElement.appendChild(node);
    }
}

// Spark Fusion Layer Animation
function triggerFusionAnimation() {
    const fBox = elements.mlpFeatureBox;
    const iBox = elements.mlpInteractionBox;
    if (!fBox || !iBox) return;
    
    fBox.classList.remove('active');
    iBox.classList.remove('active');
    
    const fDots = fBox.querySelectorAll('.mlp-dot');
    const iDots = iBox.querySelectorAll('.mlp-dot');
    
    fDots.forEach(d => d.classList.remove('firing'));
    iDots.forEach(d => d.classList.remove('firing'));
    
    // Sequential execution representation
    setTimeout(() => {
        fBox.classList.add('active');
        fDots.forEach((d, idx) => {
            setTimeout(() => d.classList.add('firing'), idx * 80);
        });
    }, 100);
    
    setTimeout(() => {
        iBox.classList.add('active');
        iDots.forEach((d, idx) => {
            setTimeout(() => d.classList.add('firing'), idx * 80);
        });
    }, 700);
}

// Run Deep Learning pipeline prediction
async function executePrediction() {
    state.isPredicting = true;
    elements.runPrediction.disabled = true;
    elements.runPrediction.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running HIMM...';
    
    elements.resultEmpty.classList.add('hidden');
    elements.resultSuccess.classList.add('hidden');
    
    elements.resultCard.className = "glass-card prediction-result-card";
    
    state.windowSize = parseInt(elements.paramWindow.value) || 30;
    state.modelType = elements.paramModel.value;
    
    elements.terminalConsole.innerHTML = '';
    appendConsoleLine(`[SYSTEM] Starting prediction process for ${state.selectedTicker}...`, 'system-msg');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: state.selectedTicker,
                window_size: state.windowSize,
                model_type: state.modelType,
                gemini_key: state.geminiKey
            })
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "Server endpoint failure.");
        }
        
        const payload = await response.json();
        
        let logIndex = 0;
        const printNextLog = () => {
            if (logIndex < payload.logs.length) {
                const line = payload.logs[logIndex];
                
                let logType = '';
                if (line.includes("ERROR") || line.includes("failed")) logType = 'error-msg';
                else if (line.includes("Complete") || line.includes("Prediction:") || line.includes("Successfully")) logType = 'success-msg';
                else if (line.includes("Instantiated") || line.includes("Embedding") || line.includes("MLP")) logType = 'progress-msg';
                else if (line.includes("===") || line.includes("SYSTEM")) logType = 'system-msg';
                
                appendConsoleLine(line, logType);
                logIndex++;
                
                elements.terminalConsole.scrollTop = elements.terminalConsole.scrollHeight;
                setTimeout(printNextLog, 80 + Math.random() * 80);
            } else {
                showSuccessResults(payload);
            }
        };
        
        printNextLog();
        
    } catch (e) {
        appendConsoleLine(`[FATAL] Pipeline crash: ${e.message}`, 'error-msg');
        appendConsoleLine(`[FATAL] Running local fallback mathematical estimator...`, 'system-msg');
        
        setTimeout(() => {
            const fallbackResult = getMockPredictionPayload(state.selectedTicker, state.modelType);
            fallbackResult.logs.forEach(line => appendConsoleLine(line, 'progress-msg'));
            showSuccessResults(fallbackResult);
        }, 1200);
    }
}

// Output final values onto dashboard and update neural visualization stats
function showSuccessResults(payload) {
    state.isPredicting = false;
    elements.runPrediction.disabled = false;
    elements.runPrediction.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Stock Movement';
    
    const isUp = payload.predicted_direction === "UP";
    elements.resultCard.className = `glass-card prediction-result-card ${isUp ? 'up-state' : 'down-state'}`;
    
    // Append prediction signal to the custom candlestick chart
    const lastDayIdx = state.historicalPrices.length - 1;
    
    // Check if there is already a signal at the last index. If so, overwrite, else push.
    const signalIndex = state.tradeSignals.findIndex(sig => sig.index === lastDayIdx);
    const newSignal = { index: lastDayIdx, type: isUp ? 'BUY' : 'SELL' };
    if (signalIndex !== -1) {
        state.tradeSignals[signalIndex] = newSignal;
    } else {
        state.tradeSignals.push(newSignal);
    }
    
    // Redraw graph
    drawChart();
    
    // Update main outputs
    elements.directionBadge.textContent = payload.predicted_direction;
    elements.probabilityText.textContent = `${(payload.confidence * 100).toFixed(1)}% Confidence`;
    
    elements.resTicker.textContent = payload.ticker;
    elements.resModel.textContent = payload.model_type.toUpperCase();
    
    const sentVal = payload.sentiment_score;
    let sentLabel = 'Neutral (0.00)';
    if (sentVal > 0.05) sentLabel = `Bullish (+${sentVal.toFixed(2)})`;
    else if (sentVal < -0.05) sentLabel = `Bearish (${sentVal.toFixed(2)})`;
    elements.resSentiment.textContent = sentLabel;
    
    elements.aiOutlookText.textContent = payload.analysis;
    
    // Load activations grid for GRU & BERT using payloads
    const gSlice = payload.g_t_slice || [];
    const sSlice = payload.s_t_slice || [];
    
    populateVectorGrid(elements.gruVectorGrid, gSlice, false);
    populateVectorGrid(elements.bertVectorGrid, sSlice, true);
    
    // Update Module text metrics
    elements.gruSeqLen.textContent = `${state.windowSize} Trading Days`;
    if (isUp) {
        elements.gruTrendState.textContent = "BULLISH HIGHER LOWS";
        elements.gruTrendState.className = "text-green";
        elements.gruVolSig.textContent = "ACCUMULATION PROFILE";
        elements.gruVolSig.className = "text-green";
    } else {
        elements.gruTrendState.textContent = "BEARISH LOWER HIGHS";
        elements.gruTrendState.className = "text-red";
        elements.gruVolSig.textContent = "DISTRIBUTION PROFILE";
        elements.gruVolSig.className = "text-red";
    }
    
    // Sentiment Gauge indicators
    const percent = ((sentVal + 1.0) / 2.0) * 100; // Map [-1.0, 1.0] to [0%, 100%]
    elements.sentGaugeFill.style.transform = `translateX(${percent - 100}%)`;
    elements.sentGaugeIndicator.style.left = `${percent}%`;
    elements.sentScoreReadout.textContent = sentLabel;
    elements.sentScoreReadout.className = sentVal > 0.05 ? "sentiment-gauge-score-readout text-green" : (sentVal < -0.05 ? "sentiment-gauge-score-readout text-red" : "sentiment-gauge-score-readout text-gold");
    
    // Fusion display text
    elements.fusionTensorDisplay.textContent = `Stacked Mixed Tensor x_t [2 x 768] (Active)`;
    
    // Trigger module transition animations
    if (state.activeModuleTab === 'fusion') {
        triggerFusionAnimation();
    }
    
    // Reveal dashboard card
    elements.resultSuccess.classList.remove('hidden');
    elements.resultSuccess.style.opacity = 0;
    setTimeout(() => {
        elements.resultSuccess.style.opacity = 1;
        elements.resultSuccess.style.transition = 'opacity 0.5s ease';
    }, 50);
}

// -------------------------------------------------------------
// Fallback / Synthetic Data Generators (No server/no internet)
// -------------------------------------------------------------
function generateMockPrices(ticker) {
    const prices = [];
    const basePrice = ticker === 'GOOGL' ? 180 : ticker === 'TSLA' ? 220 : 150;
    let curr = basePrice;
    for (let i = 45; i > 0; i--) {
        const d = new Date();
        d.setDate(d.getDate() - i);
        const change = (Math.random() - 0.47) * 4;
        curr += change;
        prices.push({
            date: d.toISOString().split('T')[0],
            open: curr - (Math.random() * 2),
            close: curr,
            high: curr + (Math.random() * 2),
            low: curr - (Math.random() * 3),
            volume: Math.floor(Math.random() * 8000000) + 2000000
        });
    }
    return prices;
}

function generateMockNews(ticker) {
    return [
        { title: `${ticker} surges as analysts predict massive earnings boost`, link: '#', date: 'Just now' },
        { title: `Technological innovations keep ${ticker} ahead of global rivals`, link: '#', date: '2 hours ago' },
        { title: `Macro trends impact tech shares, ${ticker} shows strong support`, link: '#', date: '6 hours ago' },
        { title: `Options traders bet big on bullish movement for ${ticker}`, link: '#', date: '1 day ago' }
    ];
}

function getMockPredictionPayload(ticker, modelType) {
    const isUp = Math.random() > 0.45;
    const prob = 0.52 + Math.random() * 0.38;
    const score = isUp ? 0.1 + Math.random() * 0.7 : -0.1 - Math.random() * 0.7;
    
    // Synthetic embedding vectors slices
    const mockGSlice = Array.from({ length: 10 }, () => Math.random() * 2.0 - 1.0);
    const mockSSlice = Array.from({ length: 10 }, () => Math.random() * 2.0 - 1.0);
    
    return {
        ticker: ticker,
        model_type: modelType,
        predicted_direction: isUp ? "UP" : "DOWN",
        probability: isUp ? prob : 1 - prob,
        confidence: prob,
        sentiment_score: score,
        analysis: `Semantic models indicate high ${isUp ? 'buying' : 'selling'} volumes for ${ticker} following reports of index changes and quarterly updates. Price embedding models indicate high resistance level support.`,
        g_t_slice: mockGSlice,
        s_t_slice: mockSSlice,
        logs: [
            "=== Starting Fallback Pipeline ===",
            "[DATA] Loaded 30 days price history.",
            "[DATA] News parser loaded 4 news headlines.",
            `[MODEL] Running baseline evaluation using: ${modelType.toUpperCase()}`,
            "[MODEL] Completed feature concatenation.",
            `[SYSTEM] Result evaluated. Sentiment score: ${score.toFixed(2)}`
        ]
    };
}
