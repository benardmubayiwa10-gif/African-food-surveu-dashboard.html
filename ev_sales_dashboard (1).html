<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Electric Vehicle Sales Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header p {
            font-size: 1.3em;
            opacity: 0.9;
        }

        .filters {
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .filter-group {
            margin-bottom: 20px;
        }

        .filter-group:last-child {
            margin-bottom: 0;
        }

        .filter-group label {
            display: block;
            font-weight: 600;
            color: #334155;
            margin-bottom: 10px;
            font-size: 0.95em;
        }

        .filter-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .filter-btn {
            padding: 10px 20px;
            border: 2px solid #e2e8f0;
            background: #f8fafc;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            color: #475569;
        }

        .filter-btn:hover {
            background: #e2e8f0;
            transform: translateY(-2px);
        }

        .filter-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .kpi-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }

        .kpi-card.blue {
            border-left: 5px solid #3b82f6;
        }

        .kpi-card.green {
            border-left: 5px solid #10b981;
        }

        .kpi-card.orange {
            border-left: 5px solid #f59e0b;
        }

        .kpi-card.purple {
            border-left: 5px solid #8b5cf6;
        }

        .kpi-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .kpi-title {
            color: #64748b;
            font-size: 0.9em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .kpi-icon {
            font-size: 2em;
            opacity: 0.2;
        }

        .kpi-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 5px;
        }

        .kpi-subtitle {
            color: #64748b;
            font-size: 0.85em;
        }

        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }

        .chart-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .chart-card.full-width {
            grid-column: 1 / -1;
        }

        .chart-title {
            font-size: 1.4em;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .chart-icon {
            font-size: 1.2em;
        }

        .chart-container {
            position: relative;
            height: 350px;
        }

        .table-container {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        thead {
            background: #f8fafc;
        }

        th {
            padding: 15px;
            text-align: left;
            font-weight: 700;
            color: #334155;
            border-bottom: 2px solid #e2e8f0;
        }

        td {
            padding: 15px;
            color: #475569;
            border-bottom: 1px solid #f1f5f9;
        }

        tr:hover {
            background: #f8fafc;
        }

        .footer {
            text-align: center;
            color: #64748b;
            padding: 20px;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2em;
            }
            
            .chart-grid {
                grid-template-columns: 1fr;
            }
            
            .kpi-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>⚡ Electric Vehicle Sales Dashboard</h1>
            <p>Advanced Performance Analytics & Revenue Intelligence</p>
        </div>

        <!-- Filters -->
        <div class="filters">
            <div class="filter-group">
                <label>Region Filter</label>
                <div class="filter-buttons" id="regionFilters">
                    <button class="filter-btn active" data-region="All">All Regions</button>
                    <button class="filter-btn" data-region="North America">North America</button>
                    <button class="filter-btn" data-region="Europe">Europe</button>
                    <button class="filter-btn" data-region="Asia Pacific">Asia Pacific</button>
                    <button class="filter-btn" data-region="Latin America">Latin America</button>
                </div>
            </div>
            <div class="filter-group">
                <label>Year Filter</label>
                <div class="filter-buttons" id="yearFilters">
                    <button class="filter-btn active" data-year="All">All Years</button>
                    <button class="filter-btn" data-year="2023">2023</button>
                    <button class="filter-btn" data-year="2024">2024</button>
                </div>
            </div>
        </div>

        <!-- KPIs -->
        <div class="kpi-grid">
            <div class="kpi-card blue">
                <div class="kpi-header">
                    <span class="kpi-title">Total Revenue</span>
                    <span class="kpi-icon">💰</span>
                </div>
                <div class="kpi-value" id="kpiRevenue">$0</div>
                <div class="kpi-subtitle" id="kpiRevenueGrowth">Loading...</div>
            </div>

            <div class="kpi-card green">
                <div class="kpi-header">
                    <span class="kpi-title">Total Units Sold</span>
                    <span class="kpi-icon">🚗</span>
                </div>
                <div class="kpi-value" id="kpiUnits">0</div>
                <div class="kpi-subtitle">Across all regions</div>
            </div>

            <div class="kpi-card orange">
                <div class="kpi-header">
                    <span class="kpi-title">Avg Vehicle Price</span>
                    <span class="kpi-icon">🏆</span>
                </div>
                <div class="kpi-value" id="kpiAvgPrice">$0</div>
                <div class="kpi-subtitle">Per unit average</div>
            </div>

            <div class="kpi-card purple">
                <div class="kpi-header">
                    <span class="kpi-title">Active Regions</span>
                    <span class="kpi-icon">🌍</span>
                </div>
                <div class="kpi-value" id="kpiRegions">0</div>
                <div class="kpi-subtitle">Global coverage</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="chart-grid">
            <div class="chart-card full-width">
                <h3 class="chart-title">
                    <span class="chart-icon">📈</span>
                    Monthly Revenue Trend
                </h3>
                <div class="chart-container">
                    <canvas id="revenueChart"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <h3 class="chart-title">
                    <span class="chart-icon">🌎</span>
                    Regional Performance
                </h3>
                <div class="chart-container">
                    <canvas id="regionalChart"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <h3 class="chart-title">
                    <span class="chart-icon">😊</span>
                    Customer Sentiment
                </h3>
                <div class="chart-container">
                    <canvas id="sentimentChart"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <h3 class="chart-title">
                    <span class="chart-icon">📊</span>
                    Units Sold by Month
                </h3>
                <div class="chart-container">
                    <canvas id="unitsChart"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <h3 class="chart-title">
                    <span class="chart-icon">💎</span>
                    Price vs Units Correlation
                </h3>
                <div class="chart-container">
                    <canvas id="correlationChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Summary Table -->
        <div class="table-container">
            <h3 class="chart-title">
                <span class="chart-icon">📋</span>
                Regional Summary Statistics
            </h3>
            <table id="summaryTable">
                <thead>
                    <tr>
                        <th>Region</th>
                        <th style="text-align: right;">Total Revenue</th>
                        <th style="text-align: right;">Total Units</th>
                        <th style="text-align: right;">Avg Revenue/Transaction</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <!-- Data will be populated by JavaScript -->
                </tbody>
            </table>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>📊 Dashboard powered by advanced analytics • Last updated: <span id="lastUpdate"></span> • Data points: <span id="dataPoints">0</span></p>
        </div>
    </div>

    <script>
        // Generate realistic EV sales data
        function generateData() {
            const regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America'];
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            const data = [];
            
            for (let year = 2023; year <= 2024; year++) {
                for (let monthIdx = 0; monthIdx < 12; monthIdx++) {
                    regions.forEach(region => {
                        const seasonalFactor = 1 + Math.sin(monthIdx * Math.PI / 6) * 0.3;
                        const regionalMultiplier = {
                            'North America': 1.2,
                            'Europe': 1.1,
                            'Asia Pacific': 1.5,
                            'Latin America': 0.8
                        }[region];
                        
                        const baseUnits = 80 + Math.floor(Math.random() * 60);
                        const unitsSold = Math.floor(baseUnits * seasonalFactor * regionalMultiplier);
                        const avgPrice = 48000 + Math.random() * 12000;
                        const revenue = unitsSold * avgPrice;
                        
                        const sentimentRoll = Math.random();
                        const sentiment = sentimentRoll > 0.7 ? 'Positive' : sentimentRoll > 0.3 ? 'Neutral' : 'Negative';
                        
                        data.push({
                            date: `${year}-${String(monthIdx + 1).padStart(2, '0')}-01`,
                            year: year,
                            month: months[monthIdx],
                            monthYear: `${months[monthIdx]} ${year}`,
                            region: region,
                            unitsSold: unitsSold,
                            revenue: revenue,
                            avgPrice: avgPrice,
                            sentiment: sentiment
                        });
                    });
                }
            }
            return data;
        }

        const allData = generateData();
        let currentRegion = 'All';
        let currentYear = 'All';
        let charts = {};

        // Format currency
        function formatCurrency(value) {
            if (value >= 1000000) {
                return `$${(value / 1000000).toFixed(1)}M`;
            }
            return `$${(value / 1000).toFixed(0)}K`;
        }

        // Format number
        function formatNumber(value) {
            return value.toLocaleString();
        }

        // Filter data
        function getFilteredData() {
            let filtered = allData;
            if (currentRegion !== 'All') {
                filtered = filtered.filter(d => d.region === currentRegion);
            }
            if (currentYear !== 'All') {
                filtered = filtered.filter(d => d.year === parseInt(currentYear));
            }
            return filtered;
        }

        // Calculate KPIs
        function updateKPIs() {
            const data = getFilteredData();
            const totalRevenue = data.reduce((sum, d) => sum + d.revenue, 0);
            const totalUnits = data.reduce((sum, d) => sum + d.unitsSold, 0);
            const avgPrice = data.reduce((sum, d) => sum + d.avgPrice, 0) / data.length;
            const uniqueRegions = [...new Set(data.map(d => d.region))].length;

            // Calculate growth
            const sorted = [...data].sort((a, b) => new Date(a.date) - new Date(b.date));
            const halfPoint = Math.floor(sorted.length / 2);
            const firstHalf = sorted.slice(0, halfPoint);
            const secondHalf = sorted.slice(halfPoint);
            const firstHalfRevenue = firstHalf.reduce((sum, d) => sum + d.revenue, 0);
            const secondHalfRevenue = secondHalf.reduce((sum, d) => sum + d.revenue, 0);
            const growthRate = ((secondHalfRevenue - firstHalfRevenue) / firstHalfRevenue) * 100;

            document.getElementById('kpiRevenue').textContent = formatCurrency(totalRevenue);
            document.getElementById('kpiRevenueGrowth').textContent = `${growthRate > 0 ? '↗' : '↘'} ${Math.abs(growthRate).toFixed(1)}% growth`;
            document.getElementById('kpiUnits').textContent = formatNumber(totalUnits);
            document.getElementById('kpiAvgPrice').textContent = formatCurrency(avgPrice);
            document.getElementById('kpiRegions').textContent = uniqueRegions;
            document.getElementById('dataPoints').textContent = formatNumber(data.length);
        }

        // Update charts
        function updateCharts() {
            const data = getFilteredData();

            // Monthly revenue trend
            const monthlyData = {};
            data.forEach(d => {
                if (!monthlyData[d.monthYear]) {
                    monthlyData[d.monthYear] = { revenue: 0, units: 0, date: d.date };
                }
                monthlyData[d.monthYear].revenue += d.revenue;
                monthlyData[d.monthYear].units += d.unitsSold;
            });

            const monthlyArray = Object.entries(monthlyData)
                .map(([key, val]) => ({ monthYear: key, ...val }))
                .sort((a, b) => new Date(a.date) - new Date(b.date));

            if (charts.revenue) charts.revenue.destroy();
            charts.revenue = new Chart(document.getElementById('revenueChart'), {
                type: 'line',
                data: {
                    labels: monthlyArray.map(d => d.monthYear),
                    datasets: [{
                        label: 'Revenue',
                        data: monthlyArray.map(d => d.revenue),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => formatCurrency(context.parsed.y)
                            }
                        }
                    },
                    scales: {
                        y: {
                            ticks: {
                                callback: (value) => formatCurrency(value)
                            }
                        }
                    }
                }
            });

            // Regional performance
            const regionalData = {};
            allData.forEach(d => {
                if (!regionalData[d.region]) {
                    regionalData[d.region] = { revenue: 0, units: 0 };
                }
                regionalData[d.region].revenue += d.revenue;
                regionalData[d.region].units += d.unitsSold;
            });

            const regionalArray = Object.entries(regionalData)
                .map(([region, data]) => ({ region, ...data }))
                .sort((a, b) => b.revenue - a.revenue);

            if (charts.regional) charts.regional.destroy();
            charts.regional = new Chart(document.getElementById('regionalChart'), {
                type: 'bar',
                data: {
                    labels: regionalArray.map(d => d.region),
                    datasets: [{
                        label: 'Revenue',
                        data: regionalArray.map(d => d.revenue),
                        backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => formatCurrency(context.parsed.x)
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                callback: (value) => formatCurrency(value)
                            }
                        }
                    }
                }
            });

            // Customer sentiment
            const sentimentData = {};
            data.forEach(d => {
                sentimentData[d.sentiment] = (sentimentData[d.sentiment] || 0) + 1;
            });

            if (charts.sentiment) charts.sentiment.destroy();
            charts.sentiment = new Chart(document.getElementById('sentimentChart'), {
                type: 'doughnut',
                data: {
                    labels: Object.keys(sentimentData),
                    datasets: [{
                        data: Object.values(sentimentData),
                        backgroundColor: ['#10b981', '#f59e0b', '#ef4444']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });

            // Monthly units
            const monthlyUnits = {};
            data.forEach(d => {
                monthlyUnits[d.month] = (monthlyUnits[d.month] || 0) + d.unitsSold;
            });

            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

            if (charts.units) charts.units.destroy();
            charts.units = new Chart(document.getElementById('unitsChart'), {
                type: 'bar',
                data: {
                    labels: months,
                    datasets: [{
                        label: 'Units Sold',
                        data: months.map(m => monthlyUnits[m] || 0),
                        backgroundColor: '#8b5cf6',
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    }
                }
            });

            // Correlation scatter
            const scatterData = data.slice(0, 100).map(d => ({
                x: d.unitsSold,
                y: d.avgPrice
            }));

            if (charts.correlation) charts.correlation.destroy();
            charts.correlation = new Chart(document.getElementById('correlationChart'), {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Price vs Units',
                        data: scatterData,
                        backgroundColor: 'rgba(139, 92, 246, 0.6)',
                        borderColor: '#8b5cf6',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Units Sold'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Avg Price ($)'
                            },
                            ticks: {
                                callback: (value) => `$${(value/1000).toFixed(0)}k`
                            }
                        }
                    }
                }
            });

            // Update table
            const tableBody = document.getElementById('tableBody');
            tableBody.innerHTML = '';
            regionalArray.forEach(region => {
                const row = `
                    <tr>
                        <td style="font-weight: 600;">${region.region}</td>
                        <td style="text-align: right;">${formatCurrency(region.revenue)}</td>
                        <td style="text-align: right;">${formatNumber(region.units)}</td>
                        <td style="text-align: right;">${formatCurrency(region.revenue / region.units)}</td>
                    </tr>
                `;
                tableBody.innerHTML += row;
            });
        }

        // Filter button handlers
        document.getElementById('regionFilters').addEventListener('click', (e) => {
            if (e.target.classList.contains('filter-btn')) {
                document.querySelectorAll('#regionFilters .filter-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                e.target.classList.add('active');
                currentRegion = e.target.dataset.region;
                updateKPIs();
                updateCharts();
            }
        });

        document.getElementById('yearFilters').addEventListener('click', (e) => {
            if (e.target.classList.contains('filter-btn')) {
                document.querySelectorAll('#yearFilters .filter-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                e.target.classList.add('active');
                currentYear = e.target.dataset.year;
                updateKPIs();
                updateCharts();
            }
        });

        // Initialize
        document.getElementById('lastUpdate').textContent = new Date().toLocaleDateString();
        updateKPIs();
        updateCharts();
    </script>
</body>
</html>
