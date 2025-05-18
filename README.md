# Stock Market Analysis Tool

A comprehensive Python-based tool for stock market analysis, trading signals, and portfolio management. This project combines technical analysis, options tracking, machine learning models, and automated alerts to provide a robust trading and analysis platform.

## Features

### Price Analysis
- Technical indicators (Moving Averages, Momentum, Volatility)
- Multi-timeframe signal generation
- Price data management and database connectivity
- Custom technical analysis utilities

### Options Trading
- Options chain data collection
- Statistical analysis of options data
- Tracking and scanning capabilities
- Expected moves calculations
- Comprehensive options management system

### Machine Learning Models
- Classification models (including CLDA)
- Neural Network models (PyTorch implementation)
- Time series forecasting (SARIMAX)
- Anomaly detection
- Trend detection and change point analysis
- Indicator-based models

### Additional Components
- Bond analysis and tracking
- Earnings data collection and analysis
- Automated alerts system
- Performance reporting
- Data visualization and plotting utilities

## Project Structure

```
.
├── alerts/               # Alert system for market events
├── bonds/               # Bond analysis tools
├── earnings/            # Earnings data collection and analysis
├── models/             
│   ├── anom/           # Anomaly detection models
│   ├── classification/ # Classification models
│   ├── forecast/       # Time series forecasting
│   ├── nn/            # Neural network implementations
│   ├── trees/         # Decision tree models
│   └── trends/        # Trend detection algorithms
├── options/
│   ├── optgd/         # Options data management
│   ├── stat/          # Statistical analysis
│   └── track/         # Options tracking system
├── plots/              # Visualization tools
├── price/              # Price data and technical analysis
└── utils/              # Utility functions and tools
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database and required configurations:
```bash
python utils/initialize.py
```

## Usage

### Price Analysis
```python
from price.indicators import calculate_indicators
from price.signals.signal_generation import generate_signals

# Generate trading signals
signals = generate_signals(symbol="SPY")
```

### Options Analysis
```python
from options.optgd.option_chain import get_option_chain
from options.stat.manage_stats import analyze_options

# Get current option chain
chain = get_option_chain(symbol="SPY")

# Analyze options statistics
stats = analyze_options(chain)
```

### Model Usage
```python
from models.classification.get_orders import get_trading_signals
from models.trends.trend_detector import detect_trends

# Get trading signals from classification model
signals = get_trading_signals(symbol="SPY")

# Detect market trends
trends = detect_trends(price_data)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[License Type] - See LICENSE file for details

## Notes

- Make sure to configure your database connections in relevant configuration files
- Some features require market data access
- Models may need training before first use