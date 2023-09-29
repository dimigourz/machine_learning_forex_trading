# Forex Trading Automation using LightGBM & OANDA API

This repository houses a trading bot which harnesses the predictive power of the LightGBM machine learning algorithm. It is intricately designed to interface with the OANDA trading platform via the API, ensuring all essential trading commands are supported.

## Overview

The core of this bot lies in its utilization of the LightGBM algorithm. By analyzing past Forex market data, it predicts potential price movements. Its seamless integration with the OANDA API allows for timely buy/sell order placements and trade closures based on the model's predictions.

## Dependencies

- `lightgbm`: For machine learning-based predictive modeling.
- `oandapyV20`: An efficient Python wrapper for interfacing with the OANDA trading platform.
- Various Python libraries: Such as `numpy` and `pandas`, to aid in data processing and analysis.

## Key Features

- **Comprehensive OANDA API Integration**: Fully leverages the basic commands of the OANDA API, ensuring you're not missing out on any crucial trading capability.
- **Data-Driven Decision Making with LightGBM**: Mines historical Forex data to generate trading signals for buy/sell decisions.
- **Automated Trading Dynamics**: Removes the manual labor from trading by automating order placements and trade exits based on predictive outputs.

## File Descriptions

- **features.py**: Handles feature engineering and manipulation for forex data. Contains utilities for creating or transforming features for our predictive model.

- **forex_data.py**: Responsible for fetching, processing, or storing forex-related data. It contains tools to fetch real-time or historical data and preprocess the data for model consumption.

- **forex_model.py**: Holds the machine learning logic, primarily the LGBM algorithm, for predicting forex market movements.

- **oanda_api.py**: Interfaces directly with the OANDA trading platform. It deals with authentication, executing orders, and fetching account details.

- **wegolive.py**: Deploys or runs the trading bot in real-time. It contains the primary trading loop, decision logic, and alert mechanisms.

## Configuration & Setup

1. **OANDA Credentials**: Before using the bot, ensure you have both the API token and account ID from OANDA. These are crucial for authentication and trading access.
2. **Clone**: Begin by cloning the repository to your local machine.
3. **Installation**: Set up the required packages using pip or another package manager.
4. **Configuration**: Adjust `ticker_symbols`, time frames, and other pertinent parameters to suit your trading strategy.
5. **Authentication**: With your OANDA API token and account ID on hand, ensure you're connected by supplying them to the OANDA API within the script's designated sections.
6. **Deployment**: Launch the script and watch as the bot carries out trading operations based on real-time data and algorithmic predictions.

## Expansion & Future Work

I designed this bot as a learning project, focusing on algorithmic trading on OANDA's platform. While the bot utilizes a basic LGBM-based strategy, its architecture is flexible, making it suitable for further expansions and customizations.

**Here's what we can do next:**

- **Custom Strategies**: Dive into the codebase and add your unique trading strategies. Whether it's based on machine learning models, technical indicators, or other methodologies, the bot is designed to be flexible.
  
- **Additional Features**: Consider integrating features like risk management tools or dynamic stop-loss settings.

- **Performance Monitoring**: Implement tracking metrics and logging mechanisms to closely monitor your bot's trading performance over time.
