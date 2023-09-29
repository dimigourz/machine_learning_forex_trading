# Forex Trading Automation using LightGBM & OANDA API

This repository is a representation of my journey into the world of algorithmic trading, specifically with the OANDA platform, and the experimentation with the LightGBM machine learning algorithm for predicting Forex market movements.

## Overview

The project focuses on utilizing the LightGBM algorithm, which by analyzing historical Forex market data, offers insights into potential price movements. The bot's design emphasizes seamless integration with the OANDA API, facilitating timely buy/sell order placements and trade closures.

## Dependencies

- `lightgbm`: Utilized for the machine learning-based predictive modeling.
- `oandapyV20`: A Python wrapper to interface with the OANDA trading platform.
- Various Python libraries: Essential libraries such as `numpy` and `pandas` for data handling and analysis.

## Key Features

- **Comprehensive OANDA API Integration**: This bot supports all the fundamental commands provided by the OANDA API, ensuring a comprehensive trading experience.
  
- **Data-Driven Decisions with LightGBM**: Taps into historical Forex data to derive potential trading signals.
  
- **Automated Trading Flow**: Automates the trading process, enabling order placements and trade exits as the predictive model suggests.

## File Descriptions

- **features.py**: Focuses on feature engineering for forex data. Contains utilities essential for transforming features suitable for our predictive model.

- **forex_data.py**: Manages the acquisition and preprocessing of forex-related data. It offers utilities for capturing both real-time and historical data.

- **forex_model.py**: Contains the machine learning component, where the LGBM algorithm is used for predicting potential forex market trends.

- **oanda_api.py**: Facilitates direct interactions with the OANDA platform, handling authentication, order execution, and more.

- **wegolive.py**: Acts as the heartbeat of the bot. Contains the primary trading loop, decision-making logic, and notifications.

## Configuration & Setup

1. **OANDA Credentials**: An API token and account ID from OANDA are imperative for the bot's operation. Ensure you obtain these credentials before initiating the bot.
  
2. **Clone & Setup**: Clone the repository, install the required dependencies, adjust the settings as necessary, authenticate with the OANDA credentials, and you're set to deploy the bot.

## Expansion & Future Work

This bot was designed out of personal interest to learn and experiment with algorithmic trading. While it integrates a foundational LGBM strategy, its design encourages adaptability.

**Potentials:**

- **Incorporating Unique Strategies**: The codebase is structured to accommodate various trading methodologies, not just machine learning. Customize as you see fit.
  
- **Advanced Features**: There's room for further feature integration, like advanced risk management or dynamic trading rules.

- **Performance Metrics**: Introducing performance tracking and logging will allow for better insights into the bot's trading efficacy over periods.
  
