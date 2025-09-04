# FPL AI Team Optimizer ⚽🤖

A data-driven tool for optimizing Fantasy Premier League (FPL) team selection. This project leverages machine learning to predict player performance and mathematical optimization to construct the highest-scoring squad possible under FPL's rules.

## Technology Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- PuLP
- NumPy
- Requests

## Key Features

- **🤖 Predictive Modeling:** Utilizes an **XGBoost Regressor** model, supported by the **scikit-learn** ecosystem for validation, trained on multiple seasons of historical data to forecast player points for upcoming gameweeks.
- **⚙️ Feature Engineering:** Creates over 6 predictive features for each player, including:
  - Rolling 5-gameweek form (`form_5_gw`)
  - Rolling advanced stats (`expected_goals`, `expected_assists`)
  - Fixture difficulty (based on opponent's defensive record)
  - Team attacking strength
  - Home vs. Away advantage
- **🧠 Mathematical Optimization:** Implements a **Linear Programming** model using the `PuLP` library to solve the classic "knapsack problem" of team selection. It navigates multiple constraints to find the optimal 15-player squad:
  - £100.0m budget limit
  - Positional quotas (2 GKP, 5 DEF, 5 MID, 3 FWD)
  - Maximum of 3 players per Premier League team
- **📈 Realistic Backtesting Framework:** A full simulation of the 2023-24 season, starting with an optimized GW1 team and making a single intelligent transfer each week to mimic real FPL rules and prove the model's long-term value.
- **⚽ Team Builder For Current Season:** Includes a script (main.py) that trains a model on all past data, fetches live FPL data for the current gameweek, and runs the optimizer to recommend the best possible team from scratch.

## Project Workflow

The project follows a complete, end-to-end data science pipeline:

1.  **Data Collection:** Gathers multiple seasons of historical, gameweek-by-gameweek player data.
2.  **Feature Engineering:** Cleans the data and constructs predictive features.
3.  **Model Training:** Trains an XGBoost model on a past season (e.g., 2022-23) to learn scoring patterns.
4.  **Simulation & Prediction:** For each gameweek of the test season (e.g., 2023-24), the model predicts player scores.
5.  **Optimization:** The PuLP solver takes the predictions and selects the optimal squad or transfer.
6.  **Performance Evaluation:** The selected team's actual score is recorded, and the process repeats for all 38 gameweeks.

## Performance & Results

- In a simulation of the **2023-24 season** without transfer limits, the model-managed team achieved a final score of **2082 points**.

- In a simulation of the **2023-24 season** with a one-transfer-per-week rule, the model-managed team achieved a final score of **1726 points**. This excludes features such as wildcards, free hits, and triple captains.

- The team builder (`main.py`) optimised a team for **gameweek 4 of the 25-26 season** with a total of **85.63 points**

## Installation & Usage

### Prerequisites

- Python 3.10+
- `pip` and `venv`

### Setup

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/anirudh242/FPL-Builder.git]
    cd fpl-ai-optimizer
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Scripts

- **To generate the best team for the upcoming gameweek:**

  ```bash
  python main.py
  ```

- **To run the full 2023-25 season backtest with no transfer limits:**

  ```bash
  python backtest.py
  ```

- **To run the full 2023-24 season backtest simulation:**
  ```bash
  python backtest_with_transfers.py
  ```

## Future Improvements

- **Advanced Transfer Logic:** Implement a more complex transfer agent that can save transfers and take point hits.
- **Chip Strategy:** Develop heuristics for the optimal use of the Wildcard, Free Hit, Bench Boost, and Triple Captain chips.
- **More Features:** Incorporate player injury/suspension status and betting odds for clean sheets and goalscorers.
