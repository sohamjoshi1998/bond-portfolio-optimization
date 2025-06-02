# bond-portfolio-optimization
Quant Project 1 - Credit Risk-Based Bond Portfolio Optimization

# 📊 Credit Risk-Based Bond Portfolio Optimization

This project builds a Python-based framework to optimize a corporate bond portfolio by adjusting returns for credit risk. It incorporates **Expected Credit Loss (ECL)** using **PD (Probability of Default)**, **LGD (Loss Given Default)**, and **EAD (Exposure at Default)**, then applies portfolio optimization techniques to maximize risk-adjusted returns.

## 🚀 Overview

Traditional portfolio optimization often overlooks credit risk. This model integrates credit risk into expected returns and applies modern portfolio theory to build an optimized bond portfolio.

---

## 🔧 Features

- Simulate or ingest real-world bond data
- Calculate ECL for each bond
- Adjust returns using credit loss
- Portfolio optimization using **PyPortfolioOpt**
- Visualize credit risk impact (ECL by bond)
- Extendable to include **scenario analysis** or **stress testing**

---

## 📈 Sample Output

![ECL Plot](reports/ecl_plot.png) <!-- Replace with actual image if available -->

---

## 📦 Tech Stack

- **Python**
- `Pandas`, `NumPy`, `Matplotlib`
- `PyPortfolioOpt` for optimization
- (Optional: `Streamlit` for dashboard)

---

## 🧠 Concepts Applied

- Expected Credit Loss (ECL) = PD × LGD × EAD
- Mean-Variance Optimization
- Sharpe Ratio Maximization
- Risk-adjusted Return Modeling

---

## 🚧 Future Enhancements

- 🏦 Add sector and issuer-level constraints  
- 📉 Run scenario-based stress testing  
- 📊 Develop interactive dashboard (Streamlit/Plotly)  
- 🔍 Integrate market data APIs (FRED/FINRA)  

---

## 📚 References

- Basel II/III Credit Risk Framework
- PyPortfolioOpt Documentation
- FRM Curriculum: Credit Risk & Portfolio Management

---

## 🧑‍💻 Author

**Soham Joshi**  

# Credit Risk-Based Bond Portfolio Optimization

# === Project Structure ===
# /data         --> Input data (synthetic or real)
# /notebooks    --> Jupyter notebooks for analysis
# /scripts      --> Python modules (data processing, optimization)
# /reports      --> Output reports or plots
# README.md     --> Project description and setup instructions

# === Key Components ===

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Step 2: Simulate Bond Data (Replace with real data later)
bond_data = pd.DataFrame({
    'Bond': [f'Bond_{i}' for i in range(10)],
    'PD': np.random.uniform(0.01, 0.1, 10),  # Probability of Default
    'LGD': np.random.uniform(0.4, 0.6, 10),  # Loss Given Default
    'EAD': np.random.uniform(90, 110, 10),   # Exposure at Default (face value)
    'Expected_Return': np.random.uniform(0.02, 0.08, 10),
    'Volatility': np.random.uniform(0.01, 0.05, 10)
})

# Step 3: Compute Expected Credit Loss (ECL)
bond_data['ECL'] = bond_data['PD'] * bond_data['LGD'] * bond_data['EAD']
bond_data['Adj_Return'] = bond_data['Expected_Return'] - (bond_data['ECL'] / bond_data['EAD'])

# Step 4: Portfolio Optimization
mu = bond_data['Adj_Return']
S = np.diag(bond_data['Volatility']**2)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

# Step 5: Display Results
print("\nOptimized Portfolio Weights:")
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

# Step 6: Plot ECL Contribution
plt.figure(figsize=(10,6))
plt.bar(bond_data['Bond'], bond_data['ECL'])
plt.title('Expected Credit Loss by Bond')
plt.xlabel('Bond')
plt.ylabel('ECL ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Next Steps ===
# - Add real bond market data
# - Incorporate sector exposure constraints
# - Implement scenario analysis and stress testing
# - Create a dashboard using Streamlit or Plotly Dash
