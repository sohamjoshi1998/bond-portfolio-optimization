# bond-portfolio-optimization
Credit Risk-Based Bond Portfolio Optimization

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
