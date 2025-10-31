'''
Solution for problem 3.1
Using the Monte Carlo method to optimize a portfolio.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

assets_observations_list = [
    [2.2, 0.1, -0.77, 0.6, 1.1, 0.5, 0],
    [0, 1.1, -1.2, 2.2, -0.56, -0.012, 0.21],
    [-0.5, -1.6, 1.1, -0.3, -4.05, 1.2, 0.8],
    [-0.45, 2.2, -2.3, 1.1, 1.3, 1.4, 0.2],
    [1.1, 0.11, -0.2, 4.12, 2.2, 1.8, 0.9],
]

port_returns = []
sharpe_ratio = []
port_variance = []
stock_weights = []

num_portfolios = 50000
max_asset_weight = 0.35  # Constraint


def rand_weights(high, n):
    '''Returns 1D array of weights each weight <=0.35.'''
    while True:
        k = np.random.rand(n)
        rand_mtx = (k / sum(k))
        if (np.all(rand_mtx <= high)):
            return rand_mtx


def portfolio_returns_variance_weights(high, ret):
    '''Returns expected return, volatility and weights for a given portfolio.'''
    mu = np.asmatrix(np.mean(ret, axis=1))
    w = np.asmatrix(rand_weights(high, len(ret)))
    C = np.asmatrix(np.cov(ret))
    R = mu * w.T
    sigma = np.sqrt(w * (C * w.T))
    return float(R), float(sigma), w


def nice_print(name, port_df):
    '''Print portfolio data'''
    portfolio_weights = np.array(port_df['Weights'])[0][0]
    if name == 'Sharpe Ratio':
        print("\nMaximum", name, "Portfolio:\n")
    else:
        print("\nMinimum", name, "Portfolio:\n")
    print('Sharpe Ratio', float(port_df['Sharpe Ratio']))
    print('Variance', float(port_df['Variance']))
    print('Returns', float(port_df['Returns']))
    print("Weights:")
    for j in range(0, len(portfolio_weights)):
        print("Asset index", j, "weight", float(portfolio_weights[j]))


for _ in range(num_portfolios):
    returns, variance, weights = portfolio_returns_variance_weights(max_asset_weight, assets_observations_list)
    # Risk free rate of 0% was assumed
    # Assuming optimal portfolio = sharpe ratio
    sharpe = returns / variance
    port_returns.append(returns)
    port_variance.append(variance)
    stock_weights.append(weights.tolist())
    sharpe_ratio.append(sharpe)

portfolio = {
    'Returns': port_returns,
    'Variance': port_variance,
    'Weights': stock_weights,
    'Sharpe Ratio': sharpe_ratio
}

df = pd.DataFrame(portfolio)

max_sharpe_ratio = df['Sharpe Ratio'].max()
min_variance = df['Variance'].min()

max_sharpe_ratio_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe_ratio]
min_variance_portfolio = df.loc[df['Variance'] == min_variance]

nice_print("Sharpe Ratio", max_sharpe_ratio_portfolio)
nice_print("Variance", min_variance_portfolio)

plt.style.use('seaborn-dark')
df.plot.scatter(x='Variance', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=max_sharpe_ratio_portfolio['Variance'], y=max_sharpe_ratio_portfolio['Returns'], c='red', marker='D',
            s=200)
plt.scatter(x=min_variance_portfolio['Variance'], y=min_variance_portfolio['Returns'], c='blue', marker='D',
            s=200)
plt.xlabel('Variance')
plt.ylabel('Expected Returns')
plt.show()

'''
Sample Outputs:

---Constraints wi <= 0.35---

Maximum Sharpe Ratio Portfolio:

Sharpe Ratio 0.9164832237698075
Variance 0.7756794611487344
Returns 0.7108972131656193
Weights:
Asset index 0 weight 0.3469557261431653
Asset index 1 weight 0.023456205100957864
Asset index 2 weight 0.11649284793144103
Asset index 3 weight 0.1694111360363069
Asset index 4 weight 0.34368408478812873

Maximum Variance Portfolio:

Sharpe Ratio 0.461366197369963
Variance 0.5783576800327453
Returns 0.2668346835564215
Weights:
Asset index 0 weight 0.34812168520826886
Asset index 1 weight 0.05296441134536398
Asset index 2 weight 0.28305040175632673
Asset index 3 weight 0.26483211609689133
Asset index 4 weight 0.05103138559314893

---No Constraints---

Maximum Sharpe Ratio Portfolio:

Sharpe Ratio 0.9908069809692951
Variance 1.1210143748541492
Returns 1.1107088683724213
Weights:
Asset index 0 weight 0.2783968842102354
Asset index 1 weight 0.02559127916680718
Asset index 2 weight 1.3402151436750304e-05
Asset index 3 weight 0.04388379076461679
Asset index 4 weight 0.6521146437069039

Maximum Variance Portfolio:

Sharpe Ratio 0.4213639510222053
Variance 0.5554904603051585
Returns 0.23406365510932509
Weights:
Asset index 0 weight 0.49144121097458743
Asset index 1 weight 0.049355522828479687
Asset index 2 weight 0.27711012381031896
Asset index 3 weight 0.17910111447097304
Asset index 4 weight 0.002992027915640844
'''
