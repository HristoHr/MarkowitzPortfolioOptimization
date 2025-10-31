'''
Solution for problem 3.2
Using the Monte Carlo method to optimize a portfolio.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

asset_observations_list = [
    [2.2, 0.1, -0.77, 0.6, 1.1, 0.5, 0],
    [0, 1.1, -1.2, 2.2, -0.56, -0.012, 0.21],
    [-0.5, -1.6, 1.1, -0.3, -4.05, 1.2, 0.8],
    [-0.45, 2.2, -2.3, 1.1, 1.3, 1.4, 0.2],
    [1.1, 0.11, -0.2, 4.12, 2.2, 1.8, 0.9],
]
asset_count = len(asset_observations_list)
soft_constraint = 1.33
asset_returns_list = []

port_returns = []
sharpe_ratio = []
port_variance = []
stock_weights = []

num_portfolios = 50000
max_asset_weight = 1

# Calculate the observed mean value, μi , for each asset.
print('Assets expected returns:')
for i in range(0, asset_count):
    ret = np.mean(asset_observations_list[i])
    print("Asset index", i, " ", ret)
    asset_returns_list.append(ret)

max_return = max(asset_returns_list)
'''Asset index 4 has the highest expected return of 1.432857142857143.'''

# min_weight for max return asset
min_weight = (soft_constraint / max_return)
# sum of the reminder of the weights that could be split among all 5 assets

'''If one wants to put a constraint of 1.33 the minimum weight that the 
max_return asset could have is 0.9282153539381853. 
(assuming all other assets give positive returns) 
The rest of should be split between all 5 assets'''


def rand_weights(min_w, n):
    '''Returns 1D array of weights each w2 >= 0.9282153539381853.'''
    rem_weight = 1 - min_w
    rem_rem_weights = 0
    asset_weights_list = []
    for i in range(0, n):
        asset_rem_weights = (np.random.uniform(0, rem_weight - rem_rem_weights))
        rem_rem_weights += asset_rem_weights
        if (i == n - 1):
            asset_weights_list.append(1 - sum(asset_weights_list))
        else:
            asset_weights_list.append(asset_rem_weights)
    return asset_weights_list


def portfolio_returns_variance_weights(min_, ret_):
    '''Returns expected return, volatility and weights for a given portfolio.

    Comments:
    Since not all assets give positive return the constraint of w5 => 0.9282153539381853
    is not enough. Therefore, when a portfolio that returns less than 1.33 is skipped
    and new weights are generated. However, constraint of w5 => 0.9282153539381853
    gives us a very high probability of generating a portfolio with R >=1.33.
    '''
    while True:
        mu = np.asmatrix(np.mean(ret_, axis=1))
        w = np.asmatrix(rand_weights(min_, len(ret_)))
        C = np.asmatrix(np.cov(ret_))
        R = mu * w.T
        sigma = np.sqrt(w * (C * w.T))
        if (R >= soft_constraint):
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
    returns, variance, weights = portfolio_returns_variance_weights(min_weight, asset_observations_list)
    # Risk free rate of 0% was assumed
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
Sample Output:

Assets expected returns:
Asset index 0   0.5328571428571429
Asset index 1   0.24828571428571436
Asset index 2   -0.47857142857142854
Asset index 3   0.49285714285714294
Asset index 4   1.432857142857143

Maximum Sharpe Ratio Portfolio:

Sharpe Ratio 0.9917314290645228
Variance 1.3796587243540759
Returns 1.3682509183250042
Weights:
Asset index 0 weight 0.07178448624455444
Asset index 1 weight 1.3367253227553265e-07
Asset index 2 weight 1.0597222905239526e-08
Asset index 3 weight 8.842066595940828e-09
Asset index 4 weight 0.9282153606436238

Maximum Variance Portfolio:

Sharpe Ratio 0.9838230432303288
Variance 1.3526492305152367
Returns 1.3307674823886626
Weights:
Asset index 0 weight 0.03437549985137801
Asset index 1 weight 0.00044023019322578054
Asset index 2 weight 0.03694061120236959
Asset index 3 weight 2.2219994794618712e-05
Asset index 4 weight 0.9282214387582319

Comments:

no constraints
Maximum Sharpe Ratio: 0.994
Minimum Variance: 0.557

constraints wi<0.35
Maximum Sharpe Ratio: 0.915
Minimum Variance: 0.579

constraints R>=1.33
Maximum Sharpe Ratio: 0.992
Minimum Variance: 1.354

constraints R>=1.33 and wi<0.35
Not possible

Is the new constraint violated?
It depends on what the investor's goals are and which versions of Part 1 and Part 2 are being compared.
The best result is achieved when there are no constraints.
However, an investor might choose option 3 if he favors higher returns since the Maximum Sharpe Ratios 
are so close. Furthermore, the Monte Carlo method is not 100% accurate, 
which make the difference between the Maximum Sharpe Ratio results even more insignificant.
'''
