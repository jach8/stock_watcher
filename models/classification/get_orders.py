import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .clda import ClassificationModel
from ....stockData import StockData
from ....main import Manager
from ...utils.tools import encode_orders, pretty_print
from ....backtest.strategies import Policy


def get_sd(stock, manager = Manager):
    sd = StockData(stock=stock, manager=manager, cache_dir="../../../data_cache")
    sd.clear_cache(disk=True, stock_specific=False)
    df = sd.get_features().replace(0, np.nan).dropna()
    return df 


def get_stock_data(stock, manager = Manager):
    df = get_sd(stock, manager)
    df['target'] = df['close'].pct_change().shift(-1)
    df = df.dropna()
    x = df.drop(columns=["close", "open", "high", "low","target"])
    y = df["target"]
    X_new = x.tail(1)
    X = x.dropna()
    y = pd.Series(np.where(y > 0, 1, 0), name='target', index = y.index)
    # Select 15 random features
    selected_features = np.random.choice(x.columns, size=15, replace=False)
    x = x[selected_features]
    print(f'df shape: {df.shape} X shape: {X.shape}, y shape: {y.shape}, X_new shape: {X_new.shape}')
    return x, y, X_new


def fit_models(stock, manager = Manager):
    # Get stock data
    X, y, X_new = get_stock_data(stock, manager)

    # Initialize and run the model
    model = ClassificationModel(
        X=X,
        y=y,
        numerical_cols=X.columns.tolist(),
        verbose = 0, 
        time_series=True
    )
    model.preprocess_data()
    model.train_models()

    # Display results
    results = model.get_results()
    print("\nModel Performance Results:")
    # print(results)

    next_prediction = model.predict_new_data(X_new)
    return model


def get_orders(models, stock = 'spy'):
    o = {}

    for x in models.keys():
        preds = models[x]
        orders = encode_orders(predictions = preds.values, test_index=preds.index, stock = stock, shares = 10, name = x)
        o[x] = orders

    orders = []; names = []
    for x in o.keys():
        orders.append(o[x])
        names.append(x)
    
    return orders, names 


def evaluate_orders(orders, names, policy = Policy):
    res = policy.eval_multiple_orders(
        orders = orders,
        names = names, 
        sv = 10000, 
        commission = 1.0, 
        impact = 0.0005
    )

    sim_results = policy.list_eval.copy()
    more_stats = []
    for key in sim_results.keys():
        more_stats.append(p._qs(name = key, portvals=sim_results[key]['portfolio']).T )
    more_stats = pd.concat(more_stats, axis=1)
    return res, more_stats


def main(stock, manager = Manager, policy = Policy):
    # Fit models
    model = fit_models(stock, manager)
    models = model.model_predictions
    # Get orders
    orders, names = get_orders(models, stock)
    # Evaluate orders
    res, more_stats = evaluate_orders(orders, names, policy)
    # Print results
    pretty_print(res)
    # pretty_print(more_stats)
    return model, res, more_stats


