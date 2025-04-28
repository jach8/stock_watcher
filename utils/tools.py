
import pandas as pd 
import numpy as np 
from IPython.display import display 


def pretty_print(df):
    """ Only convert numbers to 2 decimal places with a comma separator """
    return display(df.map(lambda x: "{:,.2f}".format(x) if isinstance(x, (int, float)) else x))
    

def human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        # if .0 is the decimal part, remove it
        if num == int(num):
            return '%d%s' % (np.abs(num), ['', 'K', 'M', 'B'][magnitude])
        else:
             
            return '%.1f%s' % (np.abs(num), ['', 'K', 'M', 'B'][magnitude])


def encode_orders(predictions, test_index, stock, shares = 100 , thresh = 0.003, name = None):
    """ 
    Return Orders Dataframe, predictions can either be binary or continuous
    
    Args: 
        predictions: array-like, shape (n_samples, )
            The predictions of the model. 
        test_index: array-like, shape (n_samples,)
            The index of the test data.
        stock: st
            The stock symbol.
        shares: int
            The number of shares to buy or sell.
        thresh: float
            The threshold to buy or sell.
        name: str
            The name of the model.
    """
    if predictions.dtype == 'int':
        preds = np.where(predictions == 1, "BUY", np.where(predictions == -1, "SELL", "HOLD"))
    else:
        preds = np.where(predictions > thresh, "BUY", np.where(predictions < -thresh, "SELL", "HOLD"))
    out = pd.DataFrame(preds, index = pd.to_datetime(test_index), columns = ['Order'])
    out['Symbol'] = stock
    out['Shares'] = shares
    
    # # Close orders after 2 days
    # closed = out.copy()
    # closed['Order'] = np.where(closed['Order'] == "BUY", "SELL", np.where(closed['Order'] == "SELL", "BUY", "HOLD"))   
    # closed.index = closed.index + pd.tseries.offsets.BDay(5)
    
    # out = pd.concat([out, closed]).sort_index()
    if name is None:
        return out[['Symbol', 'Order', 'Shares']]
    else:
        out["Name"] = name
        return out