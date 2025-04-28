""" 
    Base Class for Calculating Financial Ratios  and Valuation Metrics
    1. Gross Margins: Based on Gross Profit and Revenue
    2. Operating Margins: Based on Operating Income and Revenue
    3. Net Margins: Based on Net Income and Revenue
    4. EPS: Based on Net Income and Shares Outstanding
    5. P/E: Based on Current Stock Price and EPS
    6. Current Ratio: Based on Current Assets and Liabilities
    7. Intrinsic Value: Based on Future Dividends/Cashflows. 
    8. Book Value: Based on Current Assets and Liabilities
    9. Market Cap: Based on Current Stock Price and Shares Outstanding
    
"""

import numpy as np 
import pandas as pd
from pandas_datareader import data as wb