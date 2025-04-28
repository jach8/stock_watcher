# Vol Modeling:

# Vol Skew:

```python
sql:

AVG(
	CASE WHEN t2.type = "Put" and t2.moneyness < 0.95 
	then t2.iv 
	else null end
	) - 
AVG(
	CASE WHEN t2.type = "Call" and t2.moneyness > 1.05 
	then t2.iv
	else null end
) 
as vol_skew

psuedo: 
Avg_otm_put_iv / Avg_otm_call_iv
```

Volatility Skew is the Difference between the Average Call and Put OTM Implied Volatility. 

## Positive Volatility Skew:  

- OTM Puts have a higher IV than OTM Calls.
- This indicates a **reverse skew**, where investors demand higher premiums for downside protection due to the fear of a steep decline.
- Implies: Higher demand for puts.

## Negative Volatility Skew:

- OTM Calls have a higher IV than OTM puts
- Indicates forward skew, markets are anticipating upside price movement.
- Implies: Speculative Call Buying

## Near Zero Volatility Skew: 

- Symmetric Volatility Smile: Balanced Expectations for upside and downside.
- Implies no strong bias in the option market.

## Volatility Skew Across different Time until expiration: 

1. Short Term: High positive volatility skew may signal imminent downside fear. 
2. Long term: A persistent positive volatility skew reflects structural hedging. 
3. Medium Term: 

## Market Sentiment: 

1. High positive Vol_skew: ( Greater than 0.10) 
    1. Suggest bearish sentiment or hedging activity
    2. Compare with Put/Call ratio or Put Volume Percent to reinforce a downside concern
2. High negative Vol_Skew (Less then -0.10)
    1. Signal bullish speculation 
    2. Compare with call_vol_pct, otm_call_vol, to confirm the speculation 

## Trading Implications: 

1. Positive volatility skew: 
    1. Opportunities: 
        1. Selling OTM Puts if you believe that the fear is overblown. 
            1. If you expect a reversal.
        2. Buying puts will be expensive; consider a spread to reduce the cost. 
2. Negative volatility skew: 
    1. Sell OTM Calls if upside IV is overblown
        1. If you expect that the rally will fade. 
    2. Calls will be expensive. You can use spread to lower entry cost. 

## Risk Assessment: 

1. Compare Volatility Skew with IV Rank: 
    1. High Volatility skew and High IV rank: 
        1. Strong downside fear. (over-crowded trade) 
    2. Low Volatility skew and Low IV rank: 
        1. Here the probability that a surprise move is high. Calls and puts are likely cheap. 

## Temporal Trend Assessment: 

1. Track the Volatility skew over time. 
    1. Rising skew → increasing fear, the market is expecting an event. 
    2. Falling skew → Potential for a relief rally. 

## Enhancing the interpretation: 

1. Normalize the skew: 
    1. Calculate the mean, max, min to get a bench mark 
        1. High Relative Skew (stock skew - index skew) : suggest a stock specific downside risk.