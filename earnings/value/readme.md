This module focuses on Value Investing by evaluating companies based on their Balance Sheet, Income Statements, and Cash Flow statements. 
- We will also use these statements to provide appropriate ratio and valuation metrics. 


# Metrics we will consider: 
## **Gross Margins**: 
 1. Each $1 of revenue is more valuable to the business. 
 2. Higher Gross Margin means a company retains more capital. 
$$ \text{Gross Margin} = \frac{\text{Revenue} - \text{Cost of Goods Sold}}{\text{Revenue}} $$

## **Operating Margin**:
1. How much profit a company makes on a dollar of sales after paying for variable costs of production such as wages and raw materials, but before paying for interest or tax. 
2. Higher Operating Margin means the company is efficient in its operations and good at turning sales into profits. 
3. How efficiently a company is able to generate profits through core operations. 
4. When comparing companies OP margin, those that have higher margins compared to the industry are said to have a **Competitive Advantage**
5. Only to be used with companies in the same industry. 
$$ \text{Operating Margin} = \frac{\text{Operating Income (Earnings)}}{\text{Sales (Revenue)}} $$

## **EPS (Earnings Per Share)**:
1. Used to Draw Conclusions about a company's earnings stability over time, financial strength, and growth potential. 
2. How much money a company makes for each share of its stock, and widely used metric for estimating corporate value. 
3. Higher EPS indicates greater value because investors will pay more for a company's shares if they think the company has higher profits to its share price. 
$$ \text{EPS} = \frac{\text{Net Income} - \text{Dividends on Preferred Stock}}{\text{End Of Period Common Shares Outstanding}} $$    

## **Price To Earnings Ratio**
1. Measures the Company's Share price relative to earnings per share. 
2. Helps assess the company's value, and comparing it to other companies, the market, or the industry.
3. Also helps determine if the company is overvalued or undervalued.
4. A few ways P/E is calculated: 
   1. Forward P/E: 
      1. Uses Future Earnings Guidance rather than trailing figures. 
      2. Helps compare current earnings to future earnings (Guidance)
   2. Trailing P/E:
      1. Relies on Past performance by dividing the current share price by the total EPS for the last 12 months.
      2. Most Popular because it uses historical earnings. 

   $$ \text{P/E Ratio} = \frac{\text{Share Price}}{\text{Earnings Per Share}} $$

## **Current Ratio**:
1. Liquidity Ratio that measures whether a firm has enough resources to meet its short-term obligations. 
2. Shows how welll the company is able to maximize its current assets to satisfy current debt and other payables. 
$$ \text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}} $$

## **Intrinsic Value**:
1. Value of a company based on its dividends, earnings, and growth rate.
   1. Based on the time-value of money: 
   $$ \text{PV} = \frac{\text{FV}}{(1 + i)^t} $$
2. The higher the certainty that a company can pay the dividend, the lower the discount rate. 
3. The lower the certainty the company can pay the dividend, the higher the discount rate. 
4. To find the Intrinsic value, find the present value of a perpituity, using the present value of dividends. 
$$ \text{Intrinsic Value} = \frac{\text{Dividend}}{\text{Discount Rate}} $$


## **Book Value**:
1. Value of a Company based on assets the company owns. 
2. If Market Cap $<$ Book value: Buy the company 
   1. *Stocks rarely go below book value*
3. If intrinsic value drops significantly while market cap stays high, It's a good time to load the Puts.
   1. Vice Versa
4. Book Value is **the lowest price you will buy an asset at**
   1. Stocks rarely approach book value, so if they do it may be a good time to buy. 
5. Negative Book Value Happens when a company has more liabilities than assets. 
$$ \text{Book Value} = \text{Total Assets} - \text{(Intangible Assets and Liablities)} $$

## **Market Capitalization**:
1. The Market Decides Company Value.
2. Based on Intrinsic Value, Book Value, and Market Capitalization 
$$ \text{Market Cap} = \text{Share Price} \times \text{Total Shares Outstanding} $$


## Compound Annual Growth Rate (CAGR):
1. This is the annual growth rate for the geometric series, that provides a constant rate of return over the time period.

$$ \text{CAGR} = \left( \frac{\text{Ending Value}}{\text{Beginning Value}} \right)^{\frac{1}{\text{Number of Years}}} - 1 $$

## Free Cash Flow to the firm (FCFF):
1. Cashflows from assets produced by the firm 
2. If they are discounted you end up with intrinsic value of of the assets. 
3. Appropriate Discount rate is the required return of the firm (WACC)
$$ \text{FCFF} = \text{EBIT} \times (1 - \text{Tax Rate}) + \text{Depreciation} - \Delta\text{NWC}  - \text{Capital Expenditure} $$

## Weighted Average Cost of Capital (WACC):
1. The average rate of return a company is expected to pay to all its security holders to finance its assets.
2. Weights are the proportions of debt and equity from the company's balance sheet. 
$$ \text{WACC} = W_D \times R_D \times (1 - \text{T}) + W_E \times R_E $$
1. Where:
   1. $W_D$ = Weight of Debt
   2. $R_D$ = Required Return of Debt (YTM of Debt)
   3. $R_E$ = Required Return for Equity (CAPM Model or Dividend Discount Model)
   4. $T$ = Marginal Tax Rate
   5. $W_E$ = Weight of Equity
   6. $R_E$ = Cost of Equity


## Free Cash Flow to Equity (FCFE):
1. Cashflows left over to equity holders after investment in net working capital and capital expenditures have occured as well as payments to and from debt holders. 
2. if they are discounted you end up with the intrinsic value of the equity.
3. Appropiate Discount rate is the required return to equity holders. 
$$ 
\text{Net Income} = \text{EBIT} - \text{Interest} - \text{Taxes} \\
\text{FCFE} = \text{Net Income} + \text{Depreciation} -  \Delta\text{NWC} \\
- \text{Net Capital Expenditure} + \text{New Debt Issued} - \text{Debt Repayment}
$$