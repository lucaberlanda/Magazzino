from Finance.Portfolio.stocks import Stock

rets = Stock(risky_tk).price().loc[:, 'Adj Close'].pct_change()
print()
