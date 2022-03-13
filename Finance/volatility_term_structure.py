from Finance.Portfolio.stocks import Stock

risky_ri = Stock(risky_tk).price().loc[:, 'Adj Close']