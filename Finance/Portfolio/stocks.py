import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials


class Stock:

    def __init__(self, tk):
        self.tk = tk
        self.ticker = yf.Ticker(tk)

    def price(self):
        price = yf.download(self.tk, progress=False)
        return price

    def info(self):
        return self.ticker.info

    def inst_holders(self):
        return self.ticker.institutional_holders
