import backtrader as bt
import yfinance as yf
 
class SmaCross(bt.Strategy):
    pfast=10  # Period for the fast moving average
    pslow=30   # Period for the slow moving average
   
    def __init__(self):
        sma1 = bt.ind.SMA(period=self.pfast)
        sma2 = bt.ind.SMA(period=self.pslow)
        self.crossover = bt.ind.CrossOver(sma1, sma2)
 
    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

cerebro = bt.Cerebro() # Create a Cerebro engine
dataframeAAPL = yf.download('AAPL', start='2000-10-01', end='2024-11-30', interval = '1d', group_by = 'column', auto_adjust = True)
data = bt.feeds.PandasData(dataname=dataframeAAPL)
cerebro.adddata(data) # Add the data feed to Cerebro
cerebro.addstrategy(SmaCross) # Add the strategy to Cerebro
cerebro.broker.set_cash(100000) # Set the initial cash amount for the backtest
print('Starting Portfolio Value:', cerebro.broker.getvalue()) # Print the starting cash amount
cerebro.run() # Run the backtest
print('Ending Portfolio Value:', cerebro.broker.getvalue()) # Print the final cash amount after the backtest
cerebro.plot() # Plot the results with a single command

