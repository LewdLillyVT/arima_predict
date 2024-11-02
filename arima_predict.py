import sys
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def main():
    
    data = json.loads(sys.stdin.read())
    prices = np.array(data['prices'])

    
    model = ARIMA(prices, order=(5, 1, 0))
    model_fit = model.fit()

    
    forecast = model_fit.forecast(steps=30)

    
    print(json.dumps(forecast.tolist()))

if __name__ == "__main__":
    main()
