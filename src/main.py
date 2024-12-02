import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime
import math
import json


class Desision(object):
    def __init__(self) -> None:
        self.t = 0
        self.m = 0
        self.n = 0
        self.transport_price = 0
        self.oil_prices = None
        self.P_predictor = None
        self.storages = None
        self.C = None
        self.X = None
        
    def read_json(self, path: str) -> None:
        with open(path, "r") as file:
            data = json.load(file)
            
            self.oil_prices = data['oil_prices']
            self.storages = data['storages']
            self.C = data["oils_info"]['C']
            self.X = data["oils_info"]['X']
            self.n, self.m, self.t, self.transport_price = data['input']
            
    def fit_predictor_P(self) -> None:
        self.P_predictor = list()
        
        for i in range(self.t):
            
            X, y = self.oil_prices[i]
            
            dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in X]
            epoch = datetime(2000, 1, 1)
            
            X = np.array([(date - epoch).days for date in dates]).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            self.P_predictor.append((model.coef_[0], model.intercept_))
            X = X[::50]
            y = y[::50]
            y_pred = model.predict(X)

            plt.scatter(
                X,
                y,
                label=f"oil {i + 1}")
            plt.plot(
                X,
                y_pred,
                label=f"predictor {i + 1}: {round(self.P_predictor[-1][0], 3)}t + {round(self.P_predictor[-1][1], 3)}",
                linestyle="--")
          
        plt.xlabel("number days after 2000-01-01")
        plt.ylabel("Price per one barrel ($)")
        plt.legend()
        plt.savefig("P predictors.png")
        plt.close()
        
    def bisect(self,
               f_str: str,
               tol: float = 1e-6):
    
        def f(t):
            return eval(f_str)
        a = 0
        b = a + 1
        while f(a) * f(b) >= 0:
            b *= 2

        while (b - a) / 2.0 > tol:
            c = (a + b) / 2.0
            if f(c) == 0:
                return c
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return (a + b) / 2.0
    
    def integrate_monte_carlo(self, f_str : str, T: float):
        n = round(10**4 * T)

        def f(t):
            return eval(f_str)

        random_points = [np.random.random() * T for _ in range(n)]

        func_values = [f(x) for x in random_points]

        mean_value = sum(func_values) / n
        return T * mean_value
    
    def get_desision(self) -> list[tuple]:
        profit_facilies = [None] * self.m
        for idx, (X, d, t) in enumerate(self.storages):
            coeffs = self.P_predictor[t - 1]
            C_func = self.C[t - 1]
            X_func = self.X[t - 1]
            
            def C(t):
                return eval(C_func)
            
            T = self.bisect(f"{X} - {X_func}")
            integral_mean = self.integrate_monte_carlo(X_func, T)
            profit_facilies[idx] = (X*(coeffs[1] + coeffs[0] * T - d * self.transport_price) - \
                integral_mean - C(T), str(idx))
            print(profit_facilies[idx])
               
        return sorted(profit_facilies, key=lambda x: x[0], reverse=True)[:self.n]
        
            
    
    

if __name__ == "__main__":
    des = Desision()
    des.read_json("data.json")
    des.fit_predictor_P()
    
    idxs = [0] * des.n
    profits = [0.0] * des.n
    
    for i, (profit, idx) in enumerate(des.get_desision()):
        idxs[i] = idx
        profits[i] = profit
    
    plt.bar(idxs, profits, label=idxs)
    plt.title(f"Profit of {des.n} the best storages. Overall profit = {round(sum(profits), 2)}")
    plt.xlabel('profit')
    plt.ylabel('idx of storage')
    plt.savefig("result.png")  
        

    
        
        
        