import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv(r'D:\semA\COMP 5567\5567project\dataset\dataset.csv')
df.index = pd.to_datetime(df['date'])
for feature in ['avg_price', 'active_addresses', 'google_trends', 'top100_coins_percent', 'avg_polarity']:
    # df['avg_price'].plot(ylabel='bitcoin average price')
    sns.lineplot(data=df[feature])
    plt.title(f"Bitcoin {feature} historical chart")
    plt.savefig(rf"D:\semA\COMP 5567\5567project\picture\{feature}.png",dpi=200)
    plt.close()
