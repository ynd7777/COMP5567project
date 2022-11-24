import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 读取数据
df = pd.read_csv('./dataset/interpolate_dataset.csv')
df.info()

# 提取特征
X = []
y = []
for row in range(5, 2140):
    df_row = df.iloc[(row - 5):row]
    # 从第6行起，将avg_price,active_addresses,google_trends,top100_coins_percent,avg_polarity前5天的数据作为特征
    # 所以有25个特征
    # 前5天的avg_price
    f1 = df_row['avg_price'].tolist()
    # 前5天的active_addresses
    f2 = df_row['active_addresses'].tolist()
    # 前5天的google_trends
    f3 = df_row['google_trends'].tolist()
    # 前5天的top100_coins_percent
    f4 = df_row['top100_coins_percent'].tolist()
    # 前5天的avg_polarity
    f5 = df_row['avg_polarity'].tolist()
    featues = f1 + f2 + f3 + f4 + f5
    X.append(featues)
    # 因变量
    y.append(df['avg_price'].iloc[row])
# 将X转换为数据框
cols1 = ['avg_price_' + str(i) for i in range(1, 6)]
cols2 = ['active_addresses_' + str(i) for i in range(1, 6)]
cols3 = ['google_trends_' + str(i) for i in range(1, 6)]
cols4 = ['top100_coins_percent_' + str(i) for i in range(1, 6)]
cols5 = ['avg_polarity_' + str(i) for i in range(1, 6)]
X = pd.DataFrame(X, columns=cols1 + cols2 + cols3 + cols4 + cols5)
# 将y转换为Series
y = pd.Series(y)

# 划分数据集，共2135个样本，后427个数据作为测试集，前1708作为训练集
X_train = X.iloc[:1708]
X_test = X.iloc[1708:]
y_train = y.iloc[:1708]
y_test = y.iloc[1708:]

# 对特征进行z-score标准化
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)
X_z = pd.DataFrame(X, columns=X.columns)

# 定义参数区间
param = {'n_estimators': range(100, 600, 100),
         'max_depth': range(2, 10)}
rf = RandomForestRegressor(random_state=0)
# 5折交叉验证寻找最佳参数
reg = GridSearchCV(rf, param, cv=5, n_jobs=-1, scoring='neg_root_mean_squared_error')
reg.fit(X_train, y_train)

'''param = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': [1, 5, 10], 'degree': [3, 8], 'coef0': [0.01, 10, 0.5],
         'gamma': ('auto', 'scale')}

svr = SVR()

grids = GridSearchCV(estimator=svr, param_grid=param,
                     cv=3, n_jobs=-1, verbose=2)

grids.fit(X_train, y_train)
y_pred = grids.predict(X_test)
svr.fit(X_train, y_train)
print('R2 score: {:.2f}'
      .format(svr.score(X_test, y_test)))'''

# 交叉验证rmse
print(-reg.best_score_)
# 最佳参数
print(reg.best_params_)

# 预测测试集
pred_test = reg.predict(X_test)
compare = df.iloc[-427:].copy()
compare['predicted_avg_price'] = pred_test
# 输出结果到excel文件
compare.to_excel("compare.xlsx", index=False)
# 计算测试集rmse
print(mean_squared_error(y_test, pred_test, squared=False))
print(mean_squared_error(y_test, pred_test, squared=True))
print(r2_score(y_test, pred_test), '\n')

'''pred_test = svr.predict(X_test)
print(mean_squared_error(y_test, pred_test, squared=False))
print(mean_squared_error(y_test, pred_test, squared=True))
print(r2_score(y_test, pred_test))
'''