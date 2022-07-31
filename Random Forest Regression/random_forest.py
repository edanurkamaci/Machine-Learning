"""
The Random Forest regression algorithm is the mean of the decision trees, 
so the difference of the random forest from the decision tree is Random forest's use of decision trees more than once
means that the results of the random forest may be more accurate than the decision tree.

It is usually used 
to suggest something like netflix suggest u what u can watch according to 
movies that u watched
"""

from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('random+forest+regression+dataset.csv',sep=';',header=None)

x=df.iloc[:,0].values.reshape(-1,1) #distance to futboll field (tribune level)
y=df.iloc[:,1].values.reshape(-1,1) #price according to distance

#                                 number of trees
randomF_reg=RandomForestRegressor(n_estimators=100,random_state=42)
randomF_reg.fit(x,y)

print("price of level at 7.8: ",randomF_reg.predict([[7.8]]))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=randomF_reg.predict(x_)

plt.scatter(x,y,color='red')
plt.plot(x_,y_head,color='green')
plt.xlabel('Tribune Level')
plt.ylabel('Price')
plt.show()




