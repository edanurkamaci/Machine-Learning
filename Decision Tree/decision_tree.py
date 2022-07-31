from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('decision+tree+regression+dataset.csv',sep=';',header=None)

x=df.iloc[:,0].values.reshape(-1,1) #tribune level 
y=df.iloc[:,1].values.reshape(-1,1) #price that a spectator has to pay


tree_reg= DecisionTreeRegressor()
tree_reg.fit(x,y)

print(tree_reg.predict([[7]]))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=tree_reg.predict(x_)

plt.scatter(x,y,color='red')
plt.plot(x_,y_head,color='green')
plt.xlabel("Tribune Level")
plt.ylabel("Price")
plt.show()

