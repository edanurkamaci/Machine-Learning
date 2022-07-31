import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
#Read Data in csv filr
df =pd.read_csv('linear_regression_dataset.csv', sep=";") #used sep because of the default comma


#Ploting Datas
plt.scatter(df.experience,df.salary)
plt.xlabel("experience")
plt.ylabel("salary")
plt.title("Salary According to Experience")
plt.show()

#Linear regression model
linear_reg= LinearRegression()

x=df.experience.values.reshape(-1,1) #(x,)-->(x,1)
y=df.salary.values.reshape(-1,1)

linear_reg.fit(x,y)

#prediction  formula of linear regression --> ax + b --> a: coefficient / b: constant

b_=linear_reg.predict([[0]])
print("b: ",b_)

b_=linear_reg.intercept_
print("b: ",b_)

a_=linear_reg.coef_
print("a: ",a_)

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).reshape(-1,1)


plt.scatter(x,y)


y_head=linear_reg.predict(array)
plt.plot(array,y_head,color="red")
plt.show()

print("salary of 30 year experience: ",linear_reg.predict([[30]]))

