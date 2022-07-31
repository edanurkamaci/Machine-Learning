import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("polynomial+regression.csv",sep = ";")

y = df.max_car_speed.values.reshape(-1,1)
x = df.car_price.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("car_max_speed")
plt.xlabel("car_price")
#plt.show()

# linear regression =  y = b + a*x
# multiple linear regression   y = b + a1*x1 + a2*x2

# %% linear regression
lr = LinearRegression()

lr.fit(x,y)

#%% predict
y_head = lr.predict(x)

plt.plot(x,y_head,color="red",label ="linear")
#plt.show()

print("prediction of car which costs 10 billion TL: ",lr.predict([[10000]]))


# %%
# polynomial regression =  y = b + a1*x +a2*x^2 + a3*x^3 + ... + an*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(x)


# %% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

# %%

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.show()