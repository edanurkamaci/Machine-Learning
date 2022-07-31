"""After test of a model, according to results, knowing the value of accuracy is not just enough
For example u classificated dog and cat and made a test. Altough the accuracy is %99, all dogs were not
predicted right, so we should know these infos that means we need confusion matrix """

#author=@eda
#For the classification, i am gonna use random forest algorithm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot  as plt

data= pd.read_csv('data(for_confusin_matrix).csv')
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

data.diagnosis=[1 if each=='M'else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)

rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
print("Random Forest Algroithm Result: ",rf.score(x_test,y_test))

y_pred=rf.predict(x_test)
y_true = y_test

cm=confusion_matrix(y_true,y_pred)

f, ax =plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor='red',fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
