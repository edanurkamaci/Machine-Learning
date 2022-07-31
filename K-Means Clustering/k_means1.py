from turtle import color
from cv2 import kmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#  CREATING DATASET
#class1 
x1=np.random.normal(25,5,1000) #1000 numbers between 20-30
y1=np.random.normal(25,5,1000)

#class2
x2=np.random.normal(55,5,1000)
y2=np.random.normal(60,5,1000)

#class3
x3=np.random.normal(55,5,1000)
y3=np.random.normal(15,5,1000)

x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)

dictionary={"x":x,"y":y}

data=pd.DataFrame(dictionary)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.title("Datas which k means cannot see the classes")

plt.show()

plt.scatter(x1,y1,color='black')
plt.scatter(x2,y2,color='black')
plt.scatter(x3,y3,color='black')
plt.title("Datas How k means algorithm see at the first time")

plt.show()

#IMPLEMENTING K MEANS ALGORITHM

wcss=[]

for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)
plt.xlabel("number of k(cluster) value")
plt.ylabel("wcss")

plt.show()

# according to graph n_cluster selected as 3

kmeans2=KMeans(n_clusters=3)
clusters=kmeans2.fit_predict(data)

data["label"]=clusters

plt.scatter(data.x[data.label==0],data.y[data.label==0],color='red')
plt.scatter(data.x[data.label==1],data.y[data.label==1],color='green')
plt.scatter(data.x[data.label==2],data.y[data.label==2],color='yellow')

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="blue")

plt.title("After K Means Algorithm")
plt.show()

