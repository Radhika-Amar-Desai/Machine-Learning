"""
    IMPORT MODULES
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.svm import SVC
"""
    DEFINE FUNCTIONS
"""
def plot(dataframe,para1,para2,color):
    plt.scatter(dataframe[para1],dataframe[para2], c = color)
"""
    GENERATING LINEARLY SEPERABLE DATASET AND PLOTTING IT
"""
# generating data
X,y = make_blobs(n_samples=100,centers=2,random_state=0,cluster_std=0.4)

#plotting
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))

colors = {0:'blue', 1:'red'}
label_list = df["label"].unique()

for i in label_list:
    plot(df[df["label"] == i],"x","y",colors[i])
plt.show()
"""
    APPLYING SVC AND PLOTTING IT
"""
#applying svc
svc_model = SVC(kernel='linear')
svc_model.fit(X,y)

#plotting 

# getting the x and y points
w = svc_model.coef_[0]           # w consists of 2 elements
b = svc_model.intercept_[0]      # b consists of 1 element
x_points = np.linspace(min(X[:,0])-1, max(X[:,1])+1)    # generating x-points from -1 to 1

# maximum margin hyperplane
y_points_max_margin = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
for i in label_list:
    plot(df[df["label"] == i],"x","y",colors[i])
plt.plot(x_points, y_points_max_margin, c='black')

# positive hyperplane
y_points_positive = -(w[0] / w[1]) * x_points - (b+1) / w[1]  # getting corresponding y-points
for i in label_list:
    plot(df[df["label"] == i],"x","y",colors[i])
plt.plot(x_points, y_points_positive, c='black',linestyle = "dashed")

# negative hyperplane
y_points_negative = -(w[0] / w[1]) * x_points - (b-1) / w[1]  # getting corresponding y-points
for i in label_list:
    plot(df[df["label"] == i],"x","y",colors[i])
plt.plot(x_points, y_points_negative, c='black',linestyle = "dashed")
plt.show()