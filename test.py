from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

kn=joblib.load('model.pkl')
print(kn)
x_new = np.array([[0,0,0,0]])
prediction = kn.predict_proba(x_new)
print(prediction)