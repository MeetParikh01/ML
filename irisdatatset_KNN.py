from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
iris_dataset=load_iris()
#print(iris_dataset['data'])
print('===========',iris_dataset.data.shape)
feature_names=iris_dataset.feature_names
print(feature_names)
target_names=iris_dataset.target_names
data=iris_dataset.data
target=iris_dataset.target
print(target_names[2])

print(data[:5])

x_train,x_test,y_train,y_test=train_test_split(iris_dataset["data"], iris_dataset["target"], test_size=0.25,random_state=0)
#print(x_train)
kn = KNeighborsClassifier(n_neighbors=1)

kn.fit(x_train, y_train)

print('hi=',kn.fit(x_train, y_train))
joblib.dump(kn.fit(x_train, y_train), "model1.pkl")
#print(iris_dataset)
x_new = np.array([[6.3,3.3,6,2.5]])
print(x_train.shape)
print(y_test.shape)

print(x_test.shape)
#print(x_new)
prediction = kn.predict(x_new)
y_pred=kn.predict(x_test)
print("Predicted target value: {}\n".format(prediction))
print("Predicted feature name: {}\n".format
    (iris_dataset["target_names"][prediction])) 
print("Test score: {:.2f}".format(kn.score(x_test, y_test)))
print(metrics.accuracy_score(y_test, y_pred))
po=np.array([[3,4],[5,6]])
print(po)