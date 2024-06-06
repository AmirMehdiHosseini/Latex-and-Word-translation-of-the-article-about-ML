import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
#print(data.DESCR)
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = svm.SVC(kernel='rbf')
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(accuracy)