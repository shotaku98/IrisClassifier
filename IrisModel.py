import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import tree
import seaborn as sns


df=pd.read_csv("C:/Productivity/Current Work/Iris/Iris.csv")
#print(df['Species'].value_counts())

sns.pairplot(df,hue='Species',markers='o')
#plt.show()

X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

#Splitting dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5 )
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)

clf= tree.DecisionTreeClassifier()

clf=clf.fit(X_train,y_train)

predictions = clf.predict(X_test)

from sklearn import metrics

print(metrics.accuracy_score(y_test, predictions))


#Gaussian Naive Bayes


from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb=gnb.fit(X_train,y_train)

Gpredictions=gnb.predict(X_test)

print(metrics.accuracy_score(y_test, Gpredictions))

#Perceptron

from sklearn.linear_model import Perceptron

model = Perceptron()

model.fit(X_train,y_train)

LRprediction=model.predict(X_test)

print('The accuracy of the Perceptron is',metrics.accuracy_score(LRprediction,y_test))



