import pandas as pd
import numpy as np
#%%
iris_df=pd.read_csv("D:/Projects/NeuralNetworks/iris.train.csv")
iris_df.head()
iris_df.shape
iris_df.dtypes
iris_df.describe(include="all")
#%% Giving headers to the columns
iris_df.columns=['sepal length','sepal width','petal length','petal width','class']
iris_df.head(10)
#print(iris_df)
#%% Creating a copy of dataframe
iris_df_rev=pd.DataFrame.copy(iris_df)
iris_df_rev.describe(include='all')
#%%converting Depending variable(i.e class in this case)categorical to numerical by using Label encoding technique
colname=['class']
colname
from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()#convert cat to num by label encoding
    
for x in colname:
    iris_df_rev[x]=le[x].fit_transform(iris_df_rev.__getattr__(x))
    
iris_df_rev.head(60)
#0-->setosa
#1-->versicolor
#2-->virginica
#%% Separating into Dependent and Independent variables
#Considering all rows and all columns excluding last variable(i.e dependent variable)
X=iris_df_rev.values[:,:-1]
#considering all rows and column(dependent variable)
Y=iris_df_rev.values[:,-1]
print(X)
Y=Y.astype(int)
Y


#%% Scaling the variables to avoid model biasing using Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() 
scaler.fit(X)
X=scaler.transform(X)
print(X)
#%%Splitting variables into Training & testing 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression())
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
#Comparing Testing and predicted values
print(list(zip(Y_test,Y_pred)))
#%%
#Calculating Accuracy ,confusion Matrix,classification report
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report:")
print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",accuracy_score)


#%%User/Manual i/p
list1=np.array([4.5,3.2,2.3,5.4])
#converting to 1D array as it is  in 2D
list1=list1.reshape(1,-1) 
 #Tells type of flower
Y_pred=classifier.predict(list1)
print(Y_pred)   