# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries
2. Set variables for assigning dataset values
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict regression for marks by representing in a graph
6. Compare graphs and hence linear regression is obtained for the given datas.

## Program:
## Developed by : SUBASH M
## Register No : 212223040210
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train, regressor.predict(x_train),color='blue') 
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train, regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE=',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Head values
![image](https://github.com/user-attachments/assets/61fea6a0-7f5b-4ec2-863f-a6a5a12f5867)
Tail values
![image](https://github.com/user-attachments/assets/ab8007a5-b14e-438b-8b44-7bb667934350)
Compare Dataset
![image](https://github.com/user-attachments/assets/f45375ff-6260-45ca-ad45-868e39f1836b)
Predicted Values
![image](https://github.com/user-attachments/assets/5720bbb1-a6ab-4d89-86d9-ca0b8ca2705e)
Training set
![image](https://github.com/user-attachments/assets/33b070ee-1ca0-4b55-8c79-0382740fe9f5)
Testing set
![image](https://github.com/user-attachments/assets/f520a8c0-5fd8-4e14-9cd9-2d1f9fd61559)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
