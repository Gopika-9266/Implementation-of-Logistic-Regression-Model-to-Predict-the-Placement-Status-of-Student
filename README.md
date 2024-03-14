# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## Aim:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Gopika R
RegisterNumber: 212222240031

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1) # removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test , y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1= classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]) 

```

## Output:


### Placement score:
![ml exp4-1](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/034f1378-a5b2-4855-a4e1-99a04180ab7f)

### Salary data:
![ml exp4-2](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/c7384342-7a4a-438d-ba8b-b9eb483886ef)

### Checking the null function():
![ml exp4-3](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/f51b440b-8420-4b4d-b7d2-ecc80326895c)

### Duplicated data:
![ml exp4-4](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/10af4c42-e68f-4531-8a20-a4ea4cab3182)

### Print data:
![ml exp4-5](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/55320a8d-eed6-4ef3-b2ef-4d5e877f1b70)

### Data Status:
![ml exp4-6](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/ec574948-e0b0-416f-82d0-008df2935ea4)

### y-predicted array:
![ml exp4-7](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/cc871561-f133-41ff-b2f6-d812f62efce9)

### Accuracy Value:
![ml exp4-8](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/22fa9a87-fb3b-4d58-abb4-f2033ecedf1c)

### Confusion matrix:
![ml exp4-9](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/9a138e90-8934-434c-8327-fd5fc1ecaf45)


### Classification report:
![ml exp4-10](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/506b3bb9-40d5-46b2-8d64-a27a356bc267)
### Prediction of LR:
![ml exp4-11](https://github.com/Gopika-9266/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122762773/5b05a3ca-3cad-4bca-a260-036f2e3ee11a)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
