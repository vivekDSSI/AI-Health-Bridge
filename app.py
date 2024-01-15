 import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#loading the diabetes dataset to a pandas dataframe
diabetes_dataset = pd.read_csv("/content/drive/MyDrive/new_diabetics.csv")
diabetes_dataset.describe()
diabetes_dataset['diabetes'].value_counts()
diabetes_dataset.groupby('diabetes').mean()
X = diabetes_dataset.drop(columns = 'diabetes',axis=1)
Y = diabetes_dataset['diabetes']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['diabetes']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
print("enter 0 for female 1 for male and 2 for others")
print("enter age")
print("enter 1 if have hypertension else enter 0")
print("enter 1 if have heart_disease else 0")
print("enter 1 if you smoke else 0")
print("enter bmi")
print("enter HbA1c_level")
print("enter blood glucose")
input_data = list(map(float, input("Enter values for above parameters: ").split()))
print("List of values: ", input_data)
#changing the input_data to numpy aray(input_data)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('the person  is not diabetic')
else:
    print('the person is diabetic')