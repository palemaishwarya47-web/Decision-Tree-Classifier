import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#load iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

#create dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

#target column
df['species'] = iris.target

#display dataset
st.title("Decision Tree Classifier on Iris Dataset ")
st.write("Dataset Preview:")
st.dataframe(df.head())

#feature and target variable
X = df.drop('species', axis=1)
y = df['species']

#train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

#metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R² Score: {r2}")

#user input section 
st.subheader("Predict Iris Species")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

#predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=iris.feature_names)
prediction = model.predict(input_data)[0]

#convert prediction to nearest class
species_name = iris.target_names[prediction]
st.success(f"Predicted Iris Species: {species_name}")

if prediction == 0:
    st.write("The predicted species is Setosa.")
elif prediction == 1:
    st.write("The predicted species is Versicolor.")
elif prediction == 2:
    st.write("The predicted species is Virginica.")
else:
    st.write("Unable to predict the species.")

#display accuracy score
accuracy = model.score(X_test, y_test)
st.write(f"Accuracy: {accuracy}")