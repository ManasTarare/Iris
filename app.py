import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Ensure target is of integer type
df['target'] = df['target'].astype(int)

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Streamlit app
st.title('Iris Flower Prediction')
st.sidebar.header('Model Evaluation')

# Sidebar options
if st.sidebar.checkbox('Show Accuracy Score'):
    st.sidebar.write(f"Accuracy: {accuracy:.2f}")

if st.sidebar.checkbox('Show Confusion Matrix'):
    st.sidebar.subheader('Confusion Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.sidebar.pyplot(fig)

if st.sidebar.checkbox('Show Mean Squared Error'):
    st.sidebar.write(f"Mean Squared Error: {mse:.2f}")

# Get user input
st.header('Input Features')
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=5.0, value=3.5)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=5.0, value=0.2)

# Predict using the model
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(user_input)

# Display the prediction
flower_types = ['Setosa', 'Versicolor', 'Virginica']
predicted_class = flower_types[prediction[0]]
st.write(f"Predicted class: **{predicted_class}**")

# Feature importances
st.header('Feature Importances')
importances = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(data.feature_names, importances)
ax.set_xlabel("Feature Importance")
ax.set_title("Feature Importance in Decision Tree Model")
st.pyplot(fig)

# Experiment with Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
st.write(f"Model Accuracy: {lr_accuracy:.2f}")

# Sidebar Linear Regression Experiment
st.sidebar.header('Linear Regression Experiment')
if st.sidebar.checkbox('Show Linear Regression Experiment'):
    # Predicting petal length using other features as an example
    X_lin = df.drop('petal length (cm)', axis=1)
    y_lin = df['petal length (cm)']

    X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

    lin_model = LinearRegression()
    lin_model.fit(X_lin_train, y_lin_train)
    lin_pred = lin_model.predict(X_lin_test)
    lin_mse = mean_squared_error(y_lin_test, lin_pred)
    st.sidebar.write(f"Linear Regression Mean Squared Error: {lin_mse:.2f}")

    # Plot actual vs predicted petal length
    fig, ax = plt.subplots()
    ax.scatter(y_lin_test, lin_pred)
    ax.set_xlabel("Actual Petal Length")
    ax.set_ylabel("Predicted Petal Length")
    ax.set_title("Actual vs Predicted Petal Length")
    st.sidebar.pyplot(fig)
