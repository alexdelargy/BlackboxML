import streamlit as st 
import pandas as pd 
from models import CustomModel

st.title("Blackbox AI")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview")
    st.write(df.head())

    st.write("Select Features and Targets")
    X = st.multiselect("Select features for X", df.columns)
    y = st.selectbox("Select target for y", set(df.columns)-set(X))

    st.write("Select Task Type")
    task_type = st.radio("Select Task Type", ['Regression', 'Classification'])

    st.write("Select Preprocessing Steps")
    scaler = st.selectbox("Select Scaler", ["StandardScaler", "MinMaxScaler", "None"])
    encoder = st.selectbox("Select Encoder", ["OneHotEncoder", "OrdinalEncoder", "None"])
    nullHandlerNumeric = st.selectbox("Select Null Handler for Numeric Data", ["KNNImputer", "Mean", "Median", "MostFrequent", "None"])
    nullHandlerCategorical = st.selectbox("Select Null Handler for Categorical Data", ["KNNImputer", "MostFrequent", "None"])

    st.write("Select Model Type")
    model_type = st.selectbox("Model Type", ("Linear", "Logistic", "RandomForest", "SVM", "DecisionTree", "KNN", "NeuralNetwork"))

    st.write("Select Metrics")
    metrics = st.multiselect("Select Metrics", ["Accuracy", "Precision", "Recall", "F1 Score", "MSE", "RMSE", "MAE", "R2 Score"])

    if st.button("Train Model"):
        # Create and train the model using CustomModel class
        custom_model = CustomModel(df, task_type, model_type, X, y, scaler, encoder, nullHandlerNumeric, nullHandlerCategorical)
        best_params = custom_model.trainModel()

        st.write("Model trained successfully!")
        st.write(f"Best Parameters: {best_params}")

        # Evaluate the model
        metrics_dict = custom_model.evaluateModel(metrics)
        st.write("Model Evaluation")
        st.write(metrics_dict)