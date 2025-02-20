import streamlit as st
import pandas as pd
from models import CustomModel


st.set_page_config(page_title="Blackbox AI", layout="wide")

st.title("Blackbox AI")
st.markdown("### A No-Code Machine Learning Interface")

# Step 1: Upload CSV file
st.sidebar.header("Step 1: Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head(10))

    # Step 2: Select features and target
    st.sidebar.header("Step 2: Select Features and Target")
    features = st.sidebar.multiselect("Select Features", df.columns)
    target = st.sidebar.selectbox("Select Target", set(df.columns) - set(features))

    # Step 3: Select classification or regression
    st.sidebar.header("Step 3: Select Task Type")
    task_type = st.sidebar.radio("Task Type", ("Regression", "Classification"))

    # Step 4: Select preprocessing steps
    st.sidebar.header("Step 4: Select Preprocessing Steps")
    scaler = st.sidebar.selectbox("Select Scaler", ["StandardScaler", "MinMaxScaler", "None"])
    encoder = st.sidebar.selectbox("Select Encoder", ["OneHotEncoder", "OrdinalEncoder", "None"])
    nullHandlerNumeric = st.sidebar.selectbox("Select Null Handler for Numeric Data", ["KNNImputer", "Mean", "Median", "MostFrequent", "None"])
    nullHandlerCategorical = st.sidebar.selectbox("Select Null Handler for Categorical Data", ["KNNImputer", "MostFrequent", "None"])
    custom_model = CustomModel(df, task_type, features, target, scaler, encoder, nullHandlerNumeric, nullHandlerCategorical)
    if st.sidebar.button("Preprocess Data"):
        st.session_state.df_preprocess = custom_model.preprocessData()

    if "df_preprocess" in st.session_state:
        st.write("### Preprocessed Data")
        st.dataframe(st.session_state.df_preprocess.head())

    # Step 5: Select model type
    st.sidebar.header("Step 5: Select Model Type")
    model_type = st.sidebar.selectbox("Model Type", ("Linear", "Logistic", "RandomForest", "SVM", "DecisionTree", "KNN", "NeuralNetwork"))
    custom_model.modelType = model_type

    # Step 6: Select metrics
    st.sidebar.header("Step 6: Select Metrics")
    metrics = st.sidebar.multiselect("Select Metrics", ["Accuracy", "Precision", "Recall", "F1 Score", "MSE", "RMSE", "MAE", "R2 Score"])

    # Step 7: Train model
    if st.sidebar.button("Train Model"):
        st.write("### Training Model...")
        # Create and train the model using CustomModel class
        best_params = custom_model.trainModel()

        st.write("### Model trained successfully!")
        st.write("**Best Parameters:**")
        st.write(pd.DataFrame([best_params]))

        # Evaluate the model
        metrics_dict = custom_model.evaluateModel(metrics)
        st.write("### Model Evaluation")
        st.write(metrics_dict)