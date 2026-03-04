import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")

st.title("🌸 Dự đoán loại hoa Iris")

sepal_length = st.number_input("Sepal Length", 0.0, 10.0)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0)
petal_length = st.number_input("Petal Length", 0.0, 10.0)
petal_width = st.number_input("Petal Width", 0.0, 10.0)

if st.button("Dự đoán"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    iris_types = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Kết quả: {iris_types[prediction[0]]}")