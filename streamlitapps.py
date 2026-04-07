import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

st.title("Web app using Streamlit")

st.image("streamliticon.png", width=300)

st.title("Case study on Housing Dataset")

# Load data
data = pd.read_csv("housing.csv", sep=";", engine="python")


data.columns = data.columns.str.strip().str.lower()
data = data.replace('"', '', regex=True)



data.columns = ["area","bedrooms","bathrooms","stories","mainroad",
                "guestroom","basement","hotwaterheating","airconditioning",
                "parking","prefarea","price"]


data = data.apply(pd.to_numeric,errors='ignore')


st.write("Columns:", data.columns.tolist())
st.write(data.head())

st.write("Shape of dataset:", data.shape)

menu = st.sidebar.radio("Menu", ["Home", "Prediction price"])

if menu == "Home":
    st.image("Housepricepredictionimage.png")

    st.header("Tabular Data of Housing")
    if st.checkbox("Tabular Data"):
        st.table(data.head(600))

    st.header("Statistical summary of a Dataframe")
    if st.checkbox("Statistics"):
        st.table(data.describe())

    # FIXED HEADER
    st.header("Correlation graph")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.title("Graphs")

    graph = st.selectbox("Different types of graphs",
                         ["Scatter Plot", "Bar Graph", "Histogram"])

    if graph == "Scatter Plot":
        value=st.slider("Filter data using area", 0, int(data["area"].max()))

        filtered_data = data.loc[data["area"] >= value]
        fig, ax = plt.subplots(figsize=(20,12))
        sns.scatterplot(data=data, x="area", y="price", ax=ax)
        st.pyplot(fig)
    if graph=="Bar Graph": 
        fig,ax=plt.subplots(figsize=(24,18))
        sns.barplot(data=data, x="bedrooms", y="price", ax=ax)
        st.pyplot(fig)

    if graph=="Histogram":
        fig,ax=plt.subplots(figsize=(30,24))
        sns.histplot(data['price'], kde=True, ax=ax)
        st.pyplot(fig)  

if menu=="Prediction price":
    st.title("Prediction price of a house")

    from sklearn.linear_model import LinearRegression
    lr=LinearRegression()
    X=np.array(data["area"]).reshape(-1,1)
    y=np.array(data["price"]).reshape(-1,1)
    lr.fit(X,y)               
    value=st.number_input("Area",1.30,5.01,step=0.20)
    value=np.array([[value]])    # ✅ correct shape (1,1)
    prediction=lr.predict(value)[0]
    if st.button("Price Prediction(INR)"):
        st.write(f"{prediction}")