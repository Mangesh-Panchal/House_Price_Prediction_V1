import streamlit as st
st.title("House Price Prediction")

#st.selectbox("Select Area",("1","2"))

#st.text_input("Select Number of Bedrooms")


from sklearn import linear_model
import numpy as np
import pandas as pd
df=pd.read_csv("HOUSE_PRICE_PREDCITION.csv")
model=linear_model.LinearRegression()
model.fit(df[["area"]],df.price)



x=st.text_input("Enter Area ( Square ft )")
buttonpressed=st.button("Predict House Price")

if(buttonpressed):
    a = np.array(x)
    a = a.reshape(1, -1)
    pred=model.predict(a)
    st.write(pred[0])