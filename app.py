import numpy as np
import streamlit as st
import pickle

model=pickle.load(open('model.pickle','rb'))

st.title("Revenue Prediction")
X_new=st.number_input('Input Temperature')
if st.button('Predict'):
  X_new=np.array(X_new).reshape(-1,1)
  y_new=model.predict(X_new)
  st.caption('Revenue Prediction')
  st.write(y_new)
