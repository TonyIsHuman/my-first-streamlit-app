import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pickle','rb'))
st.title("Revenue Prediction")
X_new=st.number_input('Input Temperature')
if st.button('Predict'):
  X_new=np.array(X_new).reshape(-1,1)
  y_new=model.predict(X_new)
  st.caption('Revenue Prediction')
  st.success(y_new)
  
