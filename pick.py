from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('/content/drive/MyDrive/IceCreamData.csv')
df.head()
x = df['Temperature'].values
y = df['Revenue'].values
x.shape, y.shape


X_train, y_train = train_test_split(x, y, test_size=50)
model = LinearRegression()
X_train = X_train.reshape(-1,1)
model.fit(X_train, y_train)

filename = 'model.pickle'
pickle.dump(model, open(filename, "wb"))
