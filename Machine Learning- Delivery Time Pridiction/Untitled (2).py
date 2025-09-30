#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("orders.csv")

# Encode categorical
le_city = LabelEncoder()
le_mode = LabelEncoder()
le_load = LabelEncoder()

df['City'] = le_city.fit_transform(df['City'])
df['Delivery_Mode'] = le_mode.fit_transform(df['Delivery_Mode'])
df['Warehouse_Load'] = le_load.fit_transform(df['Warehouse_Load'])

# Features and target
X = df.drop(['OrderID', 'Delivery_Days'], axis=1)
y = df['Delivery_Days']


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


# In[12]:


import numpy as np
# ================================
# 🚀 Function to predict delivery days
# ================================
def predict_delivery_days(city, distance, delivery_mode, order_value, warehouse_load):
    """
    Predict delivery days for given order details.
    """
    city_enc = le_city.transform([city])[0]
    mode_enc = le_mode.transform([delivery_mode])[0]
    load_enc = le_load.transform([warehouse_load])[0]
    
    input_data = np.array([[city_enc, distance, mode_enc, order_value, load_enc]])
    predicted_days = model.predict(input_data)[0]
    return predicted_days

# ================================
# 🎯 Example usage
# ================================
days = predict_delivery_days(
    city='Mumbai',
    distance=3000,
    delivery_mode='Standard',
    order_value=5500,
    warehouse_load='Medium'
)
print(f"Estimated delivery time: {days:.1f} days")


# In[ ]:





# In[ ]:




