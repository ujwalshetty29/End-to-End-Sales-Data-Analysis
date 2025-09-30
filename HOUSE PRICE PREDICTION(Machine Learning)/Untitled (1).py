#!/usr/bin/env python
# coding: utf-8

# In[204]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_csv("Bengaluru_House_Data.csv")
df1.head()
# In[205]:


df2=df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df2.head()import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[206]:


df2.isnull().sum()


# In[207]:


df3=df2.dropna()
df3.isnull().sum()


# In[208]:


df3.total_sqft.unique()


# In[209]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[210]:


df3['bhk'].unique()


# In[211]:


df3[df3.bhk>20]


# In[212]:


def is_float(x):
 try:
   float(x)
 except:
   return False
 return True


# In[213]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[214]:


def convert_sqft_to_num(x) :
    tokens = x. split(' -')
    if len(tokens) == 2:
       return (float(tokens[0])+float(tokens[1]))/2
    try:
       return float(x)
    except :
       return None


# In[215]:


convert_sqft_to_num('2166' )


# In[216]:


convert_sqft_to_num('1015 - 1540' )


# In[217]:


df4 = df3.copy()
df4['total_sqft' ] = df4[ 'total_sqft' ] . apply(convert_sqft_to_num)
df4.head(3)


# In[218]:


df4.loc[30]


# In[219]:


df5 = df4. copy()
df5 [ 'price_per_sqft' ] = df5[ 'price' ]*100000/df5 [ 'total_sqft' ]
df5. head()


# In[220]:


df5. location = df5. location. apply(lambda x: x.strip())
location_stats = df5. groupby('location' ) ['location' ] . agg('count')
location_stats


# In[221]:


df5 = df4. copy()
df5 [ 'price_per_sqft' ] = df5[ 'price' ]*100000/df5 [ 'total_sqft' ]
df5. head()


# In[222]:


df5 [ df5. total_sqft/df5. bhk<300] . head()


# In[223]:


df5.shape


# In[224]:


df6 = df5[~(df5. total_sqft/df5.bhk<300)]
df6. shape


# In[225]:


def remove_pps_outliers (df) :
   df_out = pd. DataFrame()
   for key, subdf in df. groupby('location' ) :
      m = np.mean(subdf.price_per_sqft)
      st = np.std(subdf.price_per_sqft)
      reduced_df = subdf [ (subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft <= (m+st))]
      df_out = pd. concat([df_out, reduced_df], ignore_index=True)
   return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[226]:


df = pd.read_csv('Bengaluru_House_Data.csv')  

df = df.dropna(subset=['location', 'size', 'total_sqft', 'price'])


df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))


def convert_sqft_to_num(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        elif x.strip().isalpha():
            return None
        else:
            return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

df = df.dropna(subset=['total_sqft'])


df = df[df['total_sqft'] / df['bhk'] >= 300]


df = df.reset_index(drop=True)

print("Cleaned Data Sample:")
print(df.head())

print("\nCleaned Dataset Shape:", df.shape)


# In[227]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df_fe = df.copy()

df_fe['location'] = df_fe['location'].astype(str).apply(lambda x: x.strip())

location_stats = df_fe['location'].value_counts()
rare_locations = location_stats[location_stats <= 10].index
df_fe['location'] = df_fe['location'].apply(lambda x: 'other' if x in rare_locations else x)


df_fe = pd.get_dummies(df_fe, columns=['location'], drop_first=True)

numerical_features = ['total_sqft', 'bath', 'balcony', 'price']
scaler = MinMaxScaler()
df_fe[numerical_features] = scaler.fit_transform(df_fe[numerical_features])


print("Feature Engineering Complete")
print("Shape of engineered data:", df_fe.shape)
print("Columns sample:", df_fe.columns[:10].tolist())  # Show only a few columns
df_fe.head()


# In[228]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_fe = df.copy()



df_fe['location'] = df_fe['location'].astype(str).apply(lambda x: x.strip())


location_stats = df_fe['location'].value_counts()
rare_locations = location_stats[location_stats <= 10].index
df_fe['location'] = df_fe['location'].apply(lambda x: 'other' if x in rare_locations else x)


df_fe = pd.get_dummies(df_fe, columns=['location'], drop_first=True)


numerical_features = ['total_sqft', 'bath', 'balcony', 'price']
scaler = MinMaxScaler()
df_fe[numerical_features] = scaler.fit_transform(df_fe[numerical_features])



print("Feature Engineering Complete")
print("Shape of engineered data:", df_fe.shape)
print("Columns sample:", df_fe.columns[:10].tolist())  # Show only a few columns
df_fe.head()


# In[229]:


rare_locations = location_stats[location_stats <= 10].index
len(rare_locations)


# In[230]:


len(location_stats)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


X = pd.get_dummies(df_fe.drop('price', axis=1), drop_first=True)
y = df_fe['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler()),                  
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

lr_r2 = r2_score(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))


rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

rf_r2 = r2_score(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))


print("Linear Regression Results:")
print("R² Score:", round(lr_r2, 4))
print("RMSE:", round(lr_rmse, 4))

print("\nRandom Forest Results:")
print("R² Score:", round(rf_r2, 4))
print("RMSE:", round(rf_rmse, 4))


# In[ ]:


def predict_house_price(sqft, bath, balcony, bhk, location_name):
    
    input_data = pd.DataFrame([np.zeros(len(X.columns))], columns=X.columns)

    
    scaled_vals = scaler.transform([[sqft, bath, balcony, 0]])[0]
    input_data['total_sqft'] = scaled_vals[0]
    input_data['bath'] = scaled_vals[1]
    input_data['balcony'] = scaled_vals[2]
    input_data['bhk'] = bhk

    # Step 3: Set location column (One-hot encoded)
    location_col = f"location_{location_name}"
    if location_col in input_data.columns:
        input_data[location_col] = 1
    else:
        print(f"Location '{location_name}' not found. Using 'other' category if available.")

    
    scaled_price = rf.predict(input_data)[0]

   
    original_price = scaler.inverse_transform([[0, 0, 0, scaled_price]])[0][3]
    price_in_inr = round(original_price * 1_00_000, 2)


    print(f"Predicted Price for {bhk}BHK, {sqft} sqft in {location_name}: ₹{price_in_inr} INR")
    return price_in_inr


# In[ ]:


predict_house_price(1500, 2, 2, 3, "Whitefield")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




