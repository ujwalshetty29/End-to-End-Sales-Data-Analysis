#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install mysql-connector-python')
import mysql.connector
import pandas as pd


cnx = mysql.connector.connect(user='root', password='Ujwalrs@123',
                              host='localhost')


# In[ ]:


query="SELECT * FROM project.orde"


# In[ ]:


df=pd.read_sql(query,cnx)


# In[ ]:


cnx.close()


# In[ ]:


print(df)


# In[ ]:


df.head(5)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


df.head(5)


# In[ ]:


df.head(5)


# In[ ]:


df.columns=df.columns.str.lower()
df.columns=df.columns.str.replace(' ','_')
df.columns


# df.head(5)

# In[ ]:


df.head(5)


# In[ ]:


df.head(7)


# In[ ]:


total_sales = df['sales_price'].sum()
total_orders = df['order_id'].nunique()
total_profit = df['profit'].sum()
average_order_value = total_sales / total_orders if total_orders else 0

print(f"Total Sales: {total_sales}")
print(f"Total Orders: {total_orders}")
print(f"Total Profit: {total_profit}")
print(f"Average Order Value: {average_order_value}")


# In[ ]:


df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

df['order_date'] = pd.to_datetime(df['order_date'], format='%d-%m-%Y', errors='coerce')


sales_trend = df.groupby(df['order_date'].dt.date)['sales_price'].sum()


import matplotlib.pyplot as plt
plt.figure(figsize=(100, 6))
plt.bar(sales_trend.index, sales_trend.values, color='skyblue')


plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


print(sales_trend.head())    


# In[ ]:


sales_trend = df.groupby(df['order_date'].dt.date)['sales_price'].sum()


# In[ ]:


print(sales_trend)


# In[ ]:


top_selling = df.groupby('product_id').agg({
    'quantity': 'sum',
    'sales_price': 'sum'
}).sort_values(by='quantity', ascending=False).head(10)

print(top_selling)


# In[ ]:


most_profitable = df.groupby('product_id')['profit'].sum().sort_values(ascending=False).head(10)
print(most_profitable)


# In[ ]:


# Calculate average discount and total profit per product
product_analysis = df.groupby('product_id').agg({
    'discount_percent': 'mean',
    'profit': 'sum'
})

# Filter products with avg discount > 20% and low profit (<1000)
high_discount_low_profit = product_analysis[
    (product_analysis['discount_percent'] > 4.5) & (product_analysis['profit'] < 1000)
].sort_values(by='discount_percent', ascending=False)

print(high_discount_low_profit)


# In[ ]:


top_segment = df.groupby('segment').agg({
    'sales_price': 'sum',
    'profit': 'sum'
}).sort_values(by='sales_price', ascending=False).head(10)

print(top_segment)


# In[ ]:


region_performance = df.groupby('region').agg({
    'sales_price': 'sum',
    'profit': 'sum'
}).sort_values(by='sales_price', ascending=False)

print(region_performance)


# In[ ]:


state_performance = df.groupby('state').agg({
    'sales_price': 'sum',
    'profit': 'sum'
}).sort_values(by='sales_price', ascending=False)

print(state_performance)


# In[ ]:


shipping_sales = df.groupby('ship_mode').agg({
    'sales_price': 'sum',
    'order_date': 'count'
}).rename(columns={'order_date': 'total_orders'}).sort_values(by='sales_price', ascending=False)

print(shipping_sales)



# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

top_segment = df.groupby('segment')['sales_price'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_segment.index, y=top_segment.values, palette='viridis')
plt.title("Top 10 Customers by Sales")
plt.ylabel("Total Sales")
plt.xlabel("Customer Name")
plt.tight_layout()
plt.show()


# In[ ]:


region_perf = df.groupby('region')[['sales_price', 'profit']].sum().sort_values(by='sales_price', ascending=False)

region_perf.plot(kind='bar', figsize=(10,6), color=['skyblue', 'lightgreen'])
plt.title("Region-wise Sales and Profit")
plt.xlabel("Region")
plt.ylabel("Amount")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[ ]:


ship_sales = df.groupby('ship_mode')['sales_price'].sum()

plt.figure(figsize=(6,6))
plt.pie(ship_sales, labels=ship_sales.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Sales Distribution by Shipping Mode")
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[ ]:


df['profit_margin'] = (df['profit'] / df['sales_price']) * 100


with_discount = df[df['discount_percent'] > 0]
without_discount = df[df['discount_percent'] == 0]


avg_profit_with_discount = with_discount['profit_margin'].mean()
avg_profit_without_discount = without_discount['profit_margin'].mean()

print(f" Avg Profit Margin WITH Discount: {avg_profit_with_discount:.2f}%")
print(f" Avg Profit Margin WITHOUT Discount: {avg_profit_without_discount:.2f}%")


# In[ ]:


import matplotlib.pyplot as plt

labels = ['With Discount', 'Without Discount']
values = [avg_profit_with_discount, avg_profit_without_discount]

plt.bar(labels, values, color=['tomato', 'mediumseagreen'])
plt.title('Average Profit Margin Comparison')
plt.ylabel('Profit Margin (%)')
plt.grid(axis='y')
plt.show()


# In[ ]:


df_clean = df.dropna(subset=['discount_percent', 'sales_price'])


correlation = df_clean['discount_percent'].corr(df_clean['sales_price'])
print(f"📈 Correlation between Discount % and Sales: {correlation:.2f}")


# In[ ]:


import seaborn as sns

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='discount_percent', y='sales_price', alpha=0.5, color='purple')
plt.title('Discount % vs Sales Amount')
plt.xlabel('Discount Percentage')
plt.ylabel('Sales Amount')
plt.grid(True)
plt.show()


# In[ ]:


bins = [0, 1, 2, 3, 4, 5, 10]
labels = ['0–1%', '1–2%', '2–3%', '3–4%', '4–5%', '5%+']

df['discount_range'] = pd.cut(df['discount_percent'], bins=bins, labels=labels, right=False)


discount_impact = df.groupby('discount_range').agg({
    'sales_price': 'mean',
    'profit': 'mean',
    'profit_margin': 'mean',
    'discount_percent': 'count'
}).rename(columns={'discount_percent': 'count'})

discount_impact.reset_index(inplace=True)
print(discount_impact)


# In[ ]:


plt.figure(figsize=(10, 5))
sns.barplot(data=discount_impact, x='discount_range', y='sales_price', palette='viridis')
plt.title('Average Sales Amount by Discount Range')
plt.xlabel('Discount Range')
plt.ylabel('Avg Sales Price')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.barplot(data=discount_impact, x='discount_range', y='profit_margin', palette='coolwarm')
plt.title('Average Profit Margin by Discount Range')
plt.xlabel('Discount Range')
plt.ylabel('Avg Profit Margin (%)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[ ]:


df['Year'] = df['order_date'].dt.year
df['Month'] = df['order_date'].dt.to_period('M')  # monthly period
df['Week'] = df['order_date'].dt.to_period('W')   # weekly period
df['Day'] = df['order_date'].dt.date


# In[ ]:


# Monthly Sales
monthly_sales = df.groupby('Month')['sales_price'].sum().reset_index()

# Weekly Sales
weekly_sales = df.groupby('Week')['sales_price'].sum().reset_index()

# Yearly Sales
yearly_sales = df.groupby('Year')['sales_price'].sum().reset_index()


# In[ ]:


import matplotlib.pyplot as plt

# Monthly
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Month'].astype(str), monthly_sales['sales_price'], marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


df.head()


# In[ ]:


df.to_csv("sales_data_updated.csv", index=False)


# In[ ]:





# In[ ]:




