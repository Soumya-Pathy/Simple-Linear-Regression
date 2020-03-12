
# coding: utf-8

# # Simple linear regression

# It is a simple real-estate sample data about price and size of houses in a particular city.
# 
# The data is located in the file: 'real_estate_price_size.csv'. 
# 
# A simple linear regression is created using the data.
# 
# In this exercise, I have taken the dependent variable as 'price', while the independent variables is 'size'. The causal relationship I am looking for is that price is dependent upon the size of the building purchased. Let's checkout the relationship.

# ## Importing the relevant libraries

# In[10]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Loading the data

# In[2]:


data = pd.read_csv('real_estate_price_size.csv')


# In[3]:


data


# ### Descriptive Analytics of the data provided, by using pandas library 

# In[4]:


data.describe()


# ## Creating the regression

# ### Dependent and the independent variables

# In[5]:


x1 = data['size']
y = data['price']


# ### Scaterplot to visualize the data points

# In[6]:


plt.scatter(x1, y)
plt.xlabel('SIZE', fontsize = 20)
plt.ylabel('PRICE', fontsize = 20)
plt.show()


# The graph shows that there is a pattern to the data and price tends to increase with size of the building purchased.

# ### Regression Analysis

# In[7]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# The constant value is 101900 and the coefficient of independent variable is 223.18. So the equation of regression line is:
# ######                         y = 101900 + (223.18*x)

# R-squared value is the variability of the data that is explained by the regression model. 
# The value is considerably high (0.745). So the amount of error (or the amount of variability that is unexplained) is less. This shows that the causal relationship assumed is strong and holds. 

# ### Plot the regression line on the initial scatter

# In[8]:


plt.scatter(x1, y)
yhat = 223.1787*x1 + 101900
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('SIZE', fontsize = 20)
plt.ylabel('PRICE', fontsize = 20)
plt.show()

