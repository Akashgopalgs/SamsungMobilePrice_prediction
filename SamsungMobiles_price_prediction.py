#!/usr/bin/env python
# coding: utf-8

# # Samsung Mobiles Analysis

# In[71]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[72]:


df=pd.read_json(r"C:\Users\akash\OneDrive\Documents\Web_Screping\IMDb\Project\samsung_Mobiles.json")
df


# In[73]:


df['Price']=df['Price'].str.replace('₹','')
df['Price']=df['Price'].str.replace(',','')


# In[74]:


df.dtypes


# In[75]:


df['Price']=df['Price'].astype(float)



# In[76]:


df['Battery'] = df['Battery'].str.replace(r'(\s\w+)$', '', regex=True)
df['Battery'] = df['Battery'].str.replace(' mAh', '')
df['Battery'] = df['Battery'].str.replace(' Li-ion', '')
df['Battery'] = df['Battery'].str.replace(' Lithium-ion', '')
df['Battery'] = df['Battery'].str.replace(' Lithium', '')
df['Battery'] = df['Battery'].str.replace(' ion', '')
df['Battery'] = df['Battery'].str.replace(' Ion', '')
df['Battery'] = df['Battery'].str.replace(' Battery', '')
df


# In[77]:


df['Battery']=df['Battery'].astype(float)


# In[78]:


df.info()


# In[79]:


df['Price'].max()


# In[80]:


df['Price'].min()


# In[81]:


df['Price'].mean()


# In[82]:


df['Price'].mode()


# **Price**
# * Average price of Samsung mobile is 36789.24427480916 
# * Most Expensive samsung mobile's Price is 164999.0
# * Lowest price of mobile samsung mobile is 11348.0

# In[164]:


sns.displot(df['Price'],bins=15)
plt.title('Price distrubution of Samsung Mobies')


# * There is several mobiles are in the range of 20000 
# * most common phones are in range of 20000 to 40000
# * expensive phone's price is nearest to 160000

# In[165]:


sns.lineplot(x=df['Rating'],y=df['Price'])


# * Rating is directly influance the price of the phone
# * rating is increses with the price increses but slightly dicreses when the rating is maximum
# * rating of expensive phone's rating is 4.5

# most of phone are in range of 15999.0

# In[166]:


df.describe()


# In[15]:


df['Rating'].mean()


# In[16]:


df['Rating'].mode()


# In[17]:


df['Rating'].max()


# 
# **Rating**
# * Maximum rating of mobile is 5
# * Average and most common rating of sumsung phone is  4.2

# In[18]:


df['Battery'].max()


# In[19]:


df['Battery'].min()


# In[20]:


df['Battery'].mode()


# **Battery**
# * Maximum battery capacity is 6000 mAh
# * minimum battery capacity is 3300 mAh
# * 5000 mAh is the most commonn battery capacity if samsung mobiles

# In[173]:


sns.lineplot(x=df['Battery'],y=df['Price'])


# 
# * Expensive phone's battery is in the range of 4500 mAh
# * maximum battery capacity are in low rated phone,and which is are inthe range of 20000
# * most models are in 5000mAh

# In[174]:


sns.histplot(df['Battery'],bins=15)


# * Mejority od phones are in 5000mAh battery capacity
# * few phones are in below 3500 mAh

# In[21]:


camera_counts = df['Camera'].value_counts()
camera_counts


# In[176]:


plt.figure(figsize=(10, 10))
plt.pie(camera_counts,labels=camera_counts.index, autopct='%1.1f%%')
plt.show()


# 
# * Most models of phone have 50MP rear camera(17.3%)
# * Second largest number of phone have 50MP + 2MP | 13MP Front Camera  
# * Top camrea have 200MP + 10MP + 12MP + 10MP | 12MP Front Camera

# In[22]:


df1=df.select_dtypes([int,float])
df1


# In[23]:


df1.loc[:100].corr()


# * Battary and Price are in neagative strong correlation
# * There is no correlation b/w rating and price 

# ### Conclusion about data

# - **Average and Range of Prices**: The average price of Samsung mobiles is around ₹36,789, with prices ranging from ₹11,348 to ₹164,999.
# 
# - **Price Distribution**: Most Samsung phones are priced between ₹20,000 to ₹40,000, with a few outliers that are significantly more expensive.
# 
# - **Price and Rating Correlation**: There is a positive correlation between price and rating, with higher-priced phones generally receiving better ratings, although ratings slightly decrease for the most expensive models.
# 
# - **Battery Capacity**: The most common battery capacity among Samsung phones is 5000 mAh, with capacities ranging from 3300 mAh to 6000 mAh.
# 
# - **Battery Capacity and Price Correlation**: A strong negative correlation exists between battery capacity and price, while the correlation between rating and price is not significant.

# In[83]:


df['Mob Name'].unique()


# In[84]:


df[['model', 'colour', 'storage']] = df['Mob Name'].str.extract(r'^(.*?) \((.*?), (.*?)\)$')


# In[85]:


df


# In[86]:


df.info()


# In[87]:


df['Camera'].unique()


# In[88]:


df[['back camera', 'front camera']] = df['Camera'].str.split(' \| ', expand=True)


# In[89]:


df


# In[90]:


df.info()


# In[91]:


df['Ram&Rom'].unique()


# In[92]:


def extract_ram(ram_rom):
    if "RAM" in ram_rom:
        return ram_rom.split('|')[0]  # Extract text till " RAM"
    return None  # Return None if "RAM" is not found

# Apply the function to the DataFrame
df['ram'] = df['Ram&Rom'].apply(extract_ram)


# In[93]:


df


# In[94]:


df['ram'].isnull().sum()


# In[95]:


df['ram']=df['ram'].fillna(df['ram'].mode()[0])


# In[96]:


df


# In[97]:


df['Display'].unique()


# In[98]:


df['storage']=df['storage'].str.replace(" GB","").astype(int)


# In[99]:


df['ram']=df['ram'].str.replace(" GB RAM","").astype(int)


# In[100]:


df


# In[101]:


df.drop(columns=['Mob Name','Camera','Ram&Rom'],inplace=True)


# In[102]:


df


# In[103]:


df['back camera'].unique()


# ### Data preprocessing

# In[104]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler 


# In[105]:


cols_to_encode = ['model', 'Display', 'colour', 'back camera', 'front camera']

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(df[cols_to_encode])


# In[106]:


encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cols_to_encode))


# In[107]:


df = pd.concat([df, encoded_df], axis=1)


# In[108]:


df.head()


# In[109]:


df = df.drop(columns=cols_to_encode).reset_index(drop=True)


# In[110]:


df.head()


# In[111]:


cols_to_scale = ['Price', 'Rating', 'Battery', 'storage', 'ram']


# In[112]:


scaler = StandardScaler()


# In[115]:


df[['Rating', 'Battery', 'storage', 'ram']] = scaler.fit_transform(df[['Rating', 'Battery', 'storage', 'ram']])


# In[116]:


df.head()


# In[70]:


from sklearn.model_selection import train_test_split


# In[117]:


X = df.drop(columns='Price')
y = df['Price']


# In[119]:


X.head()


# In[120]:


y.head()


# In[121]:


X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.15,random_state=24)


# In[123]:


X_train.head()


# In[124]:


y_train.head()


# In[125]:


X_test.head()


# In[126]:


y_test.head()


# ### Model training & Evaluation

# In[142]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# In[156]:


models=[DecisionTreeRegressor(),KNeighborsRegressor(),RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor()]


# In[159]:


def model_check():
    for model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        acc= r2_score(y_test,y_pred)
        accuracy=round(acc,2)*100
        print(model,":",accuracy,"%")
        print("-----------------------------")
model_check()


# In[152]:


from sklearn.model_selection import cross_val_score


# In[165]:


def model_check(k=5):
    for model in models:
        cv_score = cross_val_score(model, X,y ,cv=k, scoring='r2')
        accuracy = sum(cv_score)/5
        a_accuracy=accuracy*100
        mean_accuracy=round(accuracy,2)*100
        print("Cross validation Accuracies for ",model,"=",cv_score)
        print("Accuracy of ",model,"=",mean_accuracy,"%")
        print("--------------------------------------------------------")
        
model_check(k=5)


# ### Decision Tree has higher accuracy with 92%

# In[ ]:


# Import necessary libraries
import joblib

# Train the model (assuming DecisionTreeRegressor was the best)
best_model = DecisionTreeRegressor()
best_model.fit(X, y)  # Train on the entire dataset for deployment

# Save the trained model to a file
joblib.dump(best_model, 'best_model.pkl')

# Save the encoder and scaler to a file
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

