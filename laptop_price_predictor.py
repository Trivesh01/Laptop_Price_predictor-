#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np 


# In[114]:


df=pd.read_csv('laptop_data.csv')


# In[115]:


df.head()


# In[116]:


df.info()


# In[117]:


df.duplicated().sum()


# In[118]:


#removing the unnamed column
df=df.drop('Unnamed: 0',axis=1)


# In[119]:


df['Ram']=df['Ram'].str.replace('GB','')


# In[120]:


df['Weight']=df['Weight'].str.replace('kg','')


# In[121]:


df['Weight']=df['Weight'].astype(float)
df['Ram']=df['Ram'].astype(float)


# In[122]:


df.info()


# In[123]:


import seaborn as sns


# In[124]:


sns.distplot(df['Price'])


# In[125]:


df['Company'].value_counts().plot(kind='bar')


# In[126]:


import matplotlib.pyplot as plt
#average price of each of the brand 
sns.barplot(x='Company',y='Price',data=df)
plt.xticks(rotation='vertical')
plt.show()


# In[127]:


df['TypeName'].value_counts().plot(kind='bar')


# In[128]:


sns.distplot(df['Inches'])


# In[129]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# In[130]:


df['ScreenResolution'].value_counts()


# In[131]:


df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[132]:


df


# In[133]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[134]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[135]:


df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[136]:


df


# In[137]:


sns.distplot(df['Ips'])


# In[138]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[139]:


new=df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[140]:


df['X_res']=new[0]
df['Y_res']=new[1]


# In[141]:


df


# In[142]:


df['X_res'].str.split(' ')[0][-1]


# In[143]:


len(df['X_res'])


# In[144]:


def retrieve_x(text):
    val=text.split(' ')
    return int(val[-1])
    


# In[145]:


retrieve_x('IPS Panel Retina Display 2560')


# In[146]:


df['X_res']=df['X_res'].apply(retrieve_x)


# In[147]:


df


# In[148]:


df['X_res']=df['X_res'].astype(int)
df['Y_res']=df['Y_res'].astype(int)


# In[149]:


df.info()


# In[150]:


df['PPI']=(((df['X_res']**2)+(df['Y_res']**2))**(1/2))/(df['Inches'])


# In[151]:


df


# In[152]:


df.info()


# In[153]:


df=df.drop(columns=['X_res','Y_res','Inches'],axis=1)


# In[154]:


df=df.drop(columns=['ScreenResolution'],axis=1)


# In[155]:


df.info()


# In[156]:


df['Cpu'].value_counts()


# In[157]:


df['CPU Name']=df['Cpu'].apply(lambda x:' '.join(x.split()[0:3]) )


# In[158]:


df['CPU Name']


# In[159]:


def fetch_processor(text):
    if text=='Intel Core i7' or text=='Intel Core i5' or text=='Intel Core i3':
        return text
    else:
        if text.split()[0]=='Intel':
            return "Other Intel Processor"
        else:
            return "AMD  Processor"
        


# In[160]:


df['CPU brand']=df['CPU Name'].apply(fetch_processor)


# In[161]:


df['CPU brand'].value_counts().plot(kind='bar')


# In[162]:


sns.barplot(x=df['CPU brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[163]:


df.head()


# In[164]:


df=df.drop(['Cpu','CPU Name'],axis=1)


# In[165]:


df.head()


# In[166]:


sns.barplot(x=df['Ram'],y=df['Price'])


# In[167]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[168]:


df.sample(5)


# In[169]:


df=df.drop('Memory',axis=1)


# In[170]:


df.head()


# In[171]:


df.corr()['Price']


# In[172]:


df=df.drop(['Hybrid','Flash_Storage'],axis=1)


# In[173]:


df.head()


# In[174]:


df['Gpu'].value_counts()


# In[175]:


df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[176]:


df['Gpu brand']


# In[177]:


df.head()


# In[178]:


df['Gpu brand'].value_counts()


# In[179]:


df=df[df['Gpu brand']!='ARM']


# In[180]:


df.head()


# In[181]:


sns.barplot(x=df["Gpu brand"],y=df['Price'],estimator =np.median)


# In[182]:


df=df.drop('Gpu',axis=1)


# In[183]:


df.head()


# In[184]:


df['OpSys'].value_counts()


# In[185]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[186]:


df['os'] = df['OpSys'].apply(cat_os)


# In[187]:


df.head()


# In[188]:


df.drop('OpSys',axis=1,inplace=True)


# In[189]:


df.head()


# In[190]:


sns.heatmap(df.corr(),annot=True)


# In[191]:


X=df.drop('Price',axis=1)
y=np.log(df['Price'])


# In[192]:


X


# In[193]:


y


# In[194]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[195]:


X_train.head()


# In[196]:


X_train.shape


# In[197]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[198]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# In[199]:


pip install xgboost


# # Linear Regression 

# In[200]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Ridge Regression 

# In[201]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Lasso Regression

# In[202]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # KNN

# In[203]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Decision Tree

# In[204]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # SVM

# In[205]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Random Forest

# In[206]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Adaboost

# In[207]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Gradient  Boost

# In[208]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # XGBoost

# In[209]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Taking the Random Forest model and performing HyperPerameter Tuning

# In[231]:


def Random_forest(n_estimators):
    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
    ],remainder='passthrough')
    step2 = RandomForestRegressor(n_estimators=n_estimators,
                                  random_state=3,
                                  max_samples=0.7,
                                  max_features=0.75,
                                  max_depth=20)
    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)

    return r2_score(y_test,y_pred)


# In[232]:


#chewcking th performace by  changing the value of n_estimators(Hyper-parameter tuning)
x=[i for i in range(100,201)]
y=[Random_forest(num) for num in x]


# In[233]:


plt.plot(x,y)


# In[234]:


Random_forest(177)


# In[235]:


def Random_forest(num):
    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
    ],remainder='passthrough')
    step2 = RandomForestRegressor(n_estimators=177,
                                  random_state=3,
                                  max_samples=num,
                                  max_features=0.75,
                                  max_depth=20)
    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)

    return r2_score(y_test,y_pred)


# In[242]:


x=np.arange(0.1,1,0.1)
y=[Random_forest(num) for num in x]


# In[243]:


plt.plot(x,y)


# In[244]:


def Random_forest(num):
    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
    ],remainder='passthrough')
    step2 = RandomForestRegressor(n_estimators=177,
                                  random_state=3,
                                  max_samples=0.9,
                                  max_features=num,
                                  max_depth=20)
    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)

    return r2_score(y_test,y_pred)


# In[245]:


x=np.arange(0.1,1,0.1)
y=[Random_forest(num) for num in x]


# In[246]:


plt.plot(x,y)


# In[247]:


def Random_forest(num):
    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
    ],remainder='passthrough')
    step2 = RandomForestRegressor(n_estimators=177,
                                  random_state=3,
                                  max_samples=0.9,
                                  max_features=0.9,
                                  max_depth=num)
    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)

    return r2_score(y_test,y_pred)


# In[248]:


x=np.arange(20,50,1)
y=[Random_forest(num) for num in x]


# In[249]:


plt.plot(x,y)


# In[252]:


def Random_forest():
    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
    ],remainder='passthrough')
    step2 = RandomForestRegressor(n_estimators=177,
                                  random_state=3,
                                  max_samples=0.9,
                                  max_features=0.9,
                                  max_depth=24)
    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)

    return r2_score(y_test,y_pred)


# In[253]:


print(Random_forest())


# In[256]:


print('The final upgraded R2 score from HyperPerimeter Tuning is:',Random_forest())


# In[ ]:




