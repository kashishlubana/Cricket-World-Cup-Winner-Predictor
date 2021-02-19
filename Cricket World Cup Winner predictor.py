#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## load the CSV(world cup matches from 1983 to 2019)

# In[2]:


matches = pd.read_csv("G:\\CWCData\\Matches\\CWC_Matches_Combined.csv") 


# In[3]:


#Need to rename team U.A.E. to UAE from all the columns 
matches['Team 2'] = matches['Team 2'].str.replace('U.A.E.','UAE')
matches['Team 1'] = matches['Team 1'].str.replace('U.A.E.','UAE')
matches['Winner'] = matches['Winner'].str.replace('U.A.E.','UAE')


# In[4]:


print(matches["Team 2"].unique())


# In[5]:


# For making a more realiable model we need more independent variable, hence adding toss winner columns
matches["Toss Winner"] = ""


# In[6]:


i = 0
j = 1
for i in range(matches.shape[0]):
    
    winner_team = matches.iloc[j,1]
    matches.iloc[j,7] = winner_team 
    print(winner_team)
    j = j + 2
    


# In[7]:


i = 1
j = 0
for i in range(matches.shape[0]):
    
    winner_team = matches.iloc[j,0]
    matches.iloc[j,7] = winner_team 
    print(winner_team)
    j = j + 2
    


# In[8]:


encode = {'Team 1': {'England':1,'Pakistan':2,'Australia':3,'India':4,'New Zealand':5,'West Indies':6,'Sri Lanka':7,'South Africa':8,'Netherlands':9,'Kenya':10,'Bangladesh':11,'Scotland':12,'Zimbabwe':13,'Canada':14,'Namibia':15,'Bermuda':16,'Ireland':17,'Afghanistan':18,'UAE':19},
          'Team 2': {'England':1,'Pakistan':2,'Australia':3,'India':4,'New Zealand':5,'West Indies':6,'Sri Lanka':7,'South Africa':8,'Netherlands':9,'Kenya':10,'Bangladesh':11,'Scotland':12,'Zimbabwe':13,'Canada':14,'Namibia':15,'Bermuda':16,'Ireland':17,'Afghanistan':18,'UAE':19},
          'Toss Winner': {'England':1,'Pakistan':2,'Australia':3,'India':4,'New Zealand':5,'West Indies':6,'Sri Lanka':7,'South Africa':8,'Netherlands':9,'Kenya':10,'Bangladesh':11,'Scotland':12,'Zimbabwe':13,'Canada':14,'Namibia':15,'Bermuda':16,'Ireland':17,'Afghanistan':18,'UAE':19},
          'Winner': {'England':1,'Pakistan':2,'Australia':3,'India':4,'New Zealand':5,'West Indies':6,'Sri Lanka':7,'South Africa':8,'Netherlands':9,'Kenya':10,'Bangladesh':11,'Scotland':12,'Zimbabwe':13,'Canada':14,'Namibia':15,'Bermuda':16,'Ireland':17,'Afghanistan':18,'UAE':19, 'no result':20, 'abandoned': 21, 'tied': 22},
          'Ground':{'The Oval':1, 'Swansea':2, 'Nottingham':3, 'Manchester':4, 'Taunton':5,
       'Birmingham':6, 'Leeds':7, 'Leicester':8, "Lord's":9, 'Bristol':10,
       'Worcester':11, 'Southampton':12, 'Derby':13, 'Tunbridge Wells':14,
       'Chelmsford':15, 'Hyderabad (Sind)':16, 'Chennai':17, 'Gujranwala':18,
       'Hyderabad (Deccan)':19, 'Rawalpindi':20, 'Karachi':21, 'Bengaluru':22,
       'Lahore':23, 'Mumbai':24, 'Peshawar':25, 'Indore':26, 'Kanpur':27, 'Delhi':28,
        'Kolkata':29, 'Faisalabad':30, 'Jaipur':31, 'Ahmedabad':32, 'Chandigarh':33,
       'Cuttack':34, 'Pune':35, 'Nagpur':36, 'Auckland':37, 'Perth':38, 'New Plymouth':39,
       'Melbourne':40, 'Hamilton':41, 'Sydney':42, 'Hobart':43, 'Mackay':44, 'Brisbane':45,
       'Adelaide':46, 'Wellington':47, 'Napier':48, 'Christchurch':49, 'Ballarat':50,
       'Canberra':51, 'Dunedin':52, 'Berri':53, 'Albury':54, 'Vadodara':55,
       'Colombo (RPS)':56, 'Colombo (SSC)':57, 'Gwalior':58, 'Visakhapatnam':59,
       'Patna':60, 'Kandy':61, 'Mohali':62, 'Hove':63, 'Canterbury':64, 'Northampton':65,
       'Cardiff':66, 'Chester-le-Street':67, 'Dublin':68, 'Edinburgh':69,
       'Amstelveen':70, 'Cape Town':71, 'Harare':72, 'Bloemfontein':73,
       'Johannesburg':74, 'Durban':75, 'Potchefstroom':76, 'Paarl':77,
       'Port Elizabeth':78, 'Pietermaritzburg':79, 'Centurion':80, 'East London':81,
       'Kimberley':82, 'Benoni':83, 'Nairobi (Gym)':84, 'Bulawayo':85, 'Kingston':86,
       'Basseterre':87, 'Gros Islet':88, 'Port of Spain':89, 'North Sound':90,
       'Providence':91, "St George's":92, 'Bridgetown':93, 'Dhaka':94, 'Hambantota':95,
       'Pallekele':96, 'Chattogram':97, 'Nelson':98}}
    


# In[9]:


matches.replace(encode, inplace=True)


# In[10]:


dicVal = encode['Winner']
print(dicVal['England']) 
print(list(dicVal.keys())[list(dicVal.values()).index(1)]) 


# In[11]:


dicValue = encode['Ground']
print(dicValue['The Oval']) 
print(list(dicValue.keys())[list(dicValue.values()).index(1)]) 


# In[12]:


matches = matches[['Team 1','Team 2','Ground','Toss Winner','Winner']]
matches.head()


# In[13]:


df = matches


# In[14]:


df.dtypes


# ## Building Variaus Machine learning models 

# In[15]:


from sklearn.model_selection import train_test_split
data_pred_final = df.copy()

# Separate X and y sets
X = data_pred_final.drop(['Winner'], axis=1)
y = data_pred_final["Winner"]


# Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=40, max_depth=10,  random_state=15)

rf.fit(X_train, y_train) 


score1 = rf.score(X_train, y_train)
score2 = rf.score(X_test, y_test)


print("Training set accuracy percentage: ", '%.2f'%(score1*100)+ " %")
print("Test set accuracy percentage: ", '%.2f'%(score2*100)+" %")


# In[17]:


predictions = rf.predict(X_test)


# In[18]:


pred_set = df.drop(['Winner'], axis=1)


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#make a class to run everything
def classification_model(model, data, predictors, outcome):
    model.fit(data[predictors],data[outcome])
    predictions = model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print('Accuracy : %s' % '{0:.2%}'.format(accuracy))
   

    model.fit(data[predictors],data[outcome]) 


# In[40]:


outcome_var=['Winner']
predictor_var = ['Team 1','Team 2','Toss Winner']
model = LogisticRegression()
classification_model(model, df,predictor_var,outcome_var)


# In[21]:


model = RandomForestClassifier(n_estimators=4000, max_depth=50,  random_state=10)
outcome_var = ['Winner']
predictor_var = ['Team 1', 'Team 2', 'Ground', 'Toss Winner']
classification_model(model, df,predictor_var,outcome_var)


# In[41]:


# Input values from the user
team1 = 'England'
team2 = 'India'
tosswinner = 'India'
ground = 'Nagpur'
inputdata=[dicVal[team1],dicVal[team2],dicValue[ground],dicVal[tosswinner]]
inputdata = np.array(inputdata).reshape((1, -1))
output= rf.predict(inputdata)
print(list(dicVal.keys())[list(dicVal.values()).index(output)])


# In[23]:


# Which varibale has the highest prediction value
imp_input = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(imp_input)


# In[24]:



temp1 = matches.filter(["Team 1", "Winner"], axis=1)
temp2 = matches.filter(["Team 1", "Toss Winner"], axis=1)
temp3 = matches.filter(["Team 1", "Team 2", "Winner", "Toss Winner"], axis=1)


# In[36]:


ax = sns.barplot(x="Team 1", y="Winner", data=temp3)
plt.xlabel('Team ', fontsize=14)
plt.ylabel('Count of Wins', fontsize=14)
plt.title('Match Winner', fontsize=14)


# In[37]:


ax1 = sns.barplot(x="Team 1", y="Toss Winner", data=temp3)
plt.xlabel('Team ', fontsize=14)
plt.ylabel('Count of Wins', fontsize=14)
plt.title('Toss Winner', fontsize=14)


# In[39]:


tempValue = matches.filter(["Winner"], axis=1)
fig, ax = plt.subplots()
ax.plot(tempValue)

