#!/usr/bin/env python
# coding: utf-8

# ### Data Set: police_department_data
# ### Submitted By: Pratik B. Kanase
# ####  kanasepratik2@gmail.com
# #### +91-8087475700

# In[ ]:





# # Project Workflow
# <br>
# 
# 
# ##  Data and Features Understanding
# <br>
# 
# - Understanding Feature Datatypes
# <br>
# 
# - Understanding Feature Characteristics
#     - Missing Data Analysis
#     - Cardinality Analysis
#     - Outlier Analysis
#     - Assumptions
#     
# <hr>
#    
# ## Data Preprocessing
# <br>
# 
# - Missing Data Treatment   
# 
# - Feature Scaling
# 
# <br>
# <hr>    
# 
# ## Data Understanding by Exploratory Data Analysis
# 
# <ol>
#  <li>Graphical EDA Analysis</li>
#     - Univariate EDA <br>
#     - Bivariate EDA<br>
#     <br>
#  <li>Statistical EDA Analysis</li>
#     - Normality Tests <br>
#     - Significance Tests <br>
#   <br>
#  <li>Class Imbalnce Treatment</li>
#     - SMOTE <br>
#     <br>
# </ol>
# 
# <hr>
# 
# ## Feature Engineering
# 
# <ol>
#  <li>Feature Imporatnce</li>
#     - Feature Significance Tests: ANOVA
#     
#  <li>Feature Encoding</li>
#     - Categorical Label Encoding
#      
# </ol>
#  <br>
# <hr>
# 
# ## Feature Importance
# <br>
# - Understanding Feature Importance and Feture Interpretation Using Recursive Feature Elimination
# 
# <br>
#  <br>
# 
# 
# <br>
#  <br>
# <hr>
# 
# ## Model Training
# 
# <br>
# - Applying Different Classification Models to Get the Best Performing Model with Better Results and Stability
# <ol>
#     <br>
#  <li>Random Forest</li>
#     - Evalution of RF Algorithm<br>
#     - Hyperparameter Tunning using Gid Search CV<br>
#     - Cross Validation<br>
#     <br>
#     <br>
#  <li>SVM</li>
#     - SVM Model Evaluation<br>
#     - Interpretaion of performance<br>
#    
# 
#     
# </ol>
# 
# <br>
# <hr>
# 
# ## Comparing & Interpreting Model Performance 
# 
# <br>
# <hr>
# 
# ## Selecting the Best Model 
# 
# <br>
# <hr>
# 
# ## Saving the Model
# 
# <br>
# <hr>
# 
# ## Conclusion
# 

# In[1]:


#Importing the Libraries
import os
import datetime
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


# In[2]:


#Importing the Data
data = pd.read_csv(r"C:\Users\Hp\Desktop\onlinesa\police_department_data (1).csv")
pd.set_option('display.max_rows' ,None)
pd.set_option('display.max_columns' ,None)

#checking the dataset
data.head(10)


# 
# ##  Data and Features Understanding

# In[4]:


#Function to get the basic information related with dataset 
def dataset_info(data):
        """
        Description: This function shll be used to generate basic information about the dataset
        """
        print('\nShape of Dataset',data.shape)
        print('\nNumber of Rows',data.shape[0],'\nNumber of Columns: ',data.shape[1])
        print('\nFeature Names : \n',data.columns.values)
        print('\nInformation about Datatypes: ')
        print('\n%s'%data.info())
        print('\nUnique values per column: \n%s'%data.nunique())
        print('\nNumber of Rows',data.shape[0],'\nNumber of Columns: ',data.shape[1])
        print('\nAny Missing Values in data?: %s'%data.isnull().values.any())
        return(data.profile_report(minimal=True))
    
  
# Calling the dunction
dataset_info(data)


# In[5]:


#Function to check the missing data related with the dataset
def missing_data_analysis(data):
        '''
        This function shall be used for the analysis of the missing data in te
        '''
        print('Any missing datapoints in dataset:',data.isnull().values.any())
        if data.isnull().values.any()==True:
            print('Columnwise missing data present in the dataset')
            missing_data=pd.DataFrame({'total_missing_count':data.isnull().sum(),
                                       'percentage_missing':data.isnull().sum()/data.shape[0]*100,
                                       'datatype':data.dtypes})

            print(missing_data[missing_data.total_missing_count>0])
            sns.heatmap(data.isnull().values)
            
            
            #Counting cells with missing values:(Total number of NA)
            a=sum(data.isnull().values.ravel())
            #Getting total number of cells
            b=np.prod(data.shape)
            #Getting percentage of NA in overall data
            print('\n','\n','Total percentage of missing data :',(a/b)*100,' % \n')
        else:
            print('There is no missing datapoints in dataset')
            sns.heatmap(data.isnull().values)
            
            
missing_data_analysis(data)


# In[6]:



#There are some missing values present in the dataset
#Imputing the missing values using simple imputer


from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")
data['department_district']=imp.fit_transform(data[['department_district']])


# In[7]:



#Data Cleaning
#Converting features in datetime formating and extracting information

data["crime_date"]= pd.to_datetime(data["crime_date"]) 

#extracting the crime hour information for detail analysis
data['crime_time'] = data['crime_date'].dt.time
data['crime_hour'] = data['crime_date'].dt.hour
data['crime_date_'] = data['crime_date'].dt.date

data['crime_date_']= pd.to_datetime(data['crime_date_']) 

#Extracting the crime month, week and crime day information from the crime date 
data['crime_month'] = data['crime_date_'].dt.month
data['crime_month'] = data['crime_month'].apply(lambda x: calendar.month_name[x])
data['crime_dt_week'] = data['crime_date_'].dt.week
data['crime_day'] = data['crime_date_'].dt.day_name()
data['crime_dt_is_weekend'] = np.where(data['crime_day'].isin(['Sunday', 'Saturday']), 'Weekend','Weekday')

#splitting the longitude and latitude values
data['location'] = data.location.str.replace('(', '') 
data['location'] = data.location.str.replace(')', '') 
data[['longitude',"latitude"]]=data.location.str.split(',',expand=True)

#converting datatype from object to float
data[['longitude','latitude']] = data[['longitude','latitude']].astype('float')

#dropping unnecessary features
data.drop(columns=['location','crime_date','crime_time','crime_date_'],inplace=True)

#dropping unnecessary columns and identifiers
data.drop(columns=['incident_id','department_id'],inplace=True)

#checking the dataframe
data.head(10)


# In[ ]:





# ## Data Understanding by Exploratory Data Analysis

# ### Univariate Analysis

# In[9]:


#Univariate Analysis
#Function to draw the count plot 
def univariate_analysis_categorical_countplot(data,label):
    plt.figure(figsize=(50,30))
    ax=sns.countplot(y=data[label],data=data,order = data[label].value_counts().index)
    ax.axes.set_title('Univariate Analysis',fontsize=35)
    ax.set_ylabel(label,fontsize=35)
    ax.set_xlabel('count',fontsize=35)
    ax.tick_params(labelsize=35)
    
    total = len(data[label])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y),fontsize=20)
    
    print('Percentage of datapoints present in class : \n\n',(data[label].value_counts()/data[label].count())*100)
        
    plt.show()
    
#Calling the function
univariate_analysis_categorical_countplot(data,'category')


# In[10]:


#Visualizaing the crime locations with the help of longitude and latitudes


plt.figure(figsize=(15,5))
axis_label_font = {'size':'20', 'color':'#165880'}
axis_tic_font = {'size':'20', 'color':'#165880'}

plt.xlabel("Latituede",axis_label_font)
plt.ylabel("Longatitue",axis_tic_font)


plt.xticks(np.arange(-122.4, -122.6))
plt.yticks(np.arange(37.46, 37.85))

plt.subplot(1,2,1)
plt.scatter(data['longitude'], data['latitude'])
plt.title("Crime Locations")
plt.subplot(1,2,2)
sf=plt.imread(r'C:\Users\Hp\Downloads\New folder (3)\map-of-san-francisco-max.jpg')
plt.xticks([])
plt.yticks([])
plt.imshow(sf)
plt.title("Geographical Map")
plt.show()


# In[11]:


#Univariate analysis of the crime category

univariate_analysis_categorical_countplot(data,'category')


# In[12]:


#Function to draw the percentage count plot for the crime description feature

univariate_analysis_categorical_countplot(data,'crime_description')


# In[13]:


#univariate analysis of department district feature
univariate_analysis_categorical_countplot(data,'department_district')


# In[14]:


#univariate analysis of the resolution feature
univariate_analysis_categorical_countplot(data,'resolution')


# In[15]:


#analysis of the crime hours
univariate_analysis_categorical_countplot(data,'crime_hour')


# In[16]:


#percentage count and the analysis of the feature 'crime month'
univariate_analysis_categorical_countplot(data,'crime_month')


# In[17]:


#Analysis of the top 15 most occuring crime locations

top_addresses = data.address.value_counts()[:15]
plt.figure(figsize=(12, 8))

pos = np.arange(len(top_addresses))
plt.bar(pos, top_addresses.values)
plt.xticks(pos, top_addresses.index, rotation = 70)
plt.title('Top 15 Locations with the most crime')
plt.xlabel('Location')
plt.ylabel('Number of Crimes')
plt.show()


# In[18]:


univariate_analysis_categorical_countplot(data,'crime_dt_week')


# In[19]:


#Analysis of the crime day
univariate_analysis_categorical_countplot(data,'crime_day')


# In[20]:


#analysis of the crime day depending upon the weekend
univariate_analysis_categorical_countplot(data,'crime_dt_is_weekend')


# In[21]:


#Analysis of the crime resoltion in the northen district
univariate_analysis_categorical_countplot(data.loc[data['department_district'] == 'NORTHERN'],'resolution')


# In[22]:


#Analysis of the crime resoltion in the mission district
univariate_analysis_categorical_countplot(data.loc[data['department_district'] == 'MISSION'],'resolution')


# In[23]:


#Crime occurance by hour
hours = data.groupby('crime_hour').size()
plt.plot(hours.values)
plt.xticks(hours.index)
plt.title('Crime Occurence Hour')
plt.ylabel ('Crimes')
plt.xlabel ('Hour')
plt.show()


# In[24]:


#gropby analysis
data.groupby('resolution').count()


# In[25]:


#aggregation analysis
data.groupby('department_district').count()


# In[27]:


#dropping unnecessary columns
data.drop(columns=['address','crime_dt_week','crime_dt_is_weekend'],inplace=True)


# In[ ]:





# ## Stattistical Analysis

# In[38]:


from statsmodels.formula.api import ols
from scipy.stats import chi2
def chi_square_test(categorical_feature1,categorical_feature2,data):
     
            dataset_table=pd.crosstab(data[categorical_feature1],data[categorical_feature2])
            Observed_Values = dataset_table.values
            val=stats.chi2_contingency(dataset_table)
            Expected_Values=val[3]
            no_of_rows=dataset_table.shape[0]
            no_of_columns=dataset_table.shape[1]
            ddof=(no_of_rows-1)*(no_of_columns-1)
            alpha = 0.05
            chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
            chi_square_statistic=chi_square[0]+chi_square[1]
            critical_value=chi2.ppf(q=1-alpha,df=ddof)
            p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
        
            print('Chi square test between features: ',categorical_feature1,' and ',categorical_feature2)
            print("chi-square statistic:-",chi_square_statistic)
            print('critical_value:',critical_value)
            print('Significance level: ',alpha)
            print('Degree of Freedom: ',ddof)
            print('p-value:',p_value)
            #if chi_square_statistic>=critical_value:
            #print("Reject H0,There is a relationship between 2 categorical variables")
            #else:
            #print("Retain H0,There is no relationship between 2 categorical variables")
            if p_value<=alpha:
                print("p_value < %s \nTest Results  Reject H0,There is a relationship between 2 categorical variables \n\n"%(alpha))
            else:
                print("p_value > %s \nTest Results : Retain H0,There is no relationship between 2 categorical variables \n\n"%(alpha))
    


# In[41]:


#Performing chi square test for multiple categorical columns
for col in ['category','department_district', 'resolution',
       'crime_month', 'crime_day']:
    chi_square_test('crime_description',col,data)
    


# In[ ]:





# ## Feature Engineering

# ### Combining rare categories

# ## Rare values are categories within a categorical variable that are present only in a small percentage of the observations. There is no rule of thumb to determine how small is a small percentage, but typically, any value below 5 % can be considered rare.

# In[43]:


#Finding rare variables in train and test datasets
for col in ['category', 'resolution']:

    temp_df = pd.Series(data[col].value_counts() / len(data) )

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.05, color='red')
    plt.show()


# In[44]:


def find_non_rare_labels(df, variable, tolerance):
    
    temp = df.groupby([variable])[variable].count() / len(df)
    
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    
    return non_rare

def rare_encoding(data,variable, tolerance):

    data = data.copy()
    
    # find the most frequent category
    frequent_cat = find_non_rare_labels(data, variable, tolerance)

    # re-group rare labels
    data[variable] = np.where(data[variable].isin(
        frequent_cat), data[variable], 'Rare')
    
    return data

for variable in ['category', 'resolution']:
    
    data = rare_encoding(data, variable, 0.05)


    


# In[45]:



#Converting the rare classes
for variable in ['crime_description']:
    
    data = rare_encoding(data, variable, 0.025)


# In[ ]:





# ## Building Machine Learning Models¶
# 

# In[ ]:





# In[46]:


#Splitting the data


# In[47]:


X=data.drop(columns=['crime_description'])
y=data['crime_description']

#Splitting Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[ ]:





# ## Feature Encoding

# ## Performing Label Encoding

# In[48]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoding_columns=['department_district','crime_month','crime_day','category',
                 'resolution']

for col in encoding_columns:
    X_train[col]=le.fit_transform(X_train[col])
    
    X_test[col]=le.fit_transform(X_test[col])


# In[49]:


y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)


# In[ ]:





# In[50]:


#Importing the libriries

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn import metrics


# ## Applying the support Vector Classifier for Multiclass Classification

# In[54]:


print('Model: Support Vector Classifier \n\n')

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.fit(X_train,y_train)

# compute confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(conf_matrix)


# plot confusion matrix
plt.subplots(figsize=(35,35))
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix \n\n')


# compute evaluation metrics
print('Accuracy Score: \n', metrics.accuracy_score(y_test, y_pred),'\n') # accuracy
print('Precision Score: \n',metrics.precision_score(y_test, y_pred,average = 'weighted',labels=np.unique(y_pred)),'\n\n') # precision
print('Recall:  \n',metrics.recall_score(y_test, y_pred, average = 'weighted',labels=np.unique(y_pred)),'\n\n') 
print('F1 Score: \n',metrics.f1_score(y_test, y_pred, average = 'weighted',labels=np.unique(y_pred)),'\n\n') # F1 score


# In[ ]:





# ## Applying the Random Forest for Multiclass Classification

# In[52]:


# applying random forest model
rand_clf = RandomForestClassifier(criterion= 'entropy',n_estimators=115,
                                          max_depth = 12,min_samples_leaf = 1,min_samples_split= 5,random_state=6)

rand_clf.fit(X_train,y_train)

y_pred = rand_clf.predict(X_test)

conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(conf_matrix,'\ng')

# plot confusion matrix
plt.subplots(figsize=(35,35))
sns.heatmap(conf_matrix, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix')

# compute evaluation metrics
print('Accuracy Score: \n', metrics.accuracy_score(y_test, y_pred),'\n') 
print('Precision Score: \n',metrics.precision_score(y_test, y_pred,average = 'weighted',labels=np.unique(y_pred)),'\n\n') # precision
print('Recall:  \n',metrics.recall_score(y_test, y_pred, average = 'weighted',labels=np.unique(y_pred)),'\n\n') 
print('F1 Score: \n',metrics.f1_score(y_test, y_pred, average = 'weighted',labels=np.unique(y_pred)),'\n\n') 


# In[ ]:


#saving models
import pickle

#Random forest model 
with open( 'random_forest.sav', 'wb') as f:
pickle.dump(rand_clf,f)
    
#svm model
with open('svm_model.sav', 'wb') as f:
pickle.dump(model,f)

