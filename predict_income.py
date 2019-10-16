import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from pylab import rcParams
import seaborn as sb
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder
import scipy
from scipy.stats.stats import pearsonr
get_ipython().run_line_magic('matplotlib', 'inline')


train = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
subm_df = pd.read_csv('tcd ml 2019-20 income prediction submission file.csv.xls')


train.columns = ['Instance','Year','Gender','Age','Country','CitySize','Profession','UniDegree','Glasses','Hair','Height','Income' ]
test.columns = ['Instance','Year','Gender','Age','Country','CitySize','Profession','UniDegree','Glasses','Hair','Height','Income' ]

train.set_index("Instance", inplace=True)
test.set_index("Instance", inplace=True)


for col in train.columns:
    print(col, train[col].unique())


def process_year(df):
    df['Year'] = df['Year'].fillna(1799)
    cut_points = [1980,1990,2000,2010,2019]
    label_names = ['80-90',"90-00","00-10","10-19"]
    df['Year_ctg'] = pd.cut(df['Year'],cut_points,labels=label_names)
    df.drop(['Year'],axis=1,inplace=True)
    return df

def process_age(df):
    df['Age'] = df['Age'].fillna(0)
    cut_points = [14,25,35,45,55,70,115]
    label_names = ["Teen-Adult","25-35","35-45","45-55","55-70","Senior"]
    df['Age_ctg'] = pd.cut(df['Age'],cut_points,labels=label_names)
    df.drop(['Age'],axis=1,inplace=True)
    return df

def process_height(df):
    cut_points = [94,150,170,195,265]
    label_names = ["94-150","150-170","170-195","195-265"]
    df['Height_ctg'] = pd.cut(df['Height'],cut_points,labels=label_names)
    df.drop(['Height'],axis=1,inplace=True)
    return df

def process_gender(df):
    Gender = 'Gender'
    df[Gender] = df[Gender].fillna(0)
    df.Gender[(df.Gender == '0') | (df.Gender == 'unknown') | (df.Gender == 'other')] = 'Unknown'
    return df

def process_degree(df):
    UniDegree = 'UniDegree'
    df[UniDegree] = df[UniDegree].fillna(0)
    df.UniDegree[(df.UniDegree == '0') | (df.UniDegree == 0)] = 'Unknown'
    return df

def process_hair(df):
    Hair = 'Hair'
    df[Hair] = df[Hair].fillna(0)
    df.Hair[(df.Hair == '0') | (df.Hair == 0)] = 'Unknown'
    return df

import pycountry_convert
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2

def process_country(df):
    Country = 'Country'
    continents = {
        'NA' : 'North America',
        'SA' : 'South America',
        'OC' : 'Australia',
        'AS' : 'Asia',
        'AF' : 'Africa',
        'EU' : 'Europe',
    }

    df.Country[df.Country == 'State of Palestine'] = 'Palestine'
    df.Country[df.Country == 'Timor-Leste'] = 'India'
    df.Country[df.Country == 'DR Congo'] = 'Congo'
    df.Country[df.Country == 'Sao Tome & Principe'] = 'Sao Tome and Principe'
    #pc.map_countries(cn_name_format="default", cn_extras={'Timor-Leste'})
    for cntry in df[Country].unique():
        country_code = country_name_to_country_alpha2(cntry)
        for continent,value in continents:
            continent = country_alpha2_to_continent_code(country_code)
        df.Country[df.Country == cntry] = continent
    return df

def process_citysize(df):
    Income = 'Income'
    CitySize = 'CitySize'
    df[CitySize] = df[CitySize].fillna(0)
    income_mean = df.Income.mean()
    for val in df[CitySize].unique():
        col_mean = df.loc[(df.CitySize == val)].Income.mean()
        mean_dif = col_mean - income_mean
        df.loc[(df.CitySize == val), "CSize_Inc_Mean"] = mean_dif

    return df

def process_prof(df):
    Income = 'Income'
    Profession = 'Profession'
    df[Profession] = df[Profession].fillna(0)
    income_mean = df.Income.mean()
    for val in df[Profession].unique():
        col_mean = df.loc[(df.Profession == val)].Income.mean()
        mean_dif = col_mean - df.Income.mean()
        df.loc[(df.Profession == val), "Prof_Inc_Mean"] = mean_dif

    return df

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    df.drop(column_name,axis=1,inplace=True)
    return df

train = process_country(train)
test = process_country(test)
train = process_year(train)
test = process_year(test)
train = process_age(train)
test = process_age(test)
train = process_height(train)
test = process_height(test)
train = process_gender(train)
test = process_gender(test)
train = process_degree(train)
test = process_degree(test)
train = process_hair(train)
test = process_hair(test)
# train = process_prof(train)
# test = process_prof(test)
# train = process_citysize(train)
# test = process_citysize(test)

#year_piv = train.pivot_table(index=['Year_ctg'],values='Income')
#year_piv.plot.line()

for column in ['Year_ctg','Age_ctg','Height_ctg','Gender','UniDegree','Hair','Country']:
    train = create_dummies(train,column)
    test = create_dummies(test,column)
    #print(train.columns)

train.isnull().any()

holdout = test

columns = ['Year_ctg_80-90','Year_ctg_90-00','Year_ctg_00-10','Year_ctg_10-19','Country_AF','Country_AS','Country_EU','Country_SA','Country_NA','Country_OC','Age_ctg_Teen-Adult','Age_ctg_25-35','Age_ctg_35-45','Age_ctg_45-55','Age_ctg_55-70','Age_ctg_Senior','Height_ctg_94-150','Height_ctg_150-170','Height_ctg_170-195','Height_ctg_195-265','Gender_Unknown','Gender_female','Gender_male','Hair_Unknown','Hair_Red','Hair_Black','Hair_Blond','Hair_Brown','Glasses','UniDegree_Unknown','UniDegree_No','UniDegree_Bachelor','UniDegree_Master','UniDegree_PhD','Glasses']

train[columns].isnull().any()
columns1=['Income']

all_X = train[columns]
all_y = train['Income']

(train_X,test_X,train_y,test_y) = train_test_split(all_X,all_y,test_size=0.3,random_state=1)

lr1 = LinearRegression(normalize=True)
lr1.fit(train_X,train_y)

coefficients = lr1.coef_
feature_importance = pd.Series(coefficients[0],index=train_X.columns)
feature_importance.plot.bar()
list(zip(train_X.columns,coefficients))

predictions = lr1.predict(test_X)
accuracy = np.sqrt(mean_squared_error(test_y,predictions))
print(accuracy)

lr = LinearRegression(normalize=True)
lr.fit(all_X,all_y)

holdout_predictions = lr.predict(holdout[columns])

# submission_df = pd.DataFrame(columns=["Instance", "Income"])
# submission_df["Instance"] = subm_df["Instance"].copy()
# submission_df["Income"] = holdout_obj.copy()
# submission_df.to_csv(r"submission.csv",index=None, header=True)
# #files.download("submission.csv")
