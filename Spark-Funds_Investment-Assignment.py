#!/usr/bin/env python
# coding: utf-8

# In[354]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format


# In[355]:


#first load the 'companies.txt' file into dataframes
companies_url = '/Users/balwant/Desktop/DataScience/PG_Diploma_in Machine_Learning_and_AI_March_2020/Assignment/Funds-Investment/data/companies.txt'
companies = pd.read_csv(companies_url, encoding= 'ISO-8859-1',delimiter='\t')
companies.head()


# In[356]:


#After that load the 'rounds2.csv' file into dataframes
#Now change the encoding to convert into redable text.

rounds2_url = '/Users/balwant/Desktop/DataScience/PG_Diploma_in Machine_Learning_and_AI_March_2020/Assignment/Funds-Investment/data/rounds2.csv'
rounds2 = pd.read_csv(rounds2_url,encoding= 'ISO-8859-1',delimiter=',') 
rounds2.head()


# In[357]:


#Here changing the encoding type to avoid encoding issue
companies.permalink = companies.permalink.str.encode('ISO-8859-1').str.decode('ascii', 'ignore')
companies.name = companies.name.str.encode('ISO-8859-1').str.decode('ascii', 'ignore')

rounds2.company_permalink = rounds2.company_permalink.str.encode('ISO-8859-1').str.decode('ascii', 'ignore')


# ## 1.1 - Understanding the Data Set

# In[358]:


# How many unique companies are present in rounds2?
rounds2['company_permalink'] = rounds2['company_permalink'].str.lower()
print(len(rounds2['company_permalink'].unique()))


# In[359]:


#Finding the same result with another way.
rounds2['company_permalink'].describe()


# In[360]:


#How many unique companies are present in the companies file?
companies['permalink'] = companies['permalink'].str.lower()
print(len(companies['permalink'].unique()))


# In[361]:


#Are there any companies in the rounds2 file which are not  present in companies ? Answer Y/N.
unique_companies = pd.DataFrame(companies.permalink.str.lower().unique())
unique_rounds2_companies = pd.DataFrame(rounds2.company_permalink.str.lower().unique())
unique_companies.equals(unique_rounds2_companies)


# In[362]:


#Merge the two data frames so that all variables (columns) in the companies frame are added to the rounds2 data frame.
#Name the merged frame master_frame. How many observations are present in master_frame ? 
master_frame = pd.merge(rounds2, companies, how = 'left', left_on = 'company_permalink', right_on = 'permalink')
len(master_frame.index)
master_frame.head()


# ## Data Cleaning

# In[363]:


#Checking for null values
master_frame.isnull().sum(axis = 0)


# In[364]:


#Inspecting the Null values in percentage.
print(round(100*(master_frame.isnull().sum()/master_frame.shape[0]), 2))


# In[365]:


#Removing or deleting unnecessary columns
master_frame = master_frame.drop(['funding_round_code', 'funding_round_permalink', 'funded_at','permalink', 'homepage_url',
                                 'state_code', 'region', 'city', 'founded_at','status'], axis = 1)


# In[366]:


#Checking for Null values after deletion of unnecessary columns.
print(round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2))


# In[367]:


#Dropping rows based on null columns
master_frame = master_frame[~(master_frame['raised_amount_usd'].isnull() | master_frame['country_code'].isnull() |
                             master_frame['category_list'].isnull())]


# ## 2.1 - Average Values of Investments for Each of these Funding Types

# In[368]:


#Extracting the unique funding_round_type
master_frame.funding_round_type.value_counts()


# In[369]:


#Removing all funding type except the following 4 investment types.
master_frame = master_frame[(master_frame['funding_round_type'] == 'venture') 
                            | (master_frame['funding_round_type'] == 'seed')
                            | (master_frame['funding_round_type'] == 'angel')
                            | (master_frame['funding_round_type'] == 'private_equity')]
master_frame.head()


# In[370]:


#Invesment sutable for spark fund in funding type
funding_type_frame = master_frame.groupby('funding_round_type')['raised_amount_usd'].count().sort_values(ascending=False).to_frame() 
plt.figure(num=None, figsize=(15, 7.5))
bar=sns.barplot(x='funding_round_type', y='raised_amount_usd', data=funding_type_frame.reset_index())
bar.set(xlabel='Funding type', ylabel='Number of Investments')
bar.set_title('Invesment sutable for spark fund in funding type')
plt.show()


# In[371]:


#Average funding amount of fundings type
round(master_frame.groupby('funding_round_type').raised_amount_usd.mean(), 2)
investments_type_mean_frame = master_frame 


# In[372]:


#Keeping rows with only venture type. because Spark Funds wants to invest between 5 to 15 million USD per investment round
master_frame = master_frame[master_frame['funding_round_type'] == 'venture'] 


# In[373]:


#Dropping the column 'funding_round_type' as all funding are belongs to venture type.
master_frame = master_frame.drop(['funding_round_type'], axis = 1)


# ## 3.1 - Analysing the Top 3 English-Speaking Countries

# In[374]:


#For the chosen investment type, make a data frame named top9 with the top nine countries.
#based on the total investment amount each country has received.

top9 = master_frame.pivot_table(values = 'raised_amount_usd', index = 'country_code', aggfunc = 'sum')
top9 = top9.sort_values(by = 'raised_amount_usd', ascending = False)
top9 = top9.iloc[:9, ]
top9


# In[375]:


#Keeping rows with only USA, GBR and IND 'country_codes'.
#As SparksFunds wants to invest in only top three English speaking countries.

master_frame = master_frame[(master_frame['country_code'] == 'USA')
                            | (master_frame['country_code'] == 'GBR')
                            | (master_frame['country_code'] == 'IND')]

master_frame.head()


# ## 5.1 - Sector-wise Investment Analysis
# ####  1. Extract the primary sector of each category list from the category_list column
# 
# ####  2. Use the mapping file 'mapping.csv' to map each primary sector to one of the eight main sectors (Note that ‘Others’ is also considered one of the main sectors)

# In[376]:


#Extracting the primary vector value
master_frame['category_list'] = master_frame['category_list'].apply(lambda x: x.split('|')[0])
master_frame.head()


# In[377]:


#Load the 'mapping.csv' file into dataframes
mapping_url = '/Users/balwant/Desktop/DataScience/PG_Diploma_in Machine_Learning_and_AI_March_2020/Assignment/Funds-Investment/data/mapping.csv'
mapping = pd.read_csv(mapping_url,encoding= 'ISO-8859-1',delimiter=',')
mapping.category_list = mapping.category_list.replace({'0':'na', '2.na' :'2.0'}, regex=True)
mapping.head()


# In[378]:


#Mapping primary sector to main sector
mapping = pd.melt(mapping, id_vars =['category_list'], value_vars = mapping.columns.values[1:]) 
mapping = mapping[~(mapping.value == 0)]
mapping = mapping.drop('value', axis = 1)
mapping = mapping.rename(columns = {"variable":"main_sector"})
mapping.head()


# In[379]:


master_frame = master_frame.merge(mapping, how = 'left', on ='category_list')
master_frame.head()


# In[380]:


master_frame = master_frame[~(master_frame.main_sector.isnull())]
len(master_frame.index)


# In[381]:


master_frame.info()


# In[382]:


#Dropping all rows where investment is not between 5 and 15 million
master_frame = master_frame.drop(master_frame[(master_frame.raised_amount_usd < 5000000)].index)
master_frame = master_frame.drop(master_frame[(master_frame.raised_amount_usd > 15000000)].index)

d1 = master_frame[master_frame['country_code'] == 'USA']
d2 = master_frame[master_frame['country_code'] == 'GBR']
d3 = master_frame[master_frame['country_code'] == 'IND']


#  1. Total number of investments (count)

# In[383]:


#Total number of investments (count) for country code USA
len(d1)


# In[384]:


#Total number of investments (count) for country code GBR
len(d2)


# In[385]:


#Total number of investments (count) for country code IND
len(d3)


# In[386]:


#2. Total amount of investment (USD)  for country code USA
d1['raised_amount_usd'].sum()


# In[387]:


#2. Total amount of investment (USD) for country code GBR
d2['raised_amount_usd'].sum()


# In[388]:


#2. Total amount of investment (USD) for country code IND
d3['raised_amount_usd'].sum()


# In[389]:


d1.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'})


# In[390]:


#3. Top sector (based on count of investments) for country code USA
#4. Second-best sector (based on count of investments) for country code USA
#5. Third-best sector (based on count of investments) for country code USA
#6. Number of investments in the top sector ('Others') for USA
sector_on_investments_count = d1.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)
sector_on_investments_sum = d1.groupby('main_sector')['raised_amount_usd'].sum().sort_values(ascending=False)
sector_on_investments_count


# In[391]:


plt.figure(num=None, figsize=(15, 7.5))
bar=sns.barplot(x='main_sector', y='raised_amount_usd', data=sector_on_investments_count.reset_index())
bar.set(xlabel='Main Sector', ylabel='Number of Investments')
bar.set_title('Sector wise analysis of investment for USA (count wise)')
bar.set_yscale('log')
bar.set_xticklabels(bar.get_xticklabels(), rotation=30)
plt.show()


# In[392]:


plt.figure(num=None, figsize=(15, 7.5))
bar=sns.barplot(x='main_sector', y='raised_amount_usd', data=sector_on_investments_sum.reset_index())
bar.set(xlabel='Main Sector', ylabel='Number of Investments')
bar.set_title('Sector wise analysis of investment for USA (investment amount wise)')
bar.set_yscale('log')
bar.set_xticklabels(bar.get_xticklabels(), rotation=30)
plt.show()


# In[393]:


#Number of investments in the top sector (refer to point 3) for country code USA
top_sector_investments_count  = sector_on_investments_count.iloc[:1]
top_sector_investments_count


# In[394]:


#7. Number of investments in the second-best sector (refer to point 4) for country code USA
second_best_sector_investments_count  = sector_on_investments_count.iloc[1:2]
second_best_sector_investments_count


# In[395]:


# 8. Number of investments in the third-best sector (refer to point 5) for country code USA
third_best_sector_investments_count  = sector_on_investments_count.iloc[2:3]
third_best_sector_investments_count


# In[396]:


#9. Number of investments in the top sector ('Others')
d1[d1['main_sector'] == "Others" ].groupby('company_permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5)


# In[397]:


#10. Number of investments in the top sector ('Social, Finance, Analytics, Advertising')
d1[d1['main_sector'] == "Social, Finance, Analytics, Advertising" ].groupby('company_permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5)


# In[398]:


d2.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'})


# In[399]:


#3. Top sector (based on count of investments) for country code GBR
#4. Second-best sector (based on count of investments) for country code GBR
#5. Third-best sector (based on count of investments) for country code GBR
sector_on_investments_count = d2.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)
sector_on_investments_sum = d2.groupby('main_sector')['raised_amount_usd'].sum().sort_values(ascending=False)
sector_on_investments_count


# In[400]:


plt.figure(num=None, figsize=(15, 7.5))
bar=sns.barplot(x='main_sector', y='raised_amount_usd', data=sector_on_investments_count.reset_index())
bar.set(xlabel='Main Sector', ylabel='Number of Investments')
bar.set_title('Sector wise analysis of investment for GBR (count wise)')
bar.set_xticklabels(bar.get_xticklabels(), rotation=30)
plt.show()


# In[401]:


plt.figure(num=None, figsize=(15, 7.5))
bar=sns.barplot(x = 'main_sector', y = 'raised_amount_usd', data = sector_on_investments_sum.reset_index())
bar.set(xlabel='Main Sector', ylabel='Number of Investments')
bar.set_title('Sector wise analysis of investment for GBR (investment amount wise)')
bar.set_xticklabels(bar.get_xticklabels(), rotation=30)
plt.show()


# In[402]:


#Number of investments in the top sector (refer to point 3) for country code GBR
top_sector_investments_count  = sector_on_investments_count.iloc[:1]
top_sector_investments_count


# In[403]:


#7. Number of investments in the second-best sector (refer to point 4) for country code GBR
second_best_sector_investments_count  = sector_on_investments_count.iloc[1:2]
second_best_sector_investments_count


# In[404]:


#8. Number of investments in the third-best sector (refer to point 5) for country code GBR
third_best_sector_investments_count  = sector_on_investments_count.iloc[2:3]
third_best_sector_investments_count


# In[405]:


#9. Number of investments in the top sector ('Others')
d2[d2['main_sector'] == "Others" ].groupby('company_permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5)


# In[406]:


#10. Number of investments in the top sector ('Social, Finance, Analytics, Advertising')
d2[d2['main_sector'] == "Social, Finance, Analytics, Advertising" ].groupby('company_permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5)


# In[407]:



d3.pivot_table(values = 'raised_amount_usd',index = ['main_sector'], aggfunc = {'sum','count'})


# In[408]:


#3. Top sector (based on count of investments) for country code IND
#4. Second-best sector (based on count of investments) for country code IND
#5. Third-best sector (based on count of investments) for country code IND
sector_on_investments_count = d3.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)
sector_on_investments_sum = d3.groupby('main_sector')['raised_amount_usd'].sum().sort_values(ascending=False)
sector_on_investments_count


# In[409]:


plt.figure(num=None, figsize=(15, 7.5))
bar=sns.barplot(x='main_sector', y='raised_amount_usd', data=sector_on_investments_count.reset_index())
bar.set(xlabel='Main Sector', ylabel='Number of Investments')
bar.set_title('Sector wise analysis of investment for IND (count wise)')
bar.set_xticklabels(bar.get_xticklabels(), rotation=30)
plt.show()


# In[410]:


plt.figure(num=None, figsize=(15, 7.5))
bar=sns.barplot(x='main_sector', y='raised_amount_usd', data=sector_on_investments_sum.reset_index())
bar.set(xlabel='Main Sector', ylabel='Number of Investments')
bar.set_title('Sector wise analysis of investment for IND (investment amount wise)')
bar.set_xticklabels(bar.get_xticklabels(), rotation=30)
plt.show()


# In[411]:


#Number of investments in the top sector (refer to point 3) for country code IND
investments_count  = sector_on_investments_count.iloc[:1]
investments_count


# In[412]:


#7. Number of investments in the second-best sector (refer to point 4) for country code IND
second_best_sector_investments_count  = sector_on_investments_count.iloc[1:2]
second_best_sector_investments_count


# In[413]:


#8. Number of investments in the third-best sector (refer to point 5) for country code IND
third_best_sector_investments_count  = sector_on_investments_count.iloc[2:3]
third_best_sector_investments_count


# In[414]:


#9. Number of investments in the top sector ('Others')
d3[d3['main_sector'] == "Others" ].groupby('company_permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5)


# In[415]:


#10. Number of investments in the top sector ('Social, Finance, Analytics, Advertising')
d3[d3['main_sector'] == "Social, Finance, Analytics, Advertising" ].groupby('company_permalink')['raised_amount_usd'].sum().sort_values(ascending=False).head(5)


#  ## 6.1 - Plots
# 1. A plot showing the fraction of total investments (globally) in angel, venture, seed, and private equity, and the average amount of investment in each funding type. This chart should make it clear that a certain funding type (FT) is best suited for Spark Funds.

# In[416]:


#showing the fraction of total investments (globally) in angel, venture, seed, and private equity, and the average amount of investment in each funding type.
plt.figure(figsize = (15,7.5))
bar = sns.barplot(x = 'funding_round_type', y = 'raised_amount_usd', hue="funding_round_type", data = investments_type_mean_frame)
bar.set(xlabel = 'Funding Type', ylabel = 'Raised Amount   ( 1 Unit = 10M USD)')
bar.set_title('Average amount of investment in each funding type')
plt.legend(title = 'Funding Type', loc = 'upper right')
plt.axhline(5000000, color = 'green')
plt.axhline(15000000, color = 'red')
plt.show()


# 2. A plot showing the top 9 countries against the total amount of investments of funding type FT. This should make the top 3 countries (Country 1, Country 2, and Country 3) very clear.

# In[417]:


#A plot showing the top 9 countries against the total amount of investments of funding type FT. This should make the top 3 countries
plt.figure(figsize = (15,7.5))
bar = sns.barplot(x='country_code', y ='raised_amount_usd', data = top9.reset_index())
bar.set_yscale('log')
bar.set(xlabel='Funding Type', ylabel='Raised Amount')
bar.set_title('showing the top 9 countries against the total amount of investments of funding type')
plt.show()


# 3. A plot showing the number of investments in the top 3 sectors of the top 3 countries on one chart (for the chosen investment type FT)

# In[418]:


#extracting top 3 main sector for country code USA and used for further analysis.
d1_main_sectors_frame = d1.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)
d1_top3_sector_frame = d1_main_sectors_frame.iloc[:3].reset_index()
d1_top3_sector = d1_top3_sector_frame['main_sector']
master_sector_frame = d1[d1['main_sector'].isin(d1_top3_sector)]
master_sector_frame.shape


# In[419]:


#extracting top 3 main sector for country GBR code and used for further analysis.
d2_main_sectors_frame = d2.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)
d2_top3_sector_frame = d2_main_sectors_frame.iloc[:3].reset_index()
d2_top3_sector = d2_top3_sector_frame['main_sector']
master_sector_frame = master_sector_frame.append(d2[d2['main_sector'].isin(d2_top3_sector)], ignore_index=True)
master_sector_frame.shape


# In[420]:


#extracting top 3 main sector for country IND code and used for further analysis.
d3_main_sectors_frame = d3.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)
d3_top3_sector_frame = d3_main_sectors_frame.iloc[:3].reset_index()
d3_top3_sector = d3_top3_sector_frame['main_sector']
master_sector_frame = master_sector_frame.append(d3[d3['main_sector'].isin(d3_top3_sector)], ignore_index=True)
master_sector_frame.shape


# In[421]:


# A plot showing the number of investments in the top 3 sectors of the top 3 countries on one chart
plt.figure(num=None, figsize=(15, 7.5))
s=sns.barplot(x='country_code', y='raised_amount_usd', hue="main_sector", data=master_sector_frame,estimator=lambda x: len(x))
s.set(xlabel='Country', ylabel='Number of Investments')
plt.legend(title = 'Main Sector', loc = 'upper right')
s.set_title('showing the number of investments in the top 3 sectors of the top 3 countries on one chart')
plt.show()

