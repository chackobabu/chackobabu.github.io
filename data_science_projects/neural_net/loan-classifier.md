---
layout: default
title: Loan Classifier
mathjax: true
---

#### **Loan Classifier**
<br>
**This project was part of a Data Analysis course I did on <a href="https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/" target="_blank" rel="noopener noreferrer">www.udemy.com</a>**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

**Skip to section of interest**

**1) [Importing the dataset](#0)**

**2) [Exploratory Data Analysis](#1)**

**3) [Data Preprocessing](#2)**

**4) [Creating Dummys and categorical variables](#4)**

**5) [Training the model](#5)**


```python
sns.set_style("whitegrid")
```

<a id=0></a>

**Importing the dataset**

Data: a subset of the LendingClub DataSet obtained from Kaggle: <a href="https://www.kaggle.com/wordsforthewise/lending-club" target="_blank" rel="noopener noreferrer">https://www.kaggle.com/wordsforthewise/lending-club</a>

**Loading the description of vars**


```python
data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')
```


```python
data_info 
```



<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Description</th>
    </tr>
    <tr>
      <th>LoanStatNew</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>The listed amount of the loan applied for by t...</td>
    </tr>
    <tr>
      <th>term</th>
      <td>The number of payments on the loan. Values are...</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>The monthly payment owed by the borrower if th...</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>sub_grade</th>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>emp_title</th>
      <td>The job title supplied by the Borrower when ap...</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>Employment length in years. Possible values ar...</td>
    </tr>
    <tr>
      <th>home_ownership</th>
      <td>The home ownership status provided by the borr...</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>The self-reported annual income provided by th...</td>
    </tr>
    <tr>
      <th>verification_status</th>
      <td>Indicates if income was verified by LC, not ve...</td>
    </tr>
    <tr>
      <th>issue_d</th>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>loan_status</th>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>purpose</th>
      <td>A category provided by the borrower for the lo...</td>
    </tr>
    <tr>
      <th>title</th>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>zip_code</th>
      <td>The first 3 numbers of the zip code provided b...</td>
    </tr>
    <tr>
      <th>addr_state</th>
      <td>The state provided by the borrower in the loan...</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>A ratio calculated using the borrower’s total ...</td>
    </tr>
    <tr>
      <th>earliest_cr_line</th>
      <td>The month the borrower's earliest reported cre...</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>The number of open credit lines in the borrowe...</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>Revolving line utilization rate, or the amount...</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>The total number of credit lines currently in ...</td>
    </tr>
    <tr>
      <th>initial_list_status</th>
      <td>The initial listing status of the loan. Possib...</td>
    </tr>
    <tr>
      <th>application_type</th>
      <td>Indicates whether the loan is an individual ap...</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>Number of public record bankruptcies</td>
    </tr>
  </tbody>
</table>
</div>
</pre>


**...defining a function to get the description of any feature when required**


```python
def get_description(feature):
    print(data_info.loc[feature]['Description']) 
```


```python
get_description('mort_acc')
```

    Number of mortgage accounts.


**Loading in the data**

With the historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model that can predict whether or not a borrower will pay back their loan


```python
df = pd.read_csv('../DATA/lending_club_loan_two.csv')
```


```python
df.head()
```



<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>...</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>application_type</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>11.44</td>
      <td>329.48</td>
      <td>B</td>
      <td>B4</td>
      <td>Marketing</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>117000.0</td>
      <td>...</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>36369.0</td>
      <td>41.8</td>
      <td>25.0</td>
      <td>w</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0174 Michelle Gateway\nMendozaberg, OK 22690</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8000.0</td>
      <td>36 months</td>
      <td>11.99</td>
      <td>265.68</td>
      <td>B</td>
      <td>B5</td>
      <td>Credit analyst</td>
      <td>4 years</td>
      <td>MORTGAGE</td>
      <td>65000.0</td>
      <td>...</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>20131.0</td>
      <td>53.3</td>
      <td>27.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1076 Carney Fort Apt. 347\nLoganmouth, SD 05113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15600.0</td>
      <td>36 months</td>
      <td>10.49</td>
      <td>506.97</td>
      <td>B</td>
      <td>B3</td>
      <td>Statistician</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>43057.0</td>
      <td>...</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>11987.0</td>
      <td>92.2</td>
      <td>26.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87025 Mark Dale Apt. 269\nNew Sabrina, WV 05113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7200.0</td>
      <td>36 months</td>
      <td>6.49</td>
      <td>220.65</td>
      <td>A</td>
      <td>A2</td>
      <td>Client Advocate</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>54000.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5472.0</td>
      <td>21.5</td>
      <td>13.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>823 Reid Ford\nDelacruzside, MA 00813</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24375.0</td>
      <td>60 months</td>
      <td>17.27</td>
      <td>609.33</td>
      <td>C</td>
      <td>C5</td>
      <td>Destiny Management Inc.</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>55000.0</td>
      <td>...</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>24584.0</td>
      <td>69.8</td>
      <td>43.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>679 Luna Roads\nGreggshire, VA 11650</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>
</pre>



```python
df.info() #there are features that have missing values
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 396030 entries, 0 to 396029
    Data columns (total 27 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   loan_amnt             396030 non-null  float64
     1   term                  396030 non-null  object 
     2   int_rate              396030 non-null  float64
     3   installment           396030 non-null  float64
     4   grade                 396030 non-null  object 
     5   sub_grade             396030 non-null  object 
     6   emp_title             373103 non-null  object 
     7   emp_length            377729 non-null  object 
     8   home_ownership        396030 non-null  object 
     9   annual_inc            396030 non-null  float64
     10  verification_status   396030 non-null  object 
     11  issue_d               396030 non-null  object 
     12  loan_status           396030 non-null  object 
     13  purpose               396030 non-null  object 
     14  title                 394275 non-null  object 
     15  dti                   396030 non-null  float64
     16  earliest_cr_line      396030 non-null  object 
     17  open_acc              396030 non-null  float64
     18  pub_rec               396030 non-null  float64
     19  revol_bal             396030 non-null  float64
     20  revol_util            395754 non-null  float64
     21  total_acc             396030 non-null  float64
     22  initial_list_status   396030 non-null  object 
     23  application_type      396030 non-null  object 
     24  mort_acc              358235 non-null  float64
     25  pub_rec_bankruptcies  395495 non-null  float64
     26  address               396030 non-null  object 
    dtypes: float64(12), object(15)
    memory usage: 81.6+ MB


**Our label is loan_status. The goal of the model is to predict loan_status based on other features. This is a classification problem.**


```python
df.describe(include='all').transpose()
```



<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14113.9</td>
      <td>8357.44</td>
      <td>500</td>
      <td>8000</td>
      <td>12000</td>
      <td>20000</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>term</th>
      <td>396030</td>
      <td>2</td>
      <td>36 months</td>
      <td>302005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.6394</td>
      <td>4.47216</td>
      <td>5.32</td>
      <td>10.49</td>
      <td>13.33</td>
      <td>16.49</td>
      <td>30.99</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>431.85</td>
      <td>250.728</td>
      <td>16.08</td>
      <td>250.33</td>
      <td>375.43</td>
      <td>567.3</td>
      <td>1533.81</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>396030</td>
      <td>7</td>
      <td>B</td>
      <td>116018</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sub_grade</th>
      <td>396030</td>
      <td>35</td>
      <td>B3</td>
      <td>26655</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>emp_title</th>
      <td>373103</td>
      <td>173105</td>
      <td>Teacher</td>
      <td>4389</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>377729</td>
      <td>11</td>
      <td>10+ years</td>
      <td>126041</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>home_ownership</th>
      <td>396030</td>
      <td>6</td>
      <td>MORTGAGE</td>
      <td>198348</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>74203.2</td>
      <td>61637.6</td>
      <td>0</td>
      <td>45000</td>
      <td>64000</td>
      <td>90000</td>
      <td>8.70658e+06</td>
    </tr>
    <tr>
      <th>verification_status</th>
      <td>396030</td>
      <td>3</td>
      <td>Verified</td>
      <td>139563</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>issue_d</th>
      <td>396030</td>
      <td>115</td>
      <td>Oct-2014</td>
      <td>14846</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>loan_status</th>
      <td>396030</td>
      <td>2</td>
      <td>Fully Paid</td>
      <td>318357</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>purpose</th>
      <td>396030</td>
      <td>14</td>
      <td>debt_consolidation</td>
      <td>234507</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>title</th>
      <td>394275</td>
      <td>48817</td>
      <td>Debt consolidation</td>
      <td>152472</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.3795</td>
      <td>18.0191</td>
      <td>0</td>
      <td>11.28</td>
      <td>16.91</td>
      <td>22.98</td>
      <td>9999</td>
    </tr>
    <tr>
      <th>earliest_cr_line</th>
      <td>396030</td>
      <td>684</td>
      <td>Oct-2000</td>
      <td>3017</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.3112</td>
      <td>5.13765</td>
      <td>0</td>
      <td>8</td>
      <td>10</td>
      <td>14</td>
      <td>90</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.178191</td>
      <td>0.530671</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>86</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15844.5</td>
      <td>20591.8</td>
      <td>0</td>
      <td>6025</td>
      <td>11181</td>
      <td>19620</td>
      <td>1.74327e+06</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>395754</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>53.7917</td>
      <td>24.4522</td>
      <td>0</td>
      <td>35.8</td>
      <td>54.8</td>
      <td>72.9</td>
      <td>892.3</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>396030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.4147</td>
      <td>11.887</td>
      <td>2</td>
      <td>17</td>
      <td>24</td>
      <td>32</td>
      <td>151</td>
    </tr>
    <tr>
      <th>initial_list_status</th>
      <td>396030</td>
      <td>2</td>
      <td>f</td>
      <td>238066</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>application_type</th>
      <td>396030</td>
      <td>3</td>
      <td>INDIVIDUAL</td>
      <td>395319</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>358235</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.81399</td>
      <td>2.14793</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>34</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>395495</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.121648</td>
      <td>0.356174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>address</th>
      <td>396030</td>
      <td>393700</td>
      <td>USNS Johnson\nFPO AE 05113</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
</pre>


<a id=1></a>
**1) EDA**

number of loans that were fully paid vs number of loans that were paid off


```python
sns.countplot('loan_status', data=df)
plt.show()
```

<img src="/assets/img/loan-classifier/output_20_0.png" alt="output_20" width="75%">


Fully paid loans are much more in number compared to charged off

**Histogram of loan amount**


```python
plt.figure(figsize=(12,5))
sns.distplot(df['loan_amnt'], kde=False)
plt.show()
```

<img src="/assets/img/loan-classifier/output_23_0.png" alt="output_23"  width="75%">

No definite trend really. However, one could say loan amounts tend to be concentrated around 5000-10000 region. 

There are many spikes. 


```python
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.show()
```

<img src="/assets/img/loan-classifier/output_25_0.png" alt="output_25"  width="75%">


Almost perfect correlation between installment and loan_amnt noticed. Exploring this relationship further....


```python
df['installment'].describe()
```




    count    396030.000000
    mean        431.849698
    std         250.727790
    min          16.080000
    25%         250.330000
    50%         375.430000
    75%         567.300000
    max        1533.810000
    Name: installment, dtype: float64




```python
df['loan_amnt'].describe()
```




    count    396030.000000
    mean      14113.888089
    std        8357.441341
    min         500.000000
    25%        8000.000000
    50%       12000.000000
    75%       20000.000000
    max       40000.000000
    Name: loan_amnt, dtype: float64




```python
plt.figure(figsize=(15,10))
sns.scatterplot(x='loan_amnt', y='installment', data=df)
plt.show()
```

<img src="/assets/img/loan-classifier/output_29_0.png" alt="output_29"  width="75%">


relationship between loan_status and loan_amnt


```python
plt.figure(figsize=(15,10))
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.show()
```

<img src="/assets/img/loan-classifier/output_31_0.png" alt="output_31"  width="75%">


There is not much difference in the mean amount of loan between the two statuses. 


```python
df.groupby('loan_status')['loan_amnt'].describe()
```



<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>77673.0</td>
      <td>15126.300967</td>
      <td>8505.090557</td>
      <td>1000.0</td>
      <td>8525.0</td>
      <td>14000.0</td>
      <td>20000.0</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>318357.0</td>
      <td>13866.878771</td>
      <td>8302.319699</td>
      <td>500.0</td>
      <td>7500.0</td>
      <td>12000.0</td>
      <td>19225.0</td>
      <td>40000.0</td>
    </tr>
  </tbody>
</table>
</div>
</pre>


Exploring grades given by lending club


```python
list(df['grade'].sort_values().unique())
```




    ['A', 'B', 'C', 'D', 'E', 'F', 'G']




```python
list(df['sub_grade'].sort_values().unique())
```




    ['A1',
     'A2',
     'A3',
     'A4',
     'A5',
     'B1',
     'B2',
     'B3',
     'B4',
     'B5',
     'C1',
     'C2',
     'C3',
     'C4',
     'C5',
     'D1',
     'D2',
     'D3',
     'D4',
     'D5',
     'E1',
     'E2',
     'E3',
     'E4',
     'E5',
     'F1',
     'F2',
     'F3',
     'F4',
     'F5',
     'G1',
     'G2',
     'G3',
     'G4',
     'G5']




```python
plt.figure(figsize=(15,10))
sns.countplot('grade', hue='loan_status', data=df.sort_values(by='grade'))
plt.show()
```

<img src="/assets/img/loan-classifier/output_37_0.png" alt="output_37"  width="75%">


```python
plt.figure(figsize=(17,7))
sns.countplot('sub_grade', data=df.sort_values(by='sub_grade'))
plt.show()
```

<img src="/assets/img/loan-classifier/output_38_0.png" alt="output_38"  width="75%">




```python
plt.figure(figsize=(17,7))
sns.countplot('sub_grade', hue='loan_status', data=df.sort_values(by='sub_grade'))
plt.show()
```

<img src="/assets/img/loan-classifier/output_39_0.png" alt="output_39"  width="75%">



Isolating F and G subgrades...


```python
F_G = [x for x in list(df['sub_grade'].unique()) if x.startswith('F') | x.startswith('G')]
```


```python
df_temp = df.set_index('sub_grade').loc[F_G].reset_index()
```


```python
plt.figure(figsize=(17,7))
sns.countplot('sub_grade', hue='loan_status', data=df_temp.sort_values(by='sub_grade'))
plt.show()
```

<img src="/assets/img/loan-classifier/output_43_0.png" alt="output_43"  width="75%">


**creating a dummy for loan_status**


```python
df['loan_repaid'] = df['loan_status'].replace({'Fully Paid':1, 'Charged Off':0})
```

**correlation among loan_repaid and other variables**

Not sure if this makes much sense since loan_repaid is a categorical variable. Doing it because it was a challenge.


```python
df.corr()['loan_repaid'][:-1].sort_values().plot(kind='bar')
plt.show()
```

<img src="/assets/img/loan-classifier/output_47_0.png" alt="output_47"  width="75%">


<a id=2></a>
**Data Preprocessing**

Percentage of missing values in each variable


```python
df.isnull().sum()*100/len(df)
```




    loan_amnt               0.000000
    term                    0.000000
    int_rate                0.000000
    installment             0.000000
    grade                   0.000000
    sub_grade               0.000000
    emp_title               5.789208
    emp_length              4.621115
    home_ownership          0.000000
    annual_inc              0.000000
    verification_status     0.000000
    issue_d                 0.000000
    loan_status             0.000000
    purpose                 0.000000
    title                   0.443148
    dti                     0.000000
    earliest_cr_line        0.000000
    open_acc                0.000000
    pub_rec                 0.000000
    revol_bal               0.000000
    revol_util              0.069692
    total_acc               0.000000
    initial_list_status     0.000000
    application_type        0.000000
    mort_acc                9.543469
    pub_rec_bankruptcies    0.135091
    address                 0.000000
    loan_repaid             0.000000
    dtype: float64



examining 'emp_title' and 'emp_length', and trying to decide if the missing values can be filled in, or if they can be dropped. 


```python
get_description('emp_title')
```

    The job title supplied by the Borrower when applying for the loan.*



```python
get_description('emp_length')
```

    Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 



```python
df['emp_title'].value_counts()
```




    Teacher                     4389
    Manager                     4250
    Registered Nurse            1856
    RN                          1846
    Supervisor                  1830
                                ... 
    maint mach                     1
    back office medical asst       1
    DBL Limited                    1
    Development Accountant         1
    Meadowbrookhealthcare          1
    Name: emp_title, Length: 173105, dtype: int64



There are 173105 employment titles, far too many to create dummy variables for (if we are to use this category in analysis). So dropping them. 


```python
df.drop('emp_title', axis=1, inplace=True)
```

**exploring employment length**


```python
plt.figure(figsize=(17,7))
sns.countplot(df['emp_length'], order=['< 1 year','1 year', '2 years', '3 years','4 years',\
                                       '5 years','6 years', '7 years', '8 years', '9 years','10+ years'])
plt.show()
```

<img src="/assets/img/loan-classifier/output_58_0.png" alt="output_58"  width="75%">


```python
plt.figure(figsize=(17,7))
sns.countplot(df['emp_length'], hue=df['loan_status'], order=['< 1 year','1 year', '2 years', '3 years','4 years',\
                                       '5 years','6 years', '7 years', '8 years', '9 years','10+ years'])
plt.show()
```

<img src="/assets/img/loan-classifier/output_59_0.png" alt="output_59"  width="75%">


looking for relationships between loan_status and employment length


```python
df_ = df.groupby('emp_length')['loan_status'].value_counts(normalize=True).to_frame()
```


```python
df_
```



<pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>loan_status</th>
    </tr>
    <tr>
      <th>emp_length</th>
      <th>loan_status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">1 year</th>
      <th>Fully Paid</th>
      <td>0.800865</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.199135</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">10+ years</th>
      <th>Fully Paid</th>
      <td>0.815814</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.184186</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2 years</th>
      <th>Fully Paid</th>
      <td>0.806738</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.193262</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3 years</th>
      <th>Fully Paid</th>
      <td>0.804769</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.195231</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4 years</th>
      <th>Fully Paid</th>
      <td>0.807615</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.192385</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">5 years</th>
      <th>Fully Paid</th>
      <td>0.807813</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.192187</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">6 years</th>
      <th>Fully Paid</th>
      <td>0.810806</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.189194</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">7 years</th>
      <th>Fully Paid</th>
      <td>0.805226</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.194774</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">8 years</th>
      <th>Fully Paid</th>
      <td>0.800240</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.199760</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">9 years</th>
      <th>Fully Paid</th>
      <td>0.799530</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.200470</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">&lt; 1 year</th>
      <th>Fully Paid</th>
      <td>0.793128</td>
    </tr>
    <tr>
      <th>Charged Off</th>
      <td>0.206872</td>
    </tr>
  </tbody>
</table>
</div>
</pre>



```python
df_.xs('Charged Off', level=1).plot(kind='bar', legend=False, figsize=(10,7))
plt.show()
```

<img src="/assets/img/loan-classifier/output_63_0.png" alt="output_63"  width="75%">


The charged off rates are very similar across categories. It is unlikely that this variable will have any use for us. 
dropping 'emp_length'


```python
df.drop('emp_length', axis=1, inplace=True)
```


```python
df.isnull().sum()*100/len(df)
```




    loan_amnt               0.000000
    term                    0.000000
    int_rate                0.000000
    installment             0.000000
    grade                   0.000000
    sub_grade               0.000000
    home_ownership          0.000000
    annual_inc              0.000000
    verification_status     0.000000
    issue_d                 0.000000
    loan_status             0.000000
    purpose                 0.000000
    title                   0.443148
    dti                     0.000000
    earliest_cr_line        0.000000
    open_acc                0.000000
    pub_rec                 0.000000
    revol_bal               0.000000
    revol_util              0.069692
    total_acc               0.000000
    initial_list_status     0.000000
    application_type        0.000000
    mort_acc                9.543469
    pub_rec_bankruptcies    0.135091
    address                 0.000000
    loan_repaid             0.000000
    dtype: float64



checking the contents of 'title' column and 'purpose' column


```python
df['title'].unique()
```




    array(['Vacation', 'Debt consolidation', 'Credit card refinancing', ...,
           'Credit buster ', 'Loanforpayoff', 'Toxic Debt Payoff'],
          dtype=object)




```python
df['title'].nunique()
```




    48817




```python
df['purpose'].unique()
```




    array(['vacation', 'debt_consolidation', 'credit_card',
           'home_improvement', 'small_business', 'major_purchase', 'other',
           'medical', 'wedding', 'car', 'moving', 'house', 'educational',
           'renewable_energy'], dtype=object)




```python
df['purpose'].nunique()
```




    14



Looks like the column 'title' has way too many unique categories to be of any use for us. And also, they seem to be just sub-categories of column 'purpose'. 

Dropping 'title'


```python
df.drop('title', axis=1, inplace=True)
```


```python
get_description('mort_acc')
```

    Number of mortgage accounts.



```python
df['mort_acc'].value_counts()
```




    0.0     139777
    1.0      60416
    2.0      49948
    3.0      38049
    4.0      27887
    5.0      18194
    6.0      11069
    7.0       6052
    8.0       3121
    9.0       1656
    10.0       865
    11.0       479
    12.0       264
    13.0       146
    14.0       107
    15.0        61
    16.0        37
    17.0        22
    18.0        18
    19.0        15
    20.0        13
    24.0        10
    22.0         7
    21.0         4
    25.0         4
    27.0         3
    23.0         2
    32.0         2
    26.0         2
    31.0         2
    30.0         1
    28.0         1
    34.0         1
    Name: mort_acc, dtype: int64




```python
df['mort_acc'].isna().sum()
```




    37795




```python
df.corr()['mort_acc'].sort_values(ascending=False)
```




    mort_acc                1.000000
    total_acc               0.381072
    annual_inc              0.236320
    loan_amnt               0.222315
    revol_bal               0.194925
    installment             0.193694
    open_acc                0.109205
    loan_repaid             0.073111
    pub_rec_bankruptcies    0.027239
    pub_rec                 0.011552
    revol_util              0.007514
    dti                    -0.025439
    int_rate               -0.082583
    Name: mort_acc, dtype: float64



**Mortgage account correlates most with total_acc.**

**Steps to fill in missing 'mort_acc' values**

    1) get the mean of 'mort_acc' values per group of 'total_acc'

    2) fill in the mean where the 'mort_acc' values are missing, according to corresding 'total_acc' value


```python
dict_ = df.groupby('total_acc')['mort_acc'].agg('mean').to_dict()
```


```python
dict_.get(7)
```




    0.22169531713100177




```python
def fill_mort_acc(cols):
    total_acc = cols[0]
    mort_acc = cols[1]
    
    if pd.isnull(mort_acc):
        return dict_.get(total_acc)
    else:
        return mort_acc
```


```python
df['mort_acc'] = df[['total_acc','mort_acc']].apply(fill_mort_acc, axis=1)
```


```python
df.isna().sum()*100/len(df)
```




    loan_amnt               0.000000
    term                    0.000000
    int_rate                0.000000
    installment             0.000000
    grade                   0.000000
    sub_grade               0.000000
    home_ownership          0.000000
    annual_inc              0.000000
    verification_status     0.000000
    issue_d                 0.000000
    loan_status             0.000000
    purpose                 0.000000
    dti                     0.000000
    earliest_cr_line        0.000000
    open_acc                0.000000
    pub_rec                 0.000000
    revol_bal               0.000000
    revol_util              0.069692
    total_acc               0.000000
    initial_list_status     0.000000
    application_type        0.000000
    mort_acc                0.000000
    pub_rec_bankruptcies    0.135091
    address                 0.000000
    loan_repaid             0.000000
    dtype: float64




```python
len(df)
```




    396030



revol_util and pub_rec_bankruptcies are missing for negligible number of observations. (0.2 % of observations). dropping observations for which these variables are missing


```python
df.dropna(subset=['revol_util','pub_rec_bankruptcies'], how='any', inplace=True)
```


```python
df.isna().sum()*100/len(df)
```




    loan_amnt               0.0
    term                    0.0
    int_rate                0.0
    installment             0.0
    grade                   0.0
    sub_grade               0.0
    home_ownership          0.0
    annual_inc              0.0
    verification_status     0.0
    issue_d                 0.0
    loan_status             0.0
    purpose                 0.0
    dti                     0.0
    earliest_cr_line        0.0
    open_acc                0.0
    pub_rec                 0.0
    revol_bal               0.0
    revol_util              0.0
    total_acc               0.0
    initial_list_status     0.0
    application_type        0.0
    mort_acc                0.0
    pub_rec_bankruptcies    0.0
    address                 0.0
    loan_repaid             0.0
    dtype: float64



<a id=4></a>

**Categorical variables and dummy variables**

List of non-numeric columns


```python
df.dtypes.unique()
```




    array([dtype('float64'), dtype('O'), dtype('int64')], dtype=object)




```python
df.dtypes[df.dtypes == 'O'].index
```




    Index(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
           'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',
           'initial_list_status', 'application_type', 'address'],
          dtype='object')



**Looking at each non-numeric column one by one**

Term


```python
df['term'].unique()
```




    array([' 36 months', ' 60 months'], dtype=object)




```python
df['term'] = df['term'].replace({' 36 months':36, ' 60 months':60})
```

**Grade**

since grade is only a broader category of subgrade, we can drop grade


```python
df.drop('grade', axis=1, inplace=True)
```


```python
df = pd.get_dummies(data=df, columns=['sub_grade'], drop_first=True)
```


```python
df.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'home_ownership',
           'annual_inc', 'verification_status', 'issue_d', 'loan_status',
           'purpose', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
           'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'address',
           'loan_repaid', 'sub_grade_A2', 'sub_grade_A3', 'sub_grade_A4',
           'sub_grade_A5', 'sub_grade_B1', 'sub_grade_B2', 'sub_grade_B3',
           'sub_grade_B4', 'sub_grade_B5', 'sub_grade_C1', 'sub_grade_C2',
           'sub_grade_C3', 'sub_grade_C4', 'sub_grade_C5', 'sub_grade_D1',
           'sub_grade_D2', 'sub_grade_D3', 'sub_grade_D4', 'sub_grade_D5',
           'sub_grade_E1', 'sub_grade_E2', 'sub_grade_E3', 'sub_grade_E4',
           'sub_grade_E5', 'sub_grade_F1', 'sub_grade_F2', 'sub_grade_F3',
           'sub_grade_F4', 'sub_grade_F5', 'sub_grade_G1', 'sub_grade_G2',
           'sub_grade_G3', 'sub_grade_G4', 'sub_grade_G5'],
          dtype='object')




```python
df.select_dtypes('object').columns
```




    Index(['home_ownership', 'verification_status', 'issue_d', 'loan_status',
           'purpose', 'earliest_cr_line', 'initial_list_status',
           'application_type', 'address'],
          dtype='object')




```python
for each in ['verification_status', 'application_type', 'initial_list_status', 'purpose']:
    print(f"\n{each} :\n")
    print(df[each].value_counts())
    
```

    
    verification_status :
    
    Verified           139451
    Source Verified    131301
    Not Verified       124467
    Name: verification_status, dtype: int64
    
    application_type :
    
    INDIVIDUAL    394508
    JOINT            425
    DIRECT_PAY       286
    Name: application_type, dtype: int64
    
    initial_list_status :
    
    f    237346
    w    157873
    Name: initial_list_status, dtype: int64
    
    purpose :
    
    debt_consolidation    234169
    credit_card            82923
    home_improvement       23961
    other                  21059
    major_purchase          8756
    small_business          5656
    car                     4670
    medical                 4175
    moving                  2842
    vacation                2442
    house                   2197
    wedding                 1794
    renewable_energy         329
    educational              246
    Name: purpose, dtype: int64


all these can be converted to dummy variables


```python
df = pd.get_dummies(df, columns=['verification_status', 'application_type', 'initial_list_status', 'purpose'],\
               drop_first=True)
```


```python
df.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'home_ownership',
           'annual_inc', 'issue_d', 'loan_status', 'dti', 'earliest_cr_line',
           'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
           'mort_acc', 'pub_rec_bankruptcies', 'address', 'loan_repaid',
           'sub_grade_A2', 'sub_grade_A3', 'sub_grade_A4', 'sub_grade_A5',
           'sub_grade_B1', 'sub_grade_B2', 'sub_grade_B3', 'sub_grade_B4',
           'sub_grade_B5', 'sub_grade_C1', 'sub_grade_C2', 'sub_grade_C3',
           'sub_grade_C4', 'sub_grade_C5', 'sub_grade_D1', 'sub_grade_D2',
           'sub_grade_D3', 'sub_grade_D4', 'sub_grade_D5', 'sub_grade_E1',
           'sub_grade_E2', 'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5',
           'sub_grade_F1', 'sub_grade_F2', 'sub_grade_F3', 'sub_grade_F4',
           'sub_grade_F5', 'sub_grade_G1', 'sub_grade_G2', 'sub_grade_G3',
           'sub_grade_G4', 'sub_grade_G5', 'verification_status_Source Verified',
           'verification_status_Verified', 'application_type_INDIVIDUAL',
           'application_type_JOINT', 'initial_list_status_w',
           'purpose_credit_card', 'purpose_debt_consolidation',
           'purpose_educational', 'purpose_home_improvement', 'purpose_house',
           'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
           'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
           'purpose_vacation', 'purpose_wedding'],
          dtype='object')



home ownership


```python
df['home_ownership'].value_counts()
```




    MORTGAGE    198022
    RENT        159395
    OWN          37660
    OTHER          110
    NONE            29
    ANY              3
    Name: home_ownership, dtype: int64




```python
df['home_ownership'].replace({'NONE':'OTHER','ANY':'OTHER'}, inplace=True)
```


```python
df['home_ownership'].value_counts()
```




    MORTGAGE    198022
    RENT        159395
    OWN          37660
    OTHER          142
    Name: home_ownership, dtype: int64




```python
df = pd.get_dummies(df, columns=['home_ownership'], drop_first=True)
```


```python
df.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'issue_d',
           'loan_status', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'mort_acc',
           'pub_rec_bankruptcies', 'address', 'loan_repaid', 'sub_grade_A2',
           'sub_grade_A3', 'sub_grade_A4', 'sub_grade_A5', 'sub_grade_B1',
           'sub_grade_B2', 'sub_grade_B3', 'sub_grade_B4', 'sub_grade_B5',
           'sub_grade_C1', 'sub_grade_C2', 'sub_grade_C3', 'sub_grade_C4',
           'sub_grade_C5', 'sub_grade_D1', 'sub_grade_D2', 'sub_grade_D3',
           'sub_grade_D4', 'sub_grade_D5', 'sub_grade_E1', 'sub_grade_E2',
           'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5', 'sub_grade_F1',
           'sub_grade_F2', 'sub_grade_F3', 'sub_grade_F4', 'sub_grade_F5',
           'sub_grade_G1', 'sub_grade_G2', 'sub_grade_G3', 'sub_grade_G4',
           'sub_grade_G5', 'verification_status_Source Verified',
           'verification_status_Verified', 'application_type_INDIVIDUAL',
           'application_type_JOINT', 'initial_list_status_w',
           'purpose_credit_card', 'purpose_debt_consolidation',
           'purpose_educational', 'purpose_home_improvement', 'purpose_house',
           'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
           'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
           'purpose_vacation', 'purpose_wedding', 'home_ownership_OTHER',
           'home_ownership_OWN', 'home_ownership_RENT'],
          dtype='object')



**Address**


```python
df.loc[9,'address'].split(" ")[-1] # taking the zip code out of address
```




    '00813'




```python
df['zipcode'] = df['address'].apply(lambda x: x.split(" ")[-1])
```


```python
df.drop('address', axis=1, inplace=True)
```


```python
df['zipcode'].unique()
```




    array(['22690', '05113', '00813', '11650', '30723', '70466', '29597',
           '48052', '86630', '93700'], dtype=object)




```python
df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)
```

**'issue_d'**


```python
get_description('issue_d')
```

    The month which the loan was funded


**In theory we wouldn't ideally know if a loan was issued or not. So knowing the date at which the loan was issued will lead to information leakage. dropping this variable**


```python
df.drop('issue_d', axis=1, inplace=True)
```

**'earliest_cr_line'**


```python
get_description('earliest_cr_line')
```

    The month the borrower's earliest reported credit line was opened



```python
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x: x.split("-")[-1])  #taking only the year
```


```python
df.drop('earliest_cr_line', axis=1, inplace=True)
```


```python
df['earliest_cr_year'] = pd.to_numeric(df['earliest_cr_year'])
```


```python
df.select_dtypes('object').columns
```




    Index(['loan_status'], dtype='object')



**'loan_status' is the only non-numeric variable remaining**

<a id=5></a>

**Train Test Split**


```python
from sklearn.model_selection import train_test_split
```


```python
df.drop('loan_status', axis=1, inplace=True)
```


```python
X = df.drop('loan_repaid', axis=1).values
```


```python
y = df['loan_repaid'].values
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
```

**Scaling the data**


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
```


```python
X_train = scaler.fit_transform(X_train)
```


```python
X_test = scaler.transform(X_test) # we only transform the testing data
```


```python
X_test.min(), X_test.max()
```




    (-0.01282051282051282, 1.1866666666666668)




```python
X_train.min(), X_train.max()
```




    (0.0, 1.0)



**Creating the model**


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```


```python
model = Sequential()

model.add(Dense(units=78, activation='relu'))
model.add(Dropout(0.2)) # 2% of the neurons (selected randomly) in the previous layer will get turned off in every iteration.

model.add(Dense(units=39, activation='relu'))
model.add(Dropout(0.2)) 

model.add(Dense(units=19, activation='relu')) #three layers with 4 nodes each, and activation functions
model.add(Dropout(0.2)) 

model.add(Dense(units=1, activation='sigmoid')) # the layer that predicts price. it has only one node

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
from tensorflow.keras.callbacks import EarlyStopping
```


```python
early_stop = EarlyStopping(monitor='val_loss', mode='min',patience=5)
```


```python
model.fit(x=X_train, y=y_train, epochs=25, validation_data=(X_test,y_test), batch_size=256, callbacks=[early_stop])
```

    Epoch 1/25
    1236/1236 [==============================] - 2s 2ms/step - loss: 0.3030 - accuracy: 0.8764 - val_loss: 0.2660 - val_accuracy: 0.8869
    Epoch 2/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2659 - accuracy: 0.8882 - val_loss: 0.2630 - val_accuracy: 0.8869
    Epoch 3/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2635 - accuracy: 0.8885 - val_loss: 0.2624 - val_accuracy: 0.8869
    Epoch 4/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2620 - accuracy: 0.8885 - val_loss: 0.2622 - val_accuracy: 0.8869
    Epoch 5/25
    1236/1236 [==============================] - 2s 2ms/step - loss: 0.2612 - accuracy: 0.8884 - val_loss: 0.2621 - val_accuracy: 0.8870
    Epoch 6/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2604 - accuracy: 0.8885 - val_loss: 0.2621 - val_accuracy: 0.8871
    Epoch 7/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2599 - accuracy: 0.8884 - val_loss: 0.2615 - val_accuracy: 0.8870
    Epoch 8/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2597 - accuracy: 0.8885 - val_loss: 0.2612 - val_accuracy: 0.8872
    Epoch 9/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2593 - accuracy: 0.8884 - val_loss: 0.2622 - val_accuracy: 0.8869
    Epoch 10/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2590 - accuracy: 0.8884 - val_loss: 0.2615 - val_accuracy: 0.8871
    Epoch 11/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2587 - accuracy: 0.8885 - val_loss: 0.2613 - val_accuracy: 0.8869
    Epoch 12/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2584 - accuracy: 0.8886 - val_loss: 0.2616 - val_accuracy: 0.8870
    Epoch 13/25
    1236/1236 [==============================] - 2s 1ms/step - loss: 0.2582 - accuracy: 0.8888 - val_loss: 0.2614 - val_accuracy: 0.8872





    <tensorflow.python.keras.callbacks.History at 0x7f1ea4104590>




```python
losses = pd.DataFrame(model.history.history)
```


```python
losses.plot(figsize=(15,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1ea4104310>

<img src="/assets/img/loan-classifier/output_147_1.png" alt="output_147_1"  width="75%">


**The lines tell us that model was not overfit. It reductions in validation loss and training loss follow the same pattern. So does accuracy**


```python
print("Loss:", model.evaluate(X_test,y_test, verbose=0)[0])
print("Accuracy:", model.evaluate(X_test,y_test, verbose=0)[1])
```

    Loss: 0.26136863231658936
    Accuracy: 0.8872020840644836



```python
predictions = model.predict(X_test) > 0.5
```


```python
from sklearn.metrics import confusion_matrix, classification_report
```


```python
predictions
```




    array([[ True],
           [ True],
           [ True],
           ...,
           [ True],
           [ True],
           [False]])




```python
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.43      0.60     15658
               1       0.88      1.00      0.93     63386
    
        accuracy                           0.89     79044
       macro avg       0.94      0.72      0.77     79044
    weighted avg       0.90      0.89      0.87     79044
    



```python
print(confusion_matrix(y_test,predictions))
```

    [[ 6759  8899]
     [   17 63369]]


**predicting whether a customer (randomly selected) pays off a loan or not, and comparing with the actual data.**


```python
import random
```


```python
i = random.randint(0,len(df))

customer = df.drop('loan_repaid', axis=1).iloc[i]

customer = scaler.transform(customer.values.reshape(-1,78))

print("Customer number:", i)

print("Did the customer repay the loan or not??\nWhat the model predicts: ", model.predict(customer)>0.5)

print("And what what data says: ",df['loan_repaid'].iloc[i])
```

    Customer number: 104027
    Did the customer repay the loan or not??
    What the model predicts:  [[ True]]
    And what what data says:  1

