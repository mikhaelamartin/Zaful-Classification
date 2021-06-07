# Why Do Consumers Cancel Their Subscriptions?

## Credit Card Company Customer Churn Classification 
Author: Mikhaela Martin

### Project Objectives:
1. Identify key patterns and factors that determine attrition rate.

2. Build a model that predicts whether a customer is going to churn using given dataset. 

### Dataset
We will be using a [credit card service company](https://www.kaggle.com/sakshigoyal7/credit-card-customers) dataset from Kaggle.  

### Bank Churners Table

- `Attrition_Flag`: Internal event (customer activity) variable - if the account is closed then 1 else 0
- `Customer_Age`: Demographic variable - Customer's Age in Years
- `Gender`: Demographic variable - M=Male, F=Female
- `Dependent_count`: Demographic variable - Number of dependents
- `Education_Level`: Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.)
- `Marital_Status`: Demographic variable - Married, Single, Divorced, Unknown
- `Income_Category`: Demographic variable - Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, > $120K, Unknown)
- `Card_Category`: Product Variable - Type of Card (Blue, Silver, Gold, Platinum)
- `Months_on_book`: Period of relationship with bank
- `Total_Relationship_Count`: Total no. of products held by the customer
- `Months_Inactive_12_mon`: No. of months inactive in the last 12 months
- `Contacts_Count_12_mon`: No. of Contacts in the last 12 months
- `Credit_Limit`: Credit Limit on the Credit Card
- `Total_Revolving_Bal`: Total Revolving Balance on the Credit Card
- `Avg_Open_To_Buy`: Open to Buy* Credit Line (Average of last 12 months)
- `Total_Amt_Chng_Q4_Q1`: Change in Transaction Amount (Q4 over Q1)
- `Total_Trans_Amt`: Total Transaction Amount (Last 12 months)
- `Total_Trans_Ct`: Total Transaction Count (Last 12 months)
- `Total_Ct_Chng_Q4_Q1`: Change in Transaction Count (Q4 over Q1) 
- `Avg_Utilization_Ratio`: Average Card Utilization Ratio***


*Open-to-buy: The difference between the credit limit assigned to a cardholder account and the present balance on the account.

***Average Card Utilization Ratio: Amount client owes divided by credit limit. (Total_Revolving_Bal / Credit_Limit)


### Why is understanding customers' propensity to churn so valued?
Companies want to maximize ther profits which takes both revenue and costs into account. There are three ways to increase revenue and we will go over why increasing retention rate is the most crucial. 

1. Enhance or improve current products. While it is important to drive innovation forward, creating new features or products can actually take a lot of time, creativity, resources, and people to produce good results.
2. Acquire new customers. Expanding your customer base seems like common sense, but this method can be hard for well-established companies to utilize. Also, identifying and advertising to targeted demographies can be costly when you have limited information.
3. Lowering churn. Identifying people in your company who are more prone to churning can be a lot easier because you have the data 

Acquiring new customers costs much more than retaining them. As stated, we will focus on how a company can identify whether a customer can churn. Once a model that can identify the types of customers who are likely to churn or provide a probability of people who are likely to churn at any given time, business solutions such as issuing a retention campaign or promotion can be put into place to target those most prone to churning.

### Methods
- Preparing Data: Random Imputation,

### Images

Gender: People who attrite are more likely to be female than male.

Income_Category: People who earn certain incomes are more likely to attrite than others.

Total_Relationship_Count: People who bought more products are less likely to attrite.

Months_Inactive_12_mon: As months inactive increases, so does the probability of attriting.

Contacts_Count_12_mon: Consumers who were in contact with more people in the credit card company are more likely to attrite.

`Total_Trans_Amt` : People who spend a lot at a time tend to not attrite.
`Total_Amt_Chng_Q4_Q1` : People who increasingly spend more in the fourth quarter tend to not attrite.
`Total_Trans_Ct` : People who make more transactions tend to not attrite.


## Further Analysis:
If I had more data about the specific time 


Probability they will churn in day X

Expected time to churn - regression

- To prevent churn: company based improvements, customer based marketing
- In churn prediction:
    - Be aware of class skew
    - Consider both customer attributes and customer networks
    - Interpretability, not just accuracy
