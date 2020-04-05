# Customer Segmentation and Campaign Response Prediction
*Tianyi Wang*
*2020 Apr*

This is a Capstone project for Udacity Data Science Nanodegree parterning with [Arvato Financial Services](https://finance.arvato.com/en-us/]). In this project, demographics data for customers of a mail-order sales company in Germany is provided along with the demographics information for the general population. We'll use unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, we'll build a machine model to predict which individuals are most likely to convert into becoming customers for the company. 

## Data

The raw data is proprietary so I won't share it in this repo. We mainly have 2 datasets which contains the demographic features for the general German population and the customers seperately. The customers data is like a subset of the general population data with 3 extra columns: CUSTOMER_GROUP, ONLINE_PURCHASE and PRODUCT_GROUP.

* Genereal German population data: 891,221 rows, 366 columns
* Customers data: 191,652 rows, 369 columns

The 366 features cover various aspects that describe a customer, including age, customers journey typology, financial typology, bank transaction activities, customer personalities, shares of different car brands in the customer's neighborhood. Interestingly, most features are built around the cars the customer owns or the cars the customer's neighbors own. 

An example of the features:
![features](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Customer_Segmentation_and_Campaign_Response_Prediction/master/screenshots/features.png)

Most of the features are ordinal categorical variables like this:
![ordinal](https://raw.githubusercontent.com/tianyiwangnova/2020_project__Customer_Segmentation_and_Campaign_Response_Prediction/master/screenshots/ordinal.png)

In our notebook `01 Data cleaning and feature engineering` we will explore and understand the demographic data and perform data cleaning process on the general German population data. We will create a python class for the data cleaning process so that it can be used for later and we will save the fitted `Imputer` and `StandardScaler` so that they can all be used directly on the customers data. (We will compare customers data with the general population later, so we want to keep the data transformation consistent).

## Understand who will be our customers --- Population segmentation!

Unlike other segmentation problems, in this project we will perform a `KMeans` clustering on the general population and then fit the `KMeans` model on the customers data.  We would like to see which clusters are more likely to be customers by compaing the clusters distribution of the general population and that of the customers.

Since we have many features, we will perform a `PCA` transformation on the demographic data to reduce the dimensions before we do the `KMeans` clustering. The process can be found in notebook `02 PCA`. We chose to keep the first 100 components, which explain 70% of the variance. By looking at the top and bottom 3 features that make up the components, we tried to understand what each component means:

* First component: Low mobility, 1-2 family houses in the PLZ8, low share of AUDI within the PLZ8.
* Second component: High online affinity and high recent transaction activity
* Third component: High share of BMW & Mercedes Benz, high share of upper middle class cars and upper class cars (BMW5er, BMW7er etc.), low share of small and very small cars (Ford Fiesta, Ford Ka etc.) in the PLZ8, low share of cars with max speed between 140 and 210 km/h within the PLZ8, low numbers of cars with 5 seats in the PLZ8
* Forth component: Low "financial typology: be prepared", older age, highly likely to be money saver or investor

We extracted the top 2 features (that can be found explanations) for the top 20 components. They are considered as "important features" and when we do clustering later, we will look at these features to understand the clusters.











