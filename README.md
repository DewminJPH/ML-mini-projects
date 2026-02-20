## House Price Predictor
  - This is a regression problem
  - House Price Dataset -> Data preprocessing -> Data Analysis -> Train Test split -> XGBoost Regressor -> Evaluation
  - **XGBoost Regressor model** is used
  - Heat map is used to visualize the correlation between the features
  
## Fake News Predictor
  - Textual data used here
  - This used a labeled data. (Binary classification project)
  - News Data -> Data Preprocessing -> Train Test Split -> **Logistic Regression model** -> Evaluate the model using the New data
  - Stemming used here
  - convert all the text into numbers using 'TfidfVectorizer()'

## Loan Status Predictor
  - this help to automate the loan approval process when considering the user given data. So output is eligibility of that person for getting the loan based on the given data
  - Data -> Data preporocessing -> Train Test Split -> Feed the training data using **SVM model** -> evaluation
  - This is a supervised machine learning model

## Wine Quality Predictor
  - Problem statement: considering the wine manufacturing company, and its want to creat a new wine brand with a desired chemical parameters like acidity, citric acid content, sugar content etc. 
  - Target is find the wine is good or bad, kind of a classification problem.(supervised learning)
  - wine dataset -> data analysis -> data preprocessing -> train test split -> Random Forest Model -> Evaluate the model
  - correlation checking: If the volatile acidity is high then the quality of the wine is less, if the volatile quality is less the quality of the wine is high. volatile acidity and the quality of the wine is inversly propotional
  - There are two types of correlations
      1. Positive corrrelation: 
          If two values are positively correlated, if one value increases and the other value also increases.
      2. Negative correlation
          If one value is increases but other one is decreases, those two values are negatively correlated each other.
  - **Random forest algorithm**
      Random forest is the ensemble model. That means it uses more than two or three models in combination for a prediction.
        multiple decision tree models

## Car Price Prediction
  Problem Statement
    predict the prices of used cars.There are several information such as car brand, year, sold price, present price and etc.
  
  - This is a regression problem
  - Car Data -> Data pre processing -> train test split -> **linear & lasso regression model** -> model evaluation
  
  **Alert**
  - The model is underperforming. There are few mistakes I have done. 
  1. I have dropped the "Brand Name" column. It is more important because the brand name can do a big affect to the selling price.
  2. I have used Label Encoding which means I assigned numbers to fuel and seller_type manually by using .replace(). That confuses the Linear Regression model because it think 0 < 1< 2 like that.
  3. Car prices are often "right-skewed"(many low-priced cars and a few very expensive luxury cars) Linear Regression assumes a normal distribution. To avoid that I have to apply log transformation to selling_price using np.log1p().

  **Solutions**
  - I switched to the Random forest algorithm. Because the linear regression tries to draw a straight line through data that naturally curves(like car depreciation). Random forest builds multiple "decision trees" to map these curves accurately
  - By extracting the Brand, the model recognizes that a "Toyota" and a "Maruti" have different baseline values, which was previously lost when I dropped the Brand "name" column

## Gold Price Prediction
  - Gold Price data -> Data Preprocessing -> Data Analysis -> Train Test Split -> Random Forest Regressor
  - metrics for evaluate the model

## Heart Disease Prediction
  - This will predict whether a person has heart disease or not
  - Heart Data -> Data Preprocessing -> Train Test Split -> Logistic Regression Model(since this is a binary classification problem) -> Model evaluation

## Credit Card Fraud Detection
  - Credit Card Data -> Data Preprocessing -> Data Analysis -> Train Test Split -> Logistic Regression Model -> Model Evaluation
  - This dataset is highly unbalanced dataset, that means the target variable has two classes but they are not equally distributed. one class has like 99% data. So the dataset is highly unbalanced.

  **Solution**
  - split the target classes into two new variables
  - then get a eqauivalent portion of dataset from the class who has the largest dataset. It is selected randomlhy
  - then merge that selected class rows and previous less class rows, and create a new dataset.
  - then that newdataset is balanced.


## Medical Insuarnce Cost Prediction
  - Insurace Cost Data -> Data analysis -> Data preprocessing -> Train Test Split -> Linear Regression model -> model evaluation
  - In this case target column is charges. so it is not classifcation problem. It's output gives continuous numbers. so this will use LinearRegression model
  - I have encoded the categorical data columns.

## Big Mart Sales Prediction
  - this will predict the sales that we are expect.
  - If we predict the sales revenue we can tell the company what are the brands, what are the products that sells most and other such kind of things. Then we can make decision based on this.
  - This is a regression based machine learning project.
  - Big Mart Sales Data -> Data preprocessing -> Data Analysis -> Train Test Split -> XGBoost Regressor model -> Model Evaluation
  - In this data set some columns has missing values. So we have to fill these values. That called as the *Data Imputation*.
  There are 2 methods we have done here.
  1. Using the mean value of that column
  2. Using the mode value of that column

  - Here we have used lambda function:   
      if we want to create a function in python we use "def label()". Then that function we can use multiple times.

      But if we want to use a function for once then we use the 'Lambda' function.
  - after plotting some features with graph, I have seen the Item_Visibility column is skewed. That means the values of that column is distributed to one side. 
  - difference between countplot and distplot provided by seaborn
    - countplot : pecialized bar chart for Categorical Data.It simply counts the number of occurrences of each unique value in a specific column
    - distplot  : used for Numerical Data. It shows the distribution of a continuous variable, helping to see the "shape" of the dataset. nowadays histplot used as distplot

## Customer Segmentation
  - In this project I will try to group customers based on their spending patterns, and purchase behavior.
  - In here I have used K-means clustering which is a unsupervised learning method.
  - Customer Data -> Data preprocessing -> Data Analysis -> Choose the Optimum Number of Cluster (WCSS) -> K-Means Clustering Algorithm -> Visualize the Clusters 
  - Here we should select a suitable number of the clusters which is very hard to select by manually looking at the dataset. So we can use WCSS for choose suitable number.
    - WCSS means Within Cluster Sum of Square.
    - Each cluster has own centroid and the datapoints should be closer to the corresponding to the centroid as much as possible. Thats how we are selecting the number of how many clusters should be there. 
    - So we need to find the WCSS values and generate the elbow graph then using it we can find the suitable k value.
![image alt](https://github.com/DewminJPH/ML-mini-projects/blob/66f26830b9bb91f9fe35dcd2a68af3471905981d/Customer%20Segmentation/elbow_graph.png)

  - After get the k number then we split the dataset into those clusters and give them a number for their corresponding cluster


## Parkinson's Disease Detection
  - This project is under the health care sector and this is useful for detect the patient who are pakinson's disease or not early. This is very valuable project.
  - Parkinson's disease is a progressive nervous system disorder that affects movement leading to shaking, stiffness and difficulty with walking, balance and coordination. Parkinson's symptoms usually begin gradually and get worse over time.
  - Since this project has only the two output, this problem is a classification problem.
  - Parkinson's Data -> Data Pre Processing -> Train Test Split -> Support Vector Machine Classifier -> Model Evaluation
  - I have satandaridize the dataset using StandardScaler
  - For model training I have used Support Vector Machine. But this method has two types. 
    1. SVC : Support Vector Classifier(I have used thisone for model training) -> use to classify
    2. SVR : Support Vector Regressor -> use to get a certain value

## Titanic Survival Prediction
  - Titanic incident is about the ship sinking due to an accident with the iceberg. So in this problem we predict the person who is survived or not survived. 
  - Its a classifcation problem.
  - Data -> Data Preprocessing -> Data Analysis -> Train Test Split -> Logistic Regression Model -> Model Evaluation
  - Logistic Regression model works well in binary classification problem
  - In the dataset there are 3 columns which has null values included. So I have to handle those missing values.
  - But among of those 3 there is one column, which most of values are null. So I decided to drop that column from the dataset.

## Calories Burnt Predictor
  - This predictor will predict how much calories the person will be burning when doing the exercise.
  - Data -> Data Pre Processing -> Data Analysisi -> Train Test Split -> Model Training(XGBoostRegressor) -> Model Evaluation
  - Here we use two dataset files and merge them into one. 
  - To evaluate the model performance, I have used the Means Absolute Error method.
  - If the mean absolute error is low as much as possible that means the model is good.
  
