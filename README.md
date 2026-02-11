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
