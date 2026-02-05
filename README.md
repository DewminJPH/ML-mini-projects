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
    