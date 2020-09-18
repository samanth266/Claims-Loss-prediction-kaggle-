Samanth changes import pandas as pd #importing pandas library and dataframe
#linear regression technique, simple linear estimator based on dependant and independant variables
from sklearn.linear_model import LinearRegression
#importing data sets from kaggle and loading the csv's
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train.head(5)
df_train.shape
#printing the first 10 columns and last 20 columns into python console to check with pandas dataframe
print('First 10 columns: {0} \n Last 20 columns: {1}'.format(list(df_train.columns[:10]), list(df_train.columns[-10:])))
#pandas has (pd.isnull) function to check if there is any null values present in the given dataset
#we can aslo eliminate the null values with pandas and numpy to avoid Overfitting of data.
pd.isnull(df_train).values.any()
#we can see that there are 116 categorical columns and 14 continous columns, categorical are masked to hide the data
# continous are numerical data in which we convert categorical into numeric to work with my regression model

features = [x for x in df_train.columns if x not in ['id','loss']]
#seperating categorical and numerical values
cat_features = [x for x in df_train.select_dtypes(include=['object']).columns if x not in ['id','loss']]
num_features = [x for x in df_train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]
#encoding categorical values into numeric to work with the linear regression
#encode cat features
df = pd.concat((df_train[features], df_test[features])).reset_index(drop=True)
for c in range(len(cat_features)):
    df[cat_features[c]] = df[cat_features[c]].astype('category').cat.codes
#preparing data for traininng.   
n_train = df_train.shape[0] 
#splitting the data,X_train,y_train
X_train, y_train = df.iloc[:n_train,:], df_train['loss']
X_test = df.iloc[n_train:,:]

#simple linear estimator, fits the data between id and loss plot which estimates the predicted values according to the linearity.
lr = LinearRegression() 
#fitting the data with X and y train parameters.
lr.fit(X_train, y_train)
#submission
submission = pd.read_csv("sample_submission.csv")
# predicting the loss by giving test set.
submission.iloc[:, 1] = lr.predict(X_test)
submission.to_csv('loss_lr.csv', index=None)   
    
