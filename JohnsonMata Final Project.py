"""
Kevin Johnson Mata
ISE 535
Final project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Setting file paths for the data. Source: https://data.world/data-society/student-alcohol-consumption
studentMatcsv = "/Users/kevinljm/Dropbox/NCSU Year 2/ISE 535/Final project/data-society-student-alcohol-consumption/student-mat.csv"
studentPorcsv = "/Users/kevinljm/Dropbox/NCSU Year 2/ISE 535/Final project/data-society-student-alcohol-consumption/student-por.csv"


#Creating Pandas dataframes from each file
studentMat = pd.read_csv(studentMatcsv)
studentPor = pd.read_csv(studentPorcsv)


#Seeing how many students are in both classes, should be 382 according to data source
studentConcat = pd.merge(studentMat, studentPor,how="inner" ,on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])


#Merging both dataframes and removing the duplicate students. 1044-382=662
studentAll = pd.concat([studentMat, studentPor]).drop_duplicates(subset=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])


#Data cleaning for graphs
studentAll["AG"] = (studentAll["G1"]+studentAll["G2"]+studentAll["G3"])//3
studentAll["WeekAlc"] = studentAll["Dalc"]+studentAll["Walc"]


#Heat Map to see possible correlation between numerical features
plt.figure(figsize=(14,14))
sns.heatmap(studentAll.drop(["G1","G2","G3","Dalc","Walc"],axis=1).corr(), annot=True, cmap="cividis")
    #There is very little correlation between most of the features (except parents' education, 
    #going out/free time, going out/weekly consumption, a negative corr between past failures/final grade
    #and interestingly, a negative corr between traveltime/mom's education)
plt.show()


#Plotting distribution of grades
sns.set_theme(style="darkgrid",palette="Paired")
sns.distplot(studentAll["AG"])
plt.xlabel("Average grade")
plt.ylabel("Frequency")
plt.title("Average Grade Distribution")
plt.show()


#Plotting average grade vs. past failures
sns.catplot(x="failures", y="AG",data = studentAll)
plt.title("Relationship between past failures and average grade")
plt.ylabel("Average grade grade")
plt.xlabel("Past class failures")
plt.show()


#Plotting parents' education
sns.barplot(x="Fedu",y="Medu",data=studentAll)
plt.title("Relationship between Father's and Mother's education")
plt.ylabel("Mother's education level")
plt.xlabel("Father's education level")
plt.show()


#Plotting travel time and mom's education since there was a negative correlation
sns.countplot(x="traveltime", hue="Medu", data=studentAll)
plt.title("Mother's education level vs. Travel time to school")
plt.xlabel("Travel time to school")
plt.show()


#Plotting going out and Weekly Alcohol Consumption
sns.set_theme(style="darkgrid",palette="Paired")
sns.barplot(x="goout", y="WeekAlc", data=studentAll)
plt.title("Relationship between Alcohol Consumption and Going Out")
plt.ylabel("Weekly Alcohol Consumption")
plt.xlabel("Frequency of going out with friends")
plt.show()


#Plotting Grades by Alcohol Consumption
sns.set_theme(style="darkgrid",palette="Paired")
sns.barplot(x="WeekAlc", y="AG", data=studentAll)
plt.xlabel("Weekly Alcohol Consumption")
plt.ylabel("Average Grade")
plt.title("Alcohol Consumption vs. Grades")
    #slightly negative correlation
plt.show()


#Violin Plot Weekly Alcohol Consumption by sex
sns.set_theme(style="darkgrid",palette="Paired")
sns.violinplot(y = "WeekAlc", x = "sex",data = studentAll)
plt.ylabel("Weekly Alcohol Consumption")
plt.xlabel("Sex")
plt.title("Weekly Alcohol Consumption by sex")
    #Men drink more
plt.show()


#Plotting Alcohol Consumption by Age
alcByAge = studentAll.groupby("age", as_index=False).WeekAlc.mean()
sns.barplot(x="age",y="WeekAlc",data= studentAll)
plt.xlabel("Age")
plt.ylabel("Weekly Alcohol Consumption")
plt.title("Average Alcohol Consumption by Age")
    #Alcohol consumption increases by age, as expected. Huge jump at 22
plt.show()



#Preparing data for analysis by converting string features into dummy variables 
studentAll = pd.get_dummies(studentAll,columns = ["sex","internet","school","address","famsize","Pstatus","schoolsup","famsup","paid","activities","nursery","higher","romantic","Mjob","Fjob","reason","guardian"], prefix=["Sex","Internet","School","Address","Famsize","Pstatus","Schoolsup","Famsup","Paid","Activities","Nursery","Higher","Romantic","Mjob","Fjob","Reason","Guardian"],drop_first=True)
studentAll = studentAll.reset_index() #Done to avoid infinity/nan error with certain sklearn methods


#Defining x and y for easy calling later
x = studentAll.drop(["G1","G2","G3","AG","index"],axis=1)
y = studentAll["AG"]


#Looking at summary statistics for each feature
statsDF = x.describe() #no need to scale data since the magnitudes are not that different


#Splitting data into training and test samples
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=55)


#Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn import metrics

linearModel = LinearRegression() 
linearModel.fit(xTrain, yTrain) #Fitting the model
linearY_Pred = linearModel.predict(xTest) #Predicting y values on test set
linearMSE = metrics.mean_squared_error(yTest, linearY_Pred) #Computing Mean Squared Error


#Fitting Lasso Regression Model
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=55) #Setting Cross-Validation options
lassoModel = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv) #Setting range of lambda values (called alphas here) for the model to try
lassoModel.fit(xTrain,yTrain) #Fitting the model
lassoY_Pred = lassoModel.predict(xTest) #Predicting y values on test set
lassoMSE = metrics.mean_squared_error(yTest, lassoY_Pred) #Computing MSE


#Fitting Ridge Regression Model
from sklearn.linear_model import RidgeCV

ridgeModel = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv) #Setting range of lambda values (called alphas here) for the model to try
ridgeModel.fit(xTrain,yTrain) #Fitting the model
ridgeY_Pred = ridgeModel.predict(xTest) #Predicting y values on test set
ridgeMSE = metrics.mean_squared_error(yTest, ridgeY_Pred) #Computing MSE


#Finding best K value for KNN algorithm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

params = {'n_neighbors':np.arange(1,20)} #range of K's for it to loop through
knn = KNeighborsRegressor()
kSearch = GridSearchCV(knn, params, cv=5) #Creating model selection object
kSearch.fit(xTrain,yTrain) #Trying each value of K
kSearch.best_params_ #K=18 is the best


#K-Nearest Neighbors model
knnModel = KNeighborsRegressor(n_neighbors = 18)
knnModel.fit(xTrain, yTrain)  #Fitting the model
knnY_Pred=knnModel.predict(xTest) #Predicting y on test set
knnMSE = metrics.mean_squared_error(yTest,knnY_Pred) #Computing MSE


#Decision Tree model
from sklearn.tree import DecisionTreeRegressor

dtModel = DecisionTreeRegressor()
dtModel.fit(xTrain, yTrain) #Fitting the model
dtY_Pred = dtModel.predict(xTest) #Predicting y values on test set
dtMSE = metrics.mean_squared_error(yTest,dtY_Pred) #Computing MSE


#Fitting Random Forest model
from sklearn.ensemble import RandomForestRegressor
rfModel = RandomForestRegressor(random_state = 99)
rfModel.fit(xTrain,yTrain) #Fitting the model
rfY_Pred = rfModel.predict(xTest) #Predicting y values on test set
rfMSE = metrics.mean_squared_error(yTest,rfY_Pred) #Computing MSE


#Comparing the accuracy of each model
models = ["Decision Tree","KNN","Lasso","Linear Regression","Random Forest","Ridge"]
mses = [dtMSE,knnMSE,lassoMSE,linearMSE,rfMSE,ridgeMSE]
modelComparison = pd.DataFrame({"Model":models,"MSE":mses}).set_index("Model")
print("\n",modelComparison)


#Feature Selection to see what the most important features are and if prediction accuracy improves
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=LinearRegression(),scoring='neg_mean_squared_error') #creating Recursive feature elimination object that uses MSE 
rfecv.fit(xTrain,yTrain)


#Finding the best features
columns = xTrain.columns
bestFeatures = columns[rfecv.support_] #selected features
print("\nThe most important features are:", *bestFeatures)
#Using this method, Alcohol consumption does not affect academic performance since it is not on this list


#Defining new x and splitting into new training and test samples
x = studentAll[bestFeatures]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33,random_state = 55)


#Fitting Lasso model with new subset
cv = RepeatedKFold(n_splits=10, n_repeats=3)
lassoModel = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv)
lassoModel.fit(xTrain,yTrain)
lassoY_Pred = lassoModel.predict(xTest)
newLassoMSE = metrics.mean_squared_error(yTest, lassoY_Pred)

print("\nLasso MSE using all features:",lassoMSE,"\nLasso MSE with feature selection:",newLassoMSE)
#Performance was not improved after feature selection














