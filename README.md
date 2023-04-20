# Climate Change Forecasting and Prediction
---
![Unsplash Climate Image](https://www.noaa.gov/sites/default/files/styles/landscape_width_1275/public/2022-03/PHOTO-Climate-Collage-Diagonal-Design-NOAA-Communications-NO-NOAA-Logo.jpg)


## Intro
Our world's climate is constantly changing, and understanding these changes is critical to mitigating the potential harm to our planet and its inhabitants. In this project, we utilized various machine learning models to forecast changes in global temperature, both on land and sea, as well as predict sea level rise. By examining data from sources such as NOAA, datahub.io, kaggle, climate.gov, and others, we aimed to better understand how global emissions, glacier melt, and other factors could impact climate change.


## Overview
Our project began by compiling data on temperature change, historical sea level change, annual global emissions, and glacier melt over time to input into our models. We utilized a variety of machine learning models, including linear regression, random forest, ridge regression, and neural network models, to forecast and predict these changes. Our goal with these predictions was to better understand the potential impact of increased sea level and temperature, such as population displacement in coastal areas and the increase in severe climate-related natural disasters.


## Limitations
One significant limitation of our project was the data itself. To develop a fully-formed model, we would require much more data, which would likely have an impact on our predictions. Additionally, our data only extended to 2014, so any new changes made regarding electric and green energy are not accounted for in our emissions numbers.


## Libraries Used
Our project utilized a variety of Python libraries, including:
- Pandas
- Matplotlib
- Numpy
- Seaborn
- TensorFlow
- SciKit-Learn
- Plotly

## Conclusion
All of our machine learning models predicted a continuation of the trend of increased temperatures and higher sea levels. This trend will most likely result in more severe weather and climate disasters, including hurricane damage, famine from drought, and other severe weather associated with those factors. By utilizing machine learning models, we can better understand the impact of climate change and take steps towards mitigating potential harm.


## Climate Change - Ridge Regression

- Global temperature has increased significantly since the beginning of the 1900's.
- Global emissions show a mostly gradual incline from the 1700s to the 1900s. Then, an explosive increase from a specific group of countries.

<table>
  <tr>
    <td><img src="https://github.com/damian-robinson/multi-model-analysis/blob/748ec2e7863684caabae8436b997bce6770f59c7/data/temperature_change.png"></td>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/emissions_change.png"></td>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/sea_level_change.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/sunspots_change.png"></td>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/x_train_predictions.png"></td>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/x_test_predictions.png"></td>
  </tr>
</table>

---


## Climate Change - Neural Network

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/Annual_Climate_Disasters.png"></td>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/Sea_Level_Change_Prediction.png"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/Sea_Level_Comparison_Graph.jpg"></td>
    <td><img src="https://raw.githubusercontent.com/damian-robinson/multi-model-analysis/main/data/heatmap.png"></td>
  </tr>
</table>
  


  
  
  
## Predicting Weather Temperature Change Using Machine Learning Models
Problem Introduction
The problem tackled is to predicting the average global land and ocean temperature using over 100 years of past weather data.
The avialable historical global temperatures averages comprises;
-global maximum temperatures,
-global minimum temperatures,
-global land and ocean temperatures.
 we used supervised ML; regression machine learning.This is possible since we have both:
- the features
-the target that we want to predict,
secondly is is regression as our target is continuous(as opposed to discrete classes in classification).

## ML Workflow
Before we jump right into programming, we should outline exactly what we want to do. The following steps are the basis of my machine learning workflow now that we have our problem and model in mind:
State the question and determine the required data (completed)
Acquire the data
Identify and correct missing data points/anomalies
Prepare the data for the machine learning model by cleaning/wrangling
Establish a baseline model
Train the model on the training data
Make predictions on the test data
Compare predictions to the known test set targets and calculate performance metrics

1.Data Acquisition
We retrieved temperature data from the Berkeley Earth Climate Change: Earth Surface Temperature Dataset found on Kaggle.com.
Dataset link:https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
After importing some important libraries and modules, the code below loads in the CSV data which I store into a variable we can use later:
The Data had the following columns:
dt: starts in 1750 for average land temperature and 1850 for max and min land temperatures and global ocean and land temperatures
LandAverageTemperature: global average land temperature in celsius
LandAverageTemperatureUncertainty: the 95% confidence interval around the average
LandMaxTemperature: global average maximum land temperature in celsius
LandMaxTemperatureUncertainty: the 95% confidence interval around the maximum land temperature
LandMinTemperature: global average minimum land temperature in celsius
LandMinTemperatureUncertainty: the 95% confidence interval around the minimum land temperature
LandAndOceanAverageTemperature: global average land and ocean temperature in celsius
LandAndOceanAverageTemperatureUncertainty: the 95% confidence interval around the global average land and ocean temperature
Identify Anomalies/ Missing Data
To identify anomalies, we used the function info(),.isnull and .sum method on our DataFrame.
Data Preparation
we had to  drop columns that hold high cardinality-with values that are uncommon or unique.
Finally, the last step in our data wrangling function was to convert the dt(Date) column to a DateTime object. After which we will create subsequent columns for the month and year, eventually dropping the dt and Month columns.
We also called the dropna(), just in case there are any other missing values in our dataset:
working with Dataframe
Quick Correlation Visualization
we  plotted a correlation matrix, just to get an understanding of how related each column is to each other:
Global Temps Correlation Matrix Plot
the matrix showed that the data was HIGHLY correlated to one another.
Separating our Target From Our Features
 -The target, also known as Y, is the value we want to predict, in this case, the actual land and ocean average temperature
-the features are all the columns (minus our target) the model uses to make a prediction:
Creating Target Vector and Features Matrix
Train-Test Split
Now we are on the final step of the data preparation part of our ML workflow: splitting data into training and testing sets.
During training, we let the model ‘see’ the answers, in this case, the actual temperature, so it can learn how to predict the temperature from the features. As we know, there is a relationship between all the features and the target value, and the model’s job is to learn this relationship during training. Then, when it comes time to evaluate the model, we ask it to make predictions on a testing set where it only has access to the features (not the target)!
Generally, when training a regression model, we randomly split the data into training and testing sets to get a representation of all data points.
For example, if we trained the model on the first nine months of the year and then used the final three months for prediction, our algorithm would not perform well because it has not seen any data from those last three months.
Make sense?
The following code splits the data sets:
Train/Test Split Creation
We looked at the shape of all the data to make sure we did everything correctly. We expect the training(X_train) features number of columns to match the testing (X_test) feature number of columns and the number of rows to match for the respective training and testing features and target:
The shape of each training and testing set
Got rid of missing values and unneeded columns
Split data into features and target
Split data into training and testing sets
These steps may seem tedious at first, but once you get the basic ML workflow, it will be generally the same for any machine learning problem. It’s all about taking human-readable data and putting it into a form that can be understood by a machine learning model.
Establish Baseline Mean Absolute Error
Before we can make and evaluate predictions, we need to establish a baseline, a sensible measure that we hope to beat with our model. If our model cannot improve upon the baseline, then it will be a failure and we should try a different model or admit that machine learning is not right for our problem.
The baseline prediction for our case will be the yearly average temperature. In other words, our baseline is the error we would get if we simply predicted the average temperature for our target dataset (Y_train)
In order to find out the MAE, very easily, we can import the mean_absolute_error method from the sci-kit learn library which will calculate it for us:
Baseline Mean Absolute Error

We now have our goal! If we can’t beat an average error of 2 degrees, then we need to rethink our approach.
Train Model
we trained the model using scikit-learn.
we employed two different models;
1.a Linear Regression Model
2.a Random Forest Regressor Model.
Linear Regression Model
Linear regression is a statistical approach that models the relationship between input features and output. Our goal here is to predict the value of the output based on the input features.
Random Forest Regressor Model
A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique commonly known as bagging.
The basic idea behind bagging is to combine multiple decision trees in determining the final output rather than relying on individual decision trees.
Random Forest has multiple decision trees as base learning models. We randomly performed row sampling and feature sampling from the dataset forming sample datasets for every model:

Random Forest Regressor Pipeline
n_estimators represents the number of trees in the random forest.
max depth represents the depth of each tree in the forest. The deeper the tree, the more splits it has and it captures more information about the data.
n_jobs refers to the number of cores the regressor will use. -1 means it will use all cores available to run the regressor.
SelectKBest just scores the features using an internal function. In this case, we chose to score all the features.
Make Predictions on the Test Set

Linear Regression MAE
Our average temperature prediction estimate is off by 0.14 degrees in our Linear Regression MAE and 0.13 for our Random Forest MAE. That is almost a 1-degree average improvement over the baseline of 1.12 degrees.this is nearly 92% better than the baseline.
Determine Performance Metrics

To put our predictions in perspective, we can calculate an accuracy using the mean average percentage error subtracted from 100 %.
Linear Regression Test/Train Accuracy:
Random Forest Regressor Train/Test Accuracy:
By looking at the error metric values we got, we can say that our model performs optimally and is able to give accurate predictions, given a new set of records(y_pred).
Our model has learned how to predict the average temperature for the next year with over 90% accuracy in both our models.
  
  
  
  
  
  
  
  
  
  
  
</table>
