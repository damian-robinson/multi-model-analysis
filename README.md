# Climate Change Forecasting and Prediction

![Unsplash Climate Image](https://www.noaa.gov/sites/default/files/styles/landscape_width_1275/public/2022-03/PHOTO-Climate-Collage-Diagonal-Design-NOAA-Communications-NO-NOAA-Logo.jpg)


## Overview
Our world's climate is constantly changing, and understanding these changes is critical to mitigating the potential harm to our planet and its inhabitants. In this project, we utilized various machine learning models to forecast changes in global temperature, both on land and sea, as well as predict sea level rise. By examining data from sources such as NOAA, datahub.io, kaggle, climate.gov, and others, we aimed to better understand how global emissions, glacier melt, and other factors could impact climate change.

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

---

## Takeaway
All of our machine learning models predicted a continuation of the trend of increased temperatures and higher sea levels. This trend will most likely result in more severe weather and climate disasters, including hurricane damage, famine from drought, and other severe weather associated with those factors. By utilizing machine learning models, we can better understand the impact of climate change and take steps towards mitigating potential harm.


### Using Linear & Ridge Regression
We used an initial linear regression model for our baseline and moved to ridge regression for further exploration. Tweaking the alpha produced only slightly useful changes in the output. Overall, we quickly found the ridge model didn't perform at a high level for the particular data set, but the predictions still line up with the others in that the upward trend remains.

- Global temperature has increased significantly since the beginning of the 1900's.
- Global emissions show a mostly gradual incline from the 1700s to the 1900s. Then, an explosive increase from specific groups.

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


### Using Neural Networks
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

### Using MAE & Random Forest Regressor
We retrieved temperature data from the Berkeley Earth Climate Change: Earth Surface Temperature Dataset found on Kaggle.com.

The Data had the following columns:
- dt: starts in 1750 for average land temperature and 1850 for max and min land temperatures and global ocean and land temperatures
- LandAverageTemperature: global average land temperature in celsius
- LandAverageTemperatureUncertainty: the 95% confidence interval around the average
- LandMaxTemperature: global average maximum land temperature in celsius
- LandMaxTemperatureUncertainty: the 95% confidence interval around the maximum land temperature
- LandMinTemperature: global average minimum land temperature in celsius
- LandMinTemperatureUncertainty: the 95% confidence interval around the minimum land temperature
- LandAndOceanAverageTemperature: global average land and ocean temperature in celsius
- LandAndOceanAverageTemperatureUncertainty: the 95% confidence interval around the global average land and ocean temperature


Our average temperature prediction estimate is off by 0.14 degrees in our Linear Regression MAE and 0.13 for our Random Forest MAE. That is almost a 1-degree average improvement over the baseline of 1.12 degrees. This is nearly 92% better than the baseline. Our model has learned how to predict the average temperature for the next year with over 90% accuracy in both our models.
  

---


### References
- https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
- https://www.kaggle.com/datasets/jarredpriester/global-annual-mean-temperature
- https://www.kaggle.com/datasets/robervalt/sunspots
- https://www.climate.gov/maps-data/dataset/global-mean-sea-level-graph
  
  
</table>
