# House Price prediction using common features

### Problem Statement :
To predict the sales price for each house based on house attributes and Location
This problem is part of Kaggle competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### Structure of Data :
Train Data has 79 features (excluding Id and SalePrice column) with 1460 observations out of which 43 columns are Categorical and 36 are Numerical.

### Aim :
To predict the values for 'SalePrice' in test data. 

### Intention :
- To explore Linear Regression and it's assumptions. 
- To explore Problems encountered while preparing a larger data set for such model
- To understand various concepts covered in Regression Techniques course prior to working on project

### Data Cleaning :
- Visualizing Missing Values
![Visualizing Missing Values](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_missing_values?raw=true)

- Null values in 'Alley' represent no alley, Null values in Basement columns represent no basement and that in Garage columns represent no garage. Hence, we should replace them with a new Level 'None'. 
- We can replace NA values of 'LotFrontage' with mean Lot size (mean of linear feet of street connected to property).
- If 'MasVnrType'(Masonry veneer type) is Null, the type is None and 'MasVnrArea' (Masonry veneer area in square feet) is 0.
- For NA values in 'Electrical' (Electrical system) we can replace them with the most frequent label 'SBrkr'.
- Columns 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature' have close to or more than 50% values null. Even if we replace them with a single value these columns will contain a single dominant value. Hence we choose to remove these columns.
- For the same reason we will remove the any categorical variables with a single dominant level. (columns with same level in 90%+ observations)
- 'GarageYrBlt' is incorrectly structured as string.

### Explanatory Data Analysis
As we are considering to fit Multivariate Linear Regression Model, we need to consider the following assumptions : 
1. Errors are normally distributed with mean 0 and constant variance
2. SalePrice is linearly dependent on other variables
3. There is no Multicollinearity
4. Error terms are independent

Distribution of Response variable 'SalePrice' -
![Distribution of Response variable](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_saleprice?raw=true)

Average Sales Price of homes is USD 180,921 with standard deviation USD 79,442.5. It is also positively skewed. It does not appear normally distributed.

Considering log of SalePrice as response variable - 
![Distribution of log of SalePrice](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_lnsaleprice?raw=true)

From QQ plot and values of skewness (0.12) and kurtosis (3.8) we can conclude ln(SalePrice) follows normal distribution relatively closer than 'SalePrice'. Hence we will consider it as our response variable.

Seeing Relationship of columns against response variable we observe that 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'X1stFlrSF', 'X2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SaleCondition' have good linear relation with ln(SalePrice).


### Preparing Data for Linear Regression
- Correlation plot of numerical variables - 
  ![Correlation Plot](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_corrplot?raw=true)
  
  There is considerable correlation between :
  - GarageArea and GarageCars
  - X1stFlrSF and TotalBsmtSF
  - TotRmsAbvGrnd and GrLivArea
  
  As Multicollinearity increases variance of estimators, we will eliminate some columns with high correlation. The correlation also intuitively makes sense, hence we will remove columns 'GarageCars', 'TotRmsAbvGrd', 'TotalBsmtSF' as we expect them to convey less useful information.

- Creating Dummy Variables for Categorical Variables - 
  We use remove_first_dummy as otherwise there will be perfect multicollinearity in data
  `data = dummy_cols(data, remove_first_dummy = TRUE, ignore_na = TRUE)`
  
- Finding Leverage Points -
  We will use Mahalanobis distance to find high leverage points which is a measure of distance between the i'th observation and center of data. Considering test size 0.001 we will reject the hypothesis that xi is not a high leverage point if p < 0.001. There are 251 such values. These leverage points will not affect regression line much unless the error is also large.

### Fitting Linear Regression Model
- Fitting model initially to check residuals
  ```
  Residual standard error: 0.1323 on 1303 degrees of freedom
  Multiple R-squared:  0.9021,	Adjusted R-squared:  0.8904 
  F-statistic: 76.96 on 156 and 1303 DF,  p-value: < 2.2e-16
  ```
  p value is significantly small implying that chosen variables do explain behavior of SalePrice.
  ![Distribution of Residuals](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model1_res?raw=true)
  Clearly the residuals are not normal currently.
  
  ![Model1 Plots](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model1?raw=true)
  In the residual vs fitted values, residual values are scattered around the predicted values. This is expected from model having constant error variances and error mean zero. 
  We will now observing Variance Inflation Factor (>=1) which multiplies the variance of coefficients in a multicollinear model. This increases variance and reduces our trust on corresponding p values. Hence we will remove all columns having VIF > 5.
  
- Fitting model with these columns
  ```
  Residual standard error: 0.1456 on 1332 degrees of freedom
  Multiple R-squared:  0.8787,	Adjusted R-squared:  0.8673 
  F-statistic: 77.16 on 125 and 1332 DF,  p-value: < 2.2e-16
  ```
  ![Distribution of Residuals](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model2_res?raw=true)
  
  ![Model2 Plots](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model2?raw=true)
  
  Now that we have reduced multicollinearity, we can consider only the variables with significantly low p values of having zero regression coefficient (considering p < 0.1)

- Fitting Model on significant variables
  ```
  Residual standard error: 0.14 on 1407 degrees of freedom
  Multiple R-squared:  0.8781,	Adjusted R-squared:  0.8744 
  F-statistic: 235.7 on 43 and 1407 DF,  p-value: < 2.2e-16
  ```
  p value is significantly small implying that chosen variables do explain behavior of SalePrice.
  ![Distribution of Residuals](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model3_res?raw=true)
  The residuals behave more closer to normal here.
  
  ![Model3 Plots](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model3?raw=true)
  
  **Plotting bi's vs predicted values**
  where bi = (ei)^2/(1 - hi)
  ![bi Plots](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model3_bi?raw=true)
  As the plot is not increasing, it signifies error variance is constant but there are still some outliers in data. 
  As errors are now roughly normal we can eliminate error outliers using Z test with size 1%. 
  
- Fitting Final model after removing outliers
  ```
  Residual standard error: 0.1261 on 1380 degrees of freedom
  Multiple R-squared:  0.8934,	Adjusted R-squared:  0.8901 
  F-statistic:   269 on 43 and 1380 DF,  p-value: < 2.2e-16
  ```
  p value is significantly small implying that chosen variables do explain behavior of SalePrice.
  ![Distribution of Residuals](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model4_res?raw=true)
  The residuals behave very close to normal here.
  
  ![Model4 Plots](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model4?raw=true)
  
  **Plotting bi's vs predicted values**
  where bi = (ei)^2/(1 - hi)
  ![bi Plots](https://github.com/Hemant-Banke/House-Prices/blob/main/img/plot_model4_bi?raw=true)
  
  There are some leverage points but the error is minimal hence they will not have substantial effect on regression line.




