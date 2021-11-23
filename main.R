## AIM :
# To predict the sales price for each house based on house attributes and Location
# This problem is part of Kaggle competition
# House Prices - Advanced Regression Techniques (https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

rm(list = ls())
library(ggplot2)
library(moments)
library(car)

## Loading Train Data
train_df = read.csv("data/train.csv", sep=",")
test_df = read.csv("data/test.csv", sep=",")


## Observing Train Data
head(train_df)
View(train_df)

dim(train_df)
# Train Data has 79 features (excluding Id and SalePrice column) with 1460 observations

summary(train_df)
str(train_df)


## Data Cleaning
# Visualizing Null Values
total_na = colSums(is.na(train_df))
df_na = data.frame(
  col = colnames(train_df[which(total_na > 0)]), 
  count = total_na[which(total_na > 0)]*100/nrow(train_df)
)
barplot(df_na$count ~ df_na$col, col='blue', xlab='Columns', ylab='NA count %', 
        main="Missing values", ylim=c(0,100))
View(df_na)

# Null values in Alley represent no alley, hence we should add them in a new Level 'None'
train_df$Alley = replace(train_df$Alley, is.na(train_df$Alley), 'None')
test_df$Alley = replace(test_df$Alley, is.na(test_df$Alley), 'None')

# We can replace NA values of LotFrontage with mean Lot size (mean LotFrontage)
train_df$LotFrontage[is.na(train_df$LotFrontage)] = mean(train_df$LotFrontage, na.rm = TRUE)
test_df$LotFrontage[is.na(test_df$LotFrontage)] = mean(train_df$LotFrontage, na.rm = TRUE)

# If MasVnrType is Null, the type is None and MasVnrArea is 0
train_df$MasVnrType = replace(train_df$MasVnrType, is.na(train_df$MasVnrType), 'None')
train_df$MasVnrArea[is.na(train_df$MasVnrArea)] = 0
test_df$MasVnrType = replace(test_df$MasVnrType, is.na(test_df$MasVnrType), 'None')
test_df$MasVnrArea[is.na(test_df$MasVnrArea)] = 0

# Similarly for Basement, Garage NA values should represent Level 'None' and their corresponding attributes level 'None'
na_cols = c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'GarageType', 'GarageYrBlt', 'GarageFinish', 
                'GarageQual', 'GarageCond')

for (i in 1:length(na_cols)){
  train_df[na_cols[i]] = replace(train_df[na_cols[i]], is.na(train_df[na_cols[i]]), 'None')
  test_df[na_cols[i]] = replace(test_df[na_cols[i]], is.na(test_df[na_cols[i]]), 'None')
}

# For NA value in Electrical we replace it with most frequent label 'SBrkr'
table(train_df$Electrical)
train_df$Electrical[is.na(train_df$Electrical)] = 'SBrkr'
test_df$Electrical[is.na(test_df$Electrical)] = 'SBrkr'

# Columns FireplaceQu, PoolQC, Fence, MiscFeature have close to or more than 50% values null. 
# Even if we replace them with a single value these columns will contain a single dominant value. 
# Hence we choose to remove these columns from analysis.
train_df = train_df[, !(names(train_df) %in% c('FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'))]
test_df = test_df[, !(names(test_df) %in% c('FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'))]

# For the same reason we will remove the categorical variables with single dominant levels
cols.categorical = sapply(train_df, is.character)
del.cols = c()
for (i in 1:length(cols.categorical)){
  if (!cols.categorical[i]){
    next
  }
  max_freq = max(table(train_df[names(cols.categorical[i])]))
  if (max_freq > 0.9*nrow(train_df)){
    # Remove columns having more than 90% data as a same level
    del.cols = append(del.cols, names(cols.categorical[i]))
  }
}
train_df = train_df[, !(names(train_df) %in% del.cols)]
test_df = test_df[, !(names(test_df) %in% del.cols)]

# GarageYrBlt is incorrectly structured as string
train_df$GarageYrBlt = as.integer(train_df$GarageYrBlt)
test_df$GarageYrBlt = as.integer(test_df$GarageYrBlt)
train_df$GarageYrBlt[is.na(train_df$GarageYrBlt)] = mean(train_df$GarageYrBlt, na.rm = TRUE)
test_df$GarageYrBlt[is.na(test_df$GarageYrBlt)] = mean(train_df$GarageYrBlt, na.rm = TRUE)


## Explanatory Data Analysis
# As we are considering to fit Multivariate Linear Regression Model, 
# we need to consider the following assumptions : 
# 1. Errors are normally distributed with mean 0 and constant variance
# 2. SalePrice is linearly dependent on other variables
# 3. There is no Multicollinearity
# 4. Error terms are independent

# Distribution of Response variable SalePrice
par(mar=c(5.1,4.1,4.1,2.1))
par(mfrow=c(1,2))
hist(train_df$SalePrice, freq=FALSE, col='blue', 
     main="Distribution of SalePrice", 
     xlab="SalePrice($)")

summary(train_df$SalePrice)
sd(train_df$SalePrice)
skewness(train_df$SalePrice)
kurtosis(train_df$SalePrice)
# Average Sales Price of homes is $180,921 with standard deviation $79,442.5
# It is also positively skewed. It does not appear normally distributed.

qqnorm(y = train_df$SalePrice, pch = 1, frame=FALSE)
qqline(train_df$SalePrice, col = "steelblue", lwd = 2)

# As QQ plot and density curve shows data is skewed 
# let us consider log of SalePrice as response variable
SalePrice.log = log(train_df$SalePrice)
qqnorm(y = SalePrice.log, pch = 1, frame=FALSE)
qqline(SalePrice.log, col = "steelblue", lwd = 2)

hist(SalePrice.log, freq=FALSE, col='blue', 
     main="Distribution of ln(SalePrice)", 
     xlab="ln(SalePrice)")

summary(SalePrice.log)
sd(SalePrice.log)
skewness(SalePrice.log)
kurtosis(SalePrice.log)
# From QQ plot and values of skewness and kurtosis we can conclude 
# ln(SalePrice) follows normal distribution relatively closer than SalePrice
# Hence we will consider it as our response variable
train_df$SalePrice = SalePrice.log


# Seeing Relationship of columns against response variable
par(mar=c(4,2,1,2))
par(mfrow=c(4,4))
for (i in 2:(ncol(train_df)-1)){
  temp_plot.xname = colnames(train_df)[i]
  temp_plot.x = train_df[, temp_plot.xname]
  if (cols.categorical[temp_plot.xname] == TRUE){
    temp_plot.x = as.factor(temp_plot.x)
  }
  plot(x = temp_plot.x, 
       y = train_df$SalePrice,
       col='blue', xlab=temp_plot.xname,
       ylab='', ylim=c(10, 14))
}

# We can observe OverallQual, OverallCond, YearBuilt, YearRemodAdd, TotalBsmtSF, X1stFlrSF, X2ndFlrSF, GrLivArea, TotRmsAbvGrd, GarageCars, GarageArea, SaleCondition
# have good linear relation with SalePrice

# Correlation plot
library(corrplot)
par(mar=c(5.1,4.1,4.1,2.1))
par(mfrow=c(1,1))
cols.numeric = sapply(train_df, is.numeric)
cor.data = cor(train_df[,cols.numeric])
print(corrplot(cor.data, method="color"))
# There is considerable correlation between :
# - GarageArea and GarageCars
# - X1stFlrSF and TotalBsmtSF
# - TotRmsAbvGrnd and GrLivArea

# As Multicolliniarity increases variance of estimators we will eliminate some columns
# with high correlation
View(cor.data)
for (i in 1:(nrow(cor.data)-2)){
  for (j in (i+1):(ncol(cor.data)-1)){
    if (abs(cor.data[i, j]) > 0.7){
      print(paste(rownames(cor.data)[i], rownames(cor.data)[j]))
    }
  }
}
# The correlation also intuitively makes sense, hence we will remove columns 
# 'GarageCars', 'TotRmsAbvGrd', 'TotalBsmtSF' as we expect them to convey less useful information
train_df = train_df[, !(names(train_df) %in% c('GarageCars', 'TotRmsAbvGrd', 'TotalBsmtSF'))]
test_df = test_df[, !(names(test_df) %in% c('GarageCars', 'TotRmsAbvGrd', 'TotalBsmtSF'))]

# Creating Dummy Variables for Categorical Variables
library(fastDummies)
str(train_df)

len.train = nrow(train_df)
len.test = nrow(test_df)

data = rbind(train_df[, !(names(train_df) %in% c('SalePrice'))], test_df)

# Use Remove first dummy column to reduce multicollinearty due to dummy variables 
data = dummy_cols(data, remove_first_dummy = TRUE, ignore_na = TRUE)

# Remove categorical variables
cols.categorical = sapply(data, is.character)
del.cols = colnames(data[, cols.categorical])
data = data[, !(names(data) %in% del.cols)]

# Checking dummy variables
which(colSums(data) <= 1)
which(colSums(is.na(data)) > 0)
data.vars = colnames(data)

# Explanatory Variables
data.exp = data[1:len.train, ]
which(colSums(data.exp) == 0)
det(t(data.exp) %*% as.matrix(data.exp))
which(colSums(unique(data.exp) == data.exp) < 1460)

cor.data = cor(data.exp)
det(cor.data)
# Determinant of correlation matrix is close to 0  
corrplot(cor.data, method="color")
# This indicates high multicollinearity in data due to added dummy variables
View(cor.data)
for (i in 1:(nrow(cor.data)-2)){
  for (j in (i+1):(ncol(cor.data)-1)){
    if (abs(cor.data[i, j]) > 0.6){
      print(paste(rownames(cor.data)[i], rownames(cor.data)[j]))
    }
  }
}
# The relationship between the pairs are intuitively meaningful.
# We can treat them by removing following columns :
del.cols = c('HouseStyle_2Story', 'MSZoning_FV', 'MSZoning_RM', 'RoofStyle_Hip', 
             'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 
             'Exterior2nd_MetalSd', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng',
             'MasVnrType_None', 'ExterQual_TA','ExterCond_TA', 'BsmtQual_None',
             'BsmtCond_None', 'BsmtFinType1_None', 'BsmtExposure_None', 'KitchenQual_TA',
             'GarageType_None', 'GarageFinish_None', 'GarageYrBlt', 'BsmtFinType2_Unf',
             'HouseStyle_1Story', 'Exterior2nd_Plywood', 'Exterior2nd_Stucco',
             'Foundation_PConc', 'BsmtFinType2_None', 'BsmtQual_TA', 'GarageType_Detchd',
             'GarageQual_None', 'SaleType_WD', 'BsmtFullBath', 'BsmtFinType1_Unf',
             'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'Exterior2nd_Brk Cmn', 
             'Condition1_Norm', 'Exterior2nd_BrkFace', 'KitchenQual_Gd', 'BsmtCond_TA',
             'SaleCondition_Normal')
data = data[, !(names(data) %in% del.cols)]
data.exp = data[1:len.train, ]

train_df = cbind(data.exp, SalePrice = SalePrice.log)
test_df = data[(len.train+1):nrow(data), ]


## Dealing with Outliers
# We will use Mahalanobis distance to find high leverage points
data.cov = cov(data.exp)
data.center = colMeans(data.exp)
data.md = mahalanobis(data.exp, data.center, data.cov)
data.md

data.md_pvalues = pchisq(data.md, df = ncol(data.exp)-1, lower.tail=FALSE)
md_comp = data.frame(Id = data.exp$Id, md = data.md, pvalue = data.md_pvalues)
View(md_comp)

# Considering test size 0.001 we will reject the hypothesis that xi is not a high leverage point if p < 0.001
md_comp[which(md_comp$pvalue < 0.001),]
nrow(md_comp[which(md_comp$pvalue < 0.001),])
# There are higher number of such values in data
# These leverage points will not affect regression line much unless the residual is also large 


## Fitting Linear Model
# Fitting model initially to check residuals
model1 = lm(SalePrice ~ . , train_df)
model1.summary = summary(model1)
model1.coeff = model1.summary$coefficients[,1]
model1.summary
# From the current model, we get adjusted R^2 = 0.8904 on training data.
# p value is significantly small implying that chosen variables do explain behavior of SalePrice

model1.res = residuals(model1)

par(mfrow=c(1,2))
hist(model1.res, freq=FALSE, col='blue', 
     main="Distribution of Residuals") 

summary(model1.res)
sd(model1.res)
skewness(model1.res)
kurtosis(model1.res)
qqnorm(y = model1.res, pch = 1, frame=FALSE)
qqline(model1.res, col = "steelblue", lwd = 2)
# Clearly the residuals are not normal currently.

par(mfrow=c(2,2))
plot(model1)
# In the residual vs fitted values, residual values are scattered around the predicted values
# This is expected from model having constant error variances and error mean zero. 
# (This is also confirmed by looking at summary of residuals)

# From the QQ plot and Residual vs fitted plot, we can see that Points with 
# Id 1299, 524, 496 are error outliers.
# Out of these points 1299 and 524 were also found to be high leverage points.
# If a point is both, it has very high impact on regression line,
# hence we will remove these two points.
train_df = train_df[-c(1299, 524),]
data.exp = train_df[, !(names(train_df) %in% c('SalePrice'))]

# Observing Variance Inflation Factor
model1.vif = as.data.frame(vif(model1))
# Variance Inflation factor multiplies the variance of coefficients in a multicollinear model.
# This increases variance and reduces our trust on corresponding p values. 
# Hence we will remove all columns having VIF > 5.
del.cols = rownames(model1.vif)[which(model1.vif > 5)]
train_df = train_df[, !(names(train_df) %in% del.cols)]
test_df = test_df[, !(names(test_df) %in% del.cols)]


# Fitting Linear model 2
model2 = lm(SalePrice ~ . , train_df)
model2.summary = summary(model2)
model2.coeff = model2.summary$coefficients[,1]
model2.summary
# From the second model, we get the adjusted R^2 = 0.8673 on training data.

model2.res = residuals(model2)

par(mfrow=c(1,2))
hist(model2.res, freq=FALSE, col='blue', 
     main="Distribution of Residuals")

summary(model2.res)
sd(model2.res)
skewness(model2.res)
kurtosis(model2.res)
qqnorm(y = model2.res, pch = 1, frame=FALSE)
qqline(model2.res, col = "steelblue", lwd = 2)
# The residuals behave more closer to normal here.

par(mfrow=c(2,2))
plot(model2)


# From the QQ plot and Residual vs fitted plot, we can see that Points with 
# Id 496, 31, 633, 376, 347 are error outliers and 251, 326, 595, 1011, 1187, 1369 are leverage points (by Cook’s Distance plot).
# Out of these points 31, 376, 347, 595, 1187 were also found to be high leverage points.
# Hence we will remove these points.
train_df = train_df[!(rownames(train_df) %in% c(31, 376, 347, 496, 633, 595, 1187)),]
data.exp = train_df[, !(names(train_df) %in% c('SalePrice'))]

# Now that we have reduced multicollinearity, we can consider only the variables 
# with significantly low p values of having zero regression coefficient (considering p < 0.1)
model2.sig_vars = names(which(model2.summary$coefficients[,4] < 0.1))
model2.sig_vars = model2.sig_vars[model2.sig_vars != '(Intercept)']


# Fitting Model on significant variables
train_df = train_df[, (names(train_df) %in% c(model2.sig_vars,'SalePrice'))]
test_df = test_df[, (names(test_df) %in% model2.sig_vars)]
data.exp = train_df[, !(names(train_df) %in% c('SalePrice'))]


model3 = lm(SalePrice ~ . , train_df)
model3.summary = summary(model3)
model3.coeff = model3.summary$coefficients[,1]
model3.summary
# From the third model, we get the adjusted R^2 = 0.8744 on training data.

model3.res = residuals(model3)

par(mfrow=c(1,2))
hist(model3.res, freq=FALSE, col='blue', 
     main="Distribution of Residuals")

summary(model3.res)
sd(model3.res)
skewness(model3.res)
kurtosis(model3.res)
qqnorm(y = model3.res, pch = 1, frame=FALSE)
qqline(model3.res, col = "steelblue", lwd = 2)
# The residuals behave more closer to normal here.

par(mfrow=c(2,2))
plot(model3)

# Plotting bi's vs predicted values
model3.hat = hatvalues(model3)
model3.bi = (model3.res^2)/(1 - model3.hat)
par(mfrow=c(1,2))
plot(y = model3.bi, x = fitted(model3), xlab="Fitted Values", ylab="bi", 
     main="bi vs predicted values")
# As the plot is not increasing, it signifies error variance is constant 
# but there are still some outliers in data

plot(model3.res, main='Residual Plot')

# As errors are now roughly normal we can eliminate error outliers using Z test 
# with size 1%
lines(x=1:nrow(train_df), 
      y=rep((mean(model3.res) + 2.57*sd(model3.res)), nrow(train_df)),
      col="blue")
lines(x=1:nrow(train_df), 
      y=rep((mean(model3.res) - 2.57*sd(model3.res)), nrow(train_df)),
      col="blue")

error_outliers = abs(model3.res - mean(model3.res))/sd(model3.res) > 2.57

# Are they leverage points as well?
data.cov = cov(data.exp)
data.center = colMeans(data.exp)
data.md = mahalanobis(data.exp, data.center, data.cov)
data.md

data.md_pvalues = pchisq(data.md, df = ncol(data.exp)-1, lower.tail=FALSE)

# Considering test size 0.001 we will reject the hypothesis that xi is not a high leverage point if p < 0.001
md_comp = data.frame(md = data.md, 
                     pvalue = data.md_pvalues,
                     leverage = data.md_pvalues < 0.001,
                     error = error_outliers)
View(md_comp)
sum(md_comp$leverage)
sum(md_comp$error)

md_comp = cbind(md_comp, 
                influential = (md_comp$leverage == TRUE & md_comp$error == TRUE))
View(md_comp)
sum(md_comp$influential)
# There are only three points that are influential. 
# Hence we will choose to eliminate all error outliers
error_indices = as.integer(names(error_outliers[which(error_outliers == TRUE)]))

data.exp = data.exp[!(rownames(data.exp) %in% error_indices), ]
train_df = train_df[!(rownames(train_df) %in% error_indices), ]


# Fitting Model
model4 = lm(SalePrice ~ . , train_df)
model4.summary = summary(model4)
model4.coeff = model4.summary$coefficients[,1]
model4.summary
# From the fourth model, we get the adjusted R^2 = 0.8901 on training data.

model4.res = residuals(model4)

par(mfrow=c(1,2))
hist(model4.res, freq=FALSE, col='blue', 
     main="Distribution of Residuals")

summary(model4.res)
sd(model4.res)
skewness(model4.res)
kurtosis(model4.res)
qqnorm(y = model4.res, pch = 1, frame=FALSE)
qqline(model4.res, col = "steelblue", lwd = 2)
# The residuals behave very close to normal here.

par(mfrow=c(2,2))
plot(model4)

par(mfrow=c(1,2))
plot(model4.res, main="Residual Plot")

# Plotting bi's vs predicted values
model4.hat = hatvalues(model4)
model4.bi = (model4.res^2)/(1 - model4.hat)
plot(y = model4.bi, x = fitted(model4), xlab="Fitted Values", ylab="bi", 
     main="bi vs predicted values")
# There are some leverage points but the error is minimal hence they will not have substantial effect on regression line

# Testing Autocorrelation
par(mfrow=c(1,1))
model4.fit =fitted(model4)
plot(y = model4.res, x = model4.fit, main="Residual Plot")
lines(lowess(y = model4.res, x = model4.fit), col="red")

# Durbin Watson Test
dw_num = 0
for (i in 2:length(model4.res)){
  dw_num = dw_num + (model4.res[i] - model4.res[i-1])^2 
}
dw = dw_num / sum(model4.res^2)
print(dw)
durbinWatsonTest(model4.res)
# The test value is 1.979246. As the test statistic value is close to 2, there is no autocorrelation


## Prediction
# We will use model4 to predict the SalePrice for test data

# Fixing NA is test values
test_Id = read.csv("data/test.csv", sep=",")$Id
total_na = colSums(is.na(test_df))
df_na = data.frame(
  col = colnames(test_df[which(total_na > 0)]), 
  count = total_na[which(total_na > 0)]
)
barplot(df_na$count ~ df_na$col, col='blue', xlab='Columns', ylab='NA count')
View(df_na)

test_df$GarageArea[is.na(test_df$GarageArea)] = mean(train_df$GarageArea, na.rm = TRUE)
na_cols = c('MSZoning_RL', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_Stone')
for (i in 1:length(na_cols)){
  test_df[na_cols[i]][is.na(test_df[na_cols[i]])] = 0
}


# Predicting
predictions = predict(model4, test_df)
sum(is.na(predictions))

# as we were predicting ln(SalePrice)
SalePrice.prediction = exp(predictions)

# Preparing Submission
results = data.frame(Id = test_Id, SalePrice = SalePrice.prediction)
View(results)

write.csv(results, './submission.csv', quote = FALSE, row.names=FALSE)

# This submission scored 0.16274 and ranked 3104 on 2021-11-21

