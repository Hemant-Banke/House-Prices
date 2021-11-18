## AIM :
# To predict the sales price for each house based on house attributes and Location
rm(list = ls())
library(ggplot2)

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
  count = total_na[which(total_na > 0)]
)
barplot(df_na$count ~ df_na$col)
View(df_na)

# Null values in Alley represent no alley, hence we should add them in a new Level 'None'
train_df$Alley = replace(train_df$Alley, is.na(train_df$Alley), 'None')

# We can replace NA values of LotFrontage with mean Lot size (mean LotFrontage)
train_df$LotFrontage[is.na(train_df$LotFrontage)] = mean(train_df$LotFrontage, na.rm = TRUE)

# If MasVnrType is Null, the type is None and MasVnrArea is 0
train_df$MasVnrType = replace(train_df$MasVnrType, is.na(train_df$MasVnrType), 'None')
train_df$MasVnrArea[is.na(train_df$MasVnrArea)] = 0

# Similarly for Basement, Garage NA values should represent Level 'None' and their corresponding attributes level 'None'
na_cols = c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'GarageType', 'GarageYrBlt', 'GarageFinish', 
                'GarageQual', 'GarageCond')

for (i in 1:length(na_cols)){
  train_df[na_cols[i]] = replace(train_df[na_cols[i]], is.na(train_df[na_cols[i]]), 'None')
}

# For NA value in Electrical we replace it with most frequent label 'SBrkr'
table(train_df$Electrical)
train_df$Electrical[is.na(train_df$Electrical)] = 'SBrkr'

# Columns FireplaceQu, PoolQC, Fence, MiscFeature have close to or more than 50% values null. 
# Even if we replace them with a single value these columns will contain a single dominant value. 
# Hence we choose to remove these columns from analysis.
train_df = train_df[, !(names(train_df) %in% c('FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'))]

# For the same reason we will remove the categorical variables with dominant levels
cols.categorical = sapply(train_df, is.character)
del.cols = c()
for (i in length(cols.categorical)){
  max_freq = max(table(train_df[cols.categorical[i]]))
  if (max_freq > nrow(train_df)){
    del.cols = append(del.cols, cols.categorical[i])
  }
}
train_df = train_df[, !(names(train_df) %in% del.cols)]




# To Predict Count
plot(df$temp, df$count, col='red')
plot(df$atemp, df$count, col='red')

count.third = quantile(df$count, 3)
ggplot(df, aes(x=temp, y=count)) + geom_point(aes(color=temp)) + geom_line(aes(y = mean(count)))

ggplot(df, aes(x=datetime, y=count)) + geom_point(aes(color=season))

# corrplot
num.cols = sapply(df, is.numeric)
cor.data = cor(df[,num.cols])
print(corrplot(cor.data, method="color"))


ggplot(df, aes(x=factor(season), y=count)) + geom_boxplot(aes(color=factor(season)))


# Adding Datetime
getHour = function(x){
  return (format(as.POSIXct(x), "%H"))
}
df$hour = as.factor(sapply(df$datetime, getHour))

ggplot(df[df$workingday == 1,], aes(x=hour, y=count)) + geom_point(aes(color=temp), position = position_jitter()) + scale_color_gradient(low="green", high="red")
ggplot(df[df$workingday == 0,], aes(x=hour, y=count)) + geom_point(aes(color=temp), position = position_jitter()) + scale_color_gradient(low="green", high="red")


# Split
set.seed(100)

# Seasonal Split
s1 = subset(df, season == 1)
s2 = subset(df, season == 2)
s3 = subset(df, season == 3)
s4 = subset(df, season == 4)

sample1 = sample.split(s1$count, SplitRatio = 0.7)
sample2 = sample.split(s2$count, SplitRatio = 0.7)
sample3 = sample.split(s3$count, SplitRatio = 0.7)
sample4 = sample.split(s4$count, SplitRatio = 0.7)

train = rbind(subset(s1, sample1==TRUE), subset(s2, sample2==TRUE), subset(s3, sample3==TRUE), subset(s4, sample4==TRUE))
test = rbind(subset(s1, sample1==FALSE), subset(s2, sample2==FALSE), subset(s3, sample3==FALSE), subset(s4, sample4==FALSE))
View(train)

# Model
model = lm(count ~ season + holiday + workingday + weather + temp + atemp + humidity + windspeed, train)
summary(model)

res = residuals(model)
class(res)
res = as.data.frame(res)
ggplot(res, aes(res)) + geom_histogram(fill="blue", alpha=0.5, bins=1)

count.predictions = predict(model, test)
results = cbind(count.predictions, test$count)
colnames(results) = c('prediction', 'actual')
results = as.data.frame(results)


to_zero = function(x){
  if (x < 0){
    return(0);
  }
  else{
    return(x);
  }
}

results$prediction = sapply(results$prediction, to_zero)

head(results)

mse = mean((results$prediction - results$actual)^2)
mse

sse = sum((results$prediction - results$actual)^2)
sst = sum((mean(df$count) - results$actual)^2)
R2 = 1 - sse/sst
R2



# Backward Selection
cols = colnames(df)[-c(1, 12, 13)]
cols

while(TRUE){
  model = lm(as.formula(paste("count ~ ", paste(cols, collapse = '+'))), train)
  model.pval = summary(model)$coefficients[,4]
  model.maxp = unname(which(model.pval == max(model.pval))) - 1
  maxp = max(model.pval)
  
  print(paste(cols, collapse = '+'))
  if (maxp > 0.05){
    cols = cols[-c(model.maxp)]  
  }
  else{
    break
  }
}

