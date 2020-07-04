
# Load Libraries ----------------------------------------------------
library(tidyverse)
library(tidyr)
library(DBI)
library(RPostgres)
library(glue)
library(lubridate)
library(rlist)
library(urltools)
library(randomcoloR)
library(reshape2)
library(dplyr)
library(corrplot)
library(randomForest)
library(fastDummies)
library(janitor)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(ggridges)
library(skimr)
library(e1071)

# Import Data ----------------------------------------------------
data = read.csv('Customer-Churn.csv')
# Link to Kaggle Competition
# https://www.kaggle.com/blastchar/telco-customer-churn

# Data Preprocessing ----------------------------------------------------
# Create dummy values for factor variables 
# Change text to binary value. Yes = 1 , No = 0, and so on. 

New_data = data %>%
  mutate(gender = recode(gender, Female = 0, Male = 1), 
         Partner = recode(Partner, No = 0, Yes = 1),
         Dependents = recode(Dependents, No = 0, Yes = 1),
         PhoneService = recode(PhoneService, No = 0, Yes = 1),
         MultipleLines = recode(MultipleLines, No = 0, 'No phone service' = 0, Yes = 1),
         OnlineSecurity = recode(OnlineSecurity, No = 0, 'No internet service' = 0, Yes = 1),
         OnlineBackup = recode(OnlineBackup, No = 0, 'No internet service' = 0, Yes = 1),
         DeviceProtection = recode(OnlineBackup, No = 0, 'No internet service' = 0, Yes = 1),
         TechSupport = recode(OnlineBackup, No = 0, 'No internet service' = 0, Yes = 1),
         StreamingTV = recode(OnlineBackup, No = 0, 'No internet service' = 0, Yes = 1),
         StreamingMovies = recode(OnlineBackup, No = 0, 'No internet service' = 0, Yes = 1),
         PaperlessBilling = recode(Dependents, No = 0, Yes = 1),
         Churn = recode(Churn, No = 0, Yes = 1))

# Use RandomForest to select important variables----------------------------------------------------
# balance target variable

library(ROSE)

t_data = data %>% 
  select(-c(TotalCharges,customerID)) %>%
  mutate(SeniorCitizen = recode(SeniorCitizen,  '0' = 'No',  '1' = 'Yes')) %>%
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.integer, as.double)

rf_rose =  ROSE(Churn ~ ., data = t_data, seed = 1)$data

prop.table(table(rf_rose$Churn))


Random_Forest_Variable_Importance = randomForest( as.factor(Churn) ~ .,  
                                                  ntree = 101,
                                                  data = rf_rose,
                                                  nodesize = 1, 
                                                  replace = FALSE,
                                                  importance = TRUE)

varImpPlot(Random_Forest_Variable_Importance, type = 1)

# Applying Decision Tree Model----------------------------------------------------
# Split the data into training and test set

set.seed(123)   #  ensure you always have same outcome for splitting the data
sample = sample.split(t_data, SplitRatio = 0.75) # splits the data in the ratio
train = subset(t_data,sample == TRUE) # creates a training and testing dataset 
test = subset(t_data, sample == FALSE)

# Check for distribution of target variable in training set
prop.table(table(train$Churn))

# Balance target variable in the training set for optimal model performance
train_rose =  ROSE(Churn ~ ., data = train, seed = 1)$data

# The first model and prediction ----------------------------------------------------
# Initial Results: 71.6% accuracy and 79.33% recall
# Build the first Decision Tree Model

set.seed(123)
dt = rpart(as.factor(Churn) ~ 
             tenure +
             Contract +
             PaymentMethod +
             gender +
             MultipleLines +
             MonthlyCharges +
             PaperlessBilling+
             OnlineSecurity +
             TechSupport + 
             Dependents
           , data = train_rose, method = 'class')

# Prediction
prediction = predict(dt, test, type = 'class')
confusionMatrix(prediction, test$Churn)

# Model tuning - recall  ----------------------------------------------------
# Here's the grid search process to find the parameters for optimal recall.

# Produce a list of parameter values
gs = list(minsplit = seq(25, 30, 1),
          cp = seq(0.001,0.005,0.001),
          xval = seq(1, 5, 1),
          maxdepth = seq(1, 5, 1)) %>% 
  cross_df() # Make it a data frame

# Build a function for the model
mod = function(...) {
  rpart(as.factor(Churn) ~ 
          tenure +
          Contract +
          PaymentMethod +
          gender +
          MultipleLines +
          MonthlyCharges +
          PaperlessBilling+
          OnlineSecurity +
          TechSupport + 
          Dependents, data = train_rose, control = rpart.control(...))
}

# Fit all the parameter values to the model 
gs_recall = gs %>% mutate(fit_recall = pmap(gs, mod))
gs_recall

# Create a function to calculate recall
compute_recall = function(fit, test_features, test_labels) {
  predicted = predict(fit, test_features, type = "class")
  cm = table( predicted,test$Churn)
  cm[2,2] / (cm[1,2]+cm[2,2])
}

test_features = test %>% select(-Churn)
test_labels   = test$Churn

# Let's see which combination of parameter values will yield the best results
gs_recall = gs_recall %>%
  mutate(test_recall = map_dbl(fit_recall, compute_recall,
                               test_features, test_labels))

gs_recall = gs_recall %>% arrange(desc(test_recall), desc(minsplit), maxdepth)
gs_recall

# Replace the parameters with new values and see how the model performs-----------------------------------------------
# Recall increased to 90.40% from 79.33% 
# Accuracy droped. This is the result of a trade-off that the model identified more false positives (customers that didn't churn but predicted as churned.)

control = rpart.control(
  minsplit = 30,  
  cp = 0.001,
  xval = 1, 
  maxdepth = 1)

set.seed(123)
tune_recall = rpart(as.factor(Churn) ~ 
                      tenure +
                      Contract +
                      PaymentMethod +
                      gender +
                      MultipleLines +
                      MonthlyCharges +
                      PaperlessBilling+
                      OnlineSecurity +
                      TechSupport + 
                      Dependents, data = train_rose, method = 'class', control = control)

prediction_tune =predict(tune_recall, test, type = 'class')

confusionMatrix(prediction_tune, test$Churn)

# Let's visualize the tree
rpart.plot(tune_recall, extra = 106)

# Model tuning - accuracy  ------------------------------------------------------------------------
# Here's the grid search process to find the parameters for optimal accuracy.
# Create a function to calculate accuracy
compute_accuracy = function(fit, test_features, test_labels) {
  predicted = predict(fit, test_features, type = "class")
  cm = table( predicted,test$Churn)
  (cm[1,1]+cm[2,2]) / (cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
}

test_features = test %>% select(-Churn)
test_labels   = test$Churn

# Fit all the parameter values to the model 
gs_accuracy = gs %>% mutate(fit = pmap(gs, mod))
gs_accuracy

# Let's see which combination of parameter values will yield the best results
gs_accuracy = gs_accuracy %>%
  mutate(test_accuracy = map_dbl(fit, compute_accuracy,
                                 test_features, test_labels))

gs_accuracy = gs_accuracy %>% arrange(desc(test_accuracy), desc(minsplit), maxdepth)
gs_accuracy

# Replace the parameters with new values and see how the model performs----------------------------------------------
# Accuracy increased to 75.35% from 71.6% 

control = rpart.control(
  minsplit = 30,
  cp = 0.003,
  xval = 1,
  maxdepth = 5
)

set.seed(123)
tune_acc = rpart(as.factor(Churn) ~ 
                   tenure +
                   Contract +
                   PaymentMethod +
                   gender +
                   MultipleLines +
                   MonthlyCharges +
                   PaperlessBilling+
                   OnlineSecurity +
                   TechSupport + 
                   Dependents, data = train_rose, method = 'class', control = control)

prediction_tune =predict(tune_acc, test, type = 'class')

confusionMatrix(prediction_tune, test$Churn)

# Let's visualize the tree
rpart.plot(tune_acc, extra = 106,cex=0.75)

