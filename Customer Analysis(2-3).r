# Load Libraries-------------------------------------------------------
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


# Import Data -------------------------------------------------------
data = read.csv('Customer-Churn.csv')
# Link to Kaggle Competition
# https://www.kaggle.com/blastchar/telco-customer-churn



# Data Preprocessing-------------------------------------------------------
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

# Use RandomForest to select important variables---------------------------
# create date set for RandomForest Model

t_data = data %>% 
  select(-c(TotalCharges,customerID)) %>%
  mutate(SeniorCitizen = recode(SeniorCitizen,  '0' = 'No',  '1' = 'Yes')) %>%
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.integer, as.double)


# A quick check to see if the outcome variable 'Churn' is unbalanced.
# The ratio of non-churn to churn is about 3:1
# This imbalance will introduce bias in the Random Forest model

prop.table(table(t_data$Churn)) # 0.75:0.25

# Create a data set with balanced outcome variable

library(ROSE)

rf_rose =  ROSE(Churn ~ ., data = t_data, seed = 1)$data

prop.table(table(rf_rose$Churn)) # 0.5:0.5

# Pass the new data set through the model
Random_Forest_Variable_Importance <- randomForest( as.factor(Churn) ~ .,  
                                                   ntree = 101,
                                                   data = rf_rose,
                                                   nodesize = 1, 
                                                   replace = FALSE,
                                                   importance = TRUE)

# Visualize the outcome
varImpPlot(Random_Forest_Variable_Importance, type = 1)
