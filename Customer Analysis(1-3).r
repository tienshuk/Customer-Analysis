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


# Import Data-------------------------------------------------------

data = read.csv('Customer-Churn.csv')
# Link to Kaggle Competition
# https://www.kaggle.com/blastchar/telco-customer-churn


# Quick view of data-------------------------------------------------

skim(data)
glimpse(data)


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

# Clustered Scatter plot ---------------------------------------------------

data %>%
  ggplot(aes(x = tenure, y = MonthlyCharges, color = Churn))+
  geom_point(alpha = 0.08)+ # Show where data is concerntrated.
  stat_ellipse()  +
  scale_color_manual(values=c("blue","red")) +
  labs( x = 'Customer Tenure', y = 'Monthly Charges')

# Rigde Plot --------------------------------------------------------------

New_data %>%
  select_if(is.numeric) %>% # select all the numeric and dummy variables.
  #na.omit() %>% 
  gather(variable, value, -c(Churn,gender)) %>% # transform data 
  ggplot(aes(y = as.factor(variable), # start building plot
             fill = as.factor(Churn), 
             x = percent_rank(value))) +
  scale_fill_manual(values=c( "#66CC99","#CC6666")) + 
  geom_density_ridges()+
  scale_x_continuous(expand = c(0,0))+
  guides(fill=guide_legend(title="Churn (1 - Yes / 0 - No)")) + 
  labs(x = 'Scaled Value', y = 'Customer Information')