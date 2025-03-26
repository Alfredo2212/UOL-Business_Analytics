#King's County dataset: https://www.openml.org/search?type=data&status=active&id=44989 
#Credit default dataset: https://www.openml.org/search?type=data&status=active&id=43454 
library(caret) 
library(glmnet)
library(leaps)
library(MASS)
library(corrplot)
library(dplyr)
library(ggplot2)
library(car)
library(tidyr)
library(xgboost)
library(ROCR)
library(cluster)
library(patchwork)
library(leaflet)
library(leaflet.extras)
setwd("/Users/alfredo/Desktop/SIM GradDip/ML/ML Coursework Dataset/")
#import data set Kings County turn off credit risk
rawdata <- read.csv("/Users/alfredo/Desktop/SIM GradDip/ML/ML Coursework Dataset/kings_county.csv")

#import data set Credit Risk turn off kings county
#rawdata <- read.csv("/Users/alfredo/Desktop/SIM GradDip/ML/ML Coursework Dataset/credit_risk.csv")
#---------------------------GENERAL - DATA CLEANING & VALIDATION-----------------------------------------
# Data set Cleaning
nrow(rawdata)
check_missing <- is.na(rawdata)
check_question <- rawdata == "?"
check_empty <- rawdata == " "
checkall <- check_missing | check_question | check_empty 
cleaned_1 <- rawdata[!rowSums(checkall) > 0, ]

# Remove dupes
cleaned_2 <- cleaned_1[!duplicated(cleaned_1), ]
rawdata <- cleaned_2
nrow(rawdata)
#---------------------------------KC - DATA PREPROCESSING---------------------------------------------
# Check for inconsistencies of data vs all of the features using Mahalanobis distance
# Ensuring data doesnt give inconsistency ( too big gap of each other )
mahalanobis_dist <- mahalanobis(rawdata, colMeans(rawdata), cov(rawdata))

# Find rows with a Mahalanobis distance greater than a threshold (e.g., > 3)
threshold <- qchisq(0.99, df = ncol(rawdata))  # 99% threshold
inconsistency_rows_mahalanobis <- which(mahalanobis_dist > threshold)

#output the data file for checking
inconsistency_data <- rawdata[inconsistency_rows_mahalanobis, ]

#replace the afterprocessed dataset  & Save the file
kings_county <- rawdata[-inconsistency_rows_mahalanobis, ]
write.csv(kings_county, "kings_county_clean.csv", row.names = FALSE)

# removing zero variability feature
zero_variance_cols <- which(apply(kings_county, 2, var) == 0)

# Remove columns with zero variance
kings_county <- kings_county[, -zero_variance_cols]

# Removing the column of Grading
 kings_county <- kings_county %>% select(-kings_grade)

# After Removing Outliers
nrow(kings_county)
#-------------------------------------KC - CORRELATION MATRIX -----------------------------------------
# Compute the correlation matrix
#correlation_matrix <- cor(kings_county, use = "complete.obs", method = "pearson")

# Create only response variabel ( housing price ) correlation barchart
#kings_price_correlation <-  correlation_matrix["kings_price", ]
#barplot(kings_price_correlation, 
#        main = "Correlation of housing price", 
#        col="yellow",
#        las = 2,
#        cex.names = 0.6)

# Create a correlation plot
#corrplot(correlation_matrix, 
#         method = "circle",     # Visualization method (can also use "square", "number", etc.)
#         type = "upper",        # Show only the upper triangle
#         order = "hclust",      # Order correlations by hierarchical clustering
#         tl.col = "black",      # Set text label color
#         tl.srt = 45,           # Rotate text labels to 45 degrees
#         tl.cex = 0.4,          # text size
#         addCoef.col = "blue",  # Display correlation coefficients on the plot
#         diag = FALSE,          # Hide diagonal (1's) if needed
#         number.cex = 0.4)      #size of number

#-------------------------------KC - BENCHMARK REGRESSION MODEL--------------------------------------------
# Fit a  linear regression model
# kc_lm <- lm(kings_price ~ kings_sqftlot + kings_bathroom + kings_sqftlivingroom + kings_bedroom + 
#                 kings_yearbuilt + kings_condition + kings_floor + kings_zipcode + 
#              kings_longitude + kings_view + kings_sqftbasement + kings_latitude 
#            , data = kings_county)

# Summary of the model
# summary(kc_lm)

# Plotting kings_price vs kings_sqftlot with regression line
#ggplot(kings_county, aes(x = kings_longitude, y = kings_price)) +
 # geom_point() +  # Adds scatter plot of the data points
  #geom_smooth(method = "lm", se = FALSE, color = "Yellow") +  # Adds the linear regression line
  #labs(title = "Linear Regression",
  #     x = "Longitude",
  #     y = "Price") +
  #theme_minimal()

#--------------------------------KC - FEATURE ENGINEERING-------------------------------------------------
### Turning latitude & lot into polynomial cursive ###
kings_county$kings_latitudeD2 <- kings_county$kings_latitude^2

# Comparing latitude linear vs polynomial
model_data <- kings_county[, c("kings_price","kings_latitude","kings_latitudeD2" )]

# Train linear model
lm_model <- train (kings_price ~ kings_latitude, data = model_data, method = "lm", trControl = trainControl(method = "cv", number = 10))

# Train a polynomial model (degree 2)
lm_poly_model <- train(kings_price ~ kings_latitude + kings_latitudeD2, data = model_data, method = "lm", trControl = trainControl(method = "cv", number = 10))

#print(lm_model)
print(lm_poly_model)

# Fit a polynomial regression model (degree 2)
kings_latitudeD2 <- lm(kings_price ~ poly(kings_latitude, 2), data = kings_county)

# Generate predictions using the polynomial model
kings_county$predicted_price <- predict(kings_latitudeD2, newdata = kings_county)

# Plot the data points and the polynomial fit
#ggplot(kings_county, aes(x = kings_latitude, y = kings_price)) +
 # geom_point(color = "black", alpha = 0.5) + # Scatter plot of the data points
  #geom_line(aes(x = kings_latitude, y = predicted_price), color = "orange", size = 1) + # Polynomial fit line
  #labs(title = "Polynomial Regression", 
  #     x = "kings_latitude", 
  #     y = "kings_price") +
  #theme_minimal()

# Fit a polynomial regression model (degree 2)
kings_latitudeD2 <- lm(kings_price ~ poly(kings_latitude, 2), data = kings_county)

# Generate predictions using the polynomial model
kings_county$predicted_price <- predict(kings_latitudeD2, newdata = kings_county)

# Plot the data points and the polynomial fit
#ggplot(kings_county, aes(x = kings_latitude, y = kings_price)) +
 # geom_point(color = "black", alpha = 0.5) + # Scatter plot of the data points
  #geom_line(aes(x = kings_latitude, y = predicted_price), color = "orange", size = 1) + # Polynomial fit line
  #labs(title = "Polynomial Regression", 
  #     x = "kings_latitude", 
  #     y = "kings_price") +
  #theme_minimal()

### Basement Dummy varible ###
# Apply one-hot encoding to the 'Basement' and 'Lot' columns
kings_dummy <- dummyVars(~ kings_sqftbasement, data = kings_county)

# Renaming kings_county dummy variable and setting both feature binary
kings_county_encoded <- as.data.frame(predict(kings_dummy, newdata = kings_county))
colnames(kings_county_encoded) <- c("kings_sqftbasementDUMMY")

# add back original columns 
kings_county <- cbind(kings_county, kings_county_encoded)

# Filter & Plotting data set to have basement only
kings_basement_only <- kings_county[kings_county$kings_sqftbasementDUMMY > 0, ]
kings_nobasement_only <- kings_county[kings_county$kings_sqftbasementDUMMY == 0, ]
#ggplot(kings_nobasement_only, aes(x = kings_sqftbasementDUMMY, y = kings_price)) +
#  geom_point(color = "black", alpha = 0.5) +  # Scatter plot of the data points
#  geom_smooth(method = "lm", color = "pink", size = 1) +  # Linear regression fit line
#  labs(title = "Price vs Basement (No Basement Only)", 
#       x = "No Basement ", 
#       y = "Price") +
#  theme_minimal()

#write.csv(kings_county, "kings_county_Rwrite.csv", row.names = FALSE)

# Comparing basement with vs without
#model_data_actual <- kings_county[, c("kings_price","kings_sqftbasement" )]
#model_data_hasbasement <- kings_basement_only[, c("kings_price","kings_sqftbasementDUMMY" )]
#model_data_nobasement <- kings_nobasement_only[, c("kings_price","kings_sqftbasementDUMMY" )]

# Train linear model
#lm_model_actual <- train (kings_price ~ kings_sqftbasement , data = model_data_actual, method = "lm", trControl = trainControl(method = "cv", number = 10))
#lm_model_hasbasement <- train (kings_price ~ kings_sqftbasementDUMMY , data = model_data_hasbasement, method = "lm", trControl = trainControl(method = "cv", number = 10))
#lm_model_nobasement <- train (kings_price ~ kings_sqftbasementDUMMY , data = model_data_nobasement, method = "lm", trControl = trainControl(method = "cv", number = 10))

#print(lm_model)
#print(lm_model_hasbasement)
#print(lm_model_nobasement)

#Choose kings county with basement
kings_county <- kings_basement_only

# Removing the actually basement column leaving only dummy basement column
kings_county <- kings_county %>% select(-kings_sqftbasement)

# Remove this also if there is no basement
kings_county <- kings_county %>% select(-kings_sqftbasementDUMMY)

#-------------------------------KC - FEATURE SCALING -------------------------------------------------
# Identify the numeric columns
numeric_columns <- sapply(kings_county, is.numeric)  # Identify numeric columns
kings_county_standardized <- kings_county  # Make a copy of your original data

# Standardize only the numeric columns
kings_county_standardized[, numeric_columns] <- scale(kings_county[, numeric_columns])

#-----------------------------KC - SUBSET SELECTION VS REGULARIZATION------------------------------------
### Best subset selection ( Caret ) ###
# Train a linear model with stepwise selection
head(kings_county_standardized)
lm_model <- train(kings_price ~ ., 
                  data = kings_county_standardized, 
                  method = "leapSeq",  # Subset selection using sequential
                 tuneGrid = data.frame(nvmax = 1:10),  # Number of predictors
                  trControl = trainControl(method = "cv", number = 10))
# Get the final model coefficients (the selected predictors and their coefficients)
selected_predictors <- coef(lm_model$finalModel, id = 7)  # 'id = 7' is the subset with 7 predictors
selected_predictors_names <- names(selected_predictors)[-1]
print(selected_predictors_names)
kings_county_subset <- kings_county_standardized[, c("kings_price",selected_predictors_names)]

# Display the selected predictors and their coefficients    
#print(selected_predictors)

# View final model
#print(lm_model)

# Define the tuning grid for lambda values
lambda_grid <- expand.grid(alpha = c(1, 0),  
                           lambda = seq(0.00001, 0.5, length = 100))  # Lambda values for regularization

# Train the Lasso model (L1 regularization)
lasso_model <- train(kings_price ~ ., 
                     data = kings_county_subset, 
                     method = "glmnet", 
                     tuneGrid = lambda_grid[lambda_grid$alpha == 1, ],  
                     trControl = trainControl(method = "cv", number = 10))  # 10-fold cross-validation

# Print out the Lasso model results
print(lasso_model)

# Train the Ridge model (L2 regularization)
ridge_model <- train(kings_price ~ ., 
                     data = kings_county_subset, 
                     method = "glmnet", 
                     tuneGrid = lambda_grid[lambda_grid$alpha == 0, ],  # Ridge (alpha = 0)
                     trControl = trainControl(method = "cv", number = 10))  # 10-fold cross-validation

# Print out the Ridge model results
print(ridge_model)

# Compare the results of Lasso and Ridge models
results <- resamples(list(Lasso = lasso_model, Ridge = ridge_model))
summary(results)

# Train the Lasso model (L1 regularization)
lasso_model_all <- train(kings_price ~ ., 
                     data = kings_county_standardized, 
                     method = "glmnet", 
                     tuneGrid = lambda_grid[lambda_grid$alpha == 1, ],  
                     trControl = trainControl(method = "cv", number = 10))  # 10-fold cross-validation

# Print out the Lasso model results
print(lasso_model_all)

# Train the Ridge model (L2 regularization)
ridge_model_all <- train(kings_price ~ ., 
                     data = kings_county_standardized, 
                     method = "glmnet", 
                     tuneGrid = lambda_grid[lambda_grid$alpha == 0, ],  # Ridge (alpha = 0)
                     trControl = trainControl(method = "cv", number = 10))  # 10-fold cross-validation

# Print out the Ridge model results
print(ridge_model_all)

# Compare the results of Lasso and Ridge models
results_all <- resamples(list(Lasso = lasso_model_all, Ridge = ridge_model_all))
summary(results_all)

#------------------------------KC - PREDICTION ( leverage existing dataset )------------------------------------------------
# Standardize the current data (kings_county)
#pre_proc <- preProcess(kings_county, method = c("center", "scale"))
#kings_county_standardized <- predict(pre_proc, newdata = kings_county_prediction)  

# Use the trained Ridge model to predict prices for your current data
#predictions <- predict(ridge_model, newdata = kings_county_prediction)

# Add the predicted prices to the dataset
#kings_county$predicted_price <- predictions

# View the actual prices vs the predicted actual 
# Converting back standardized predicted price to predicted price
#mean_price <- mean(kings_county$kings_price, na.rm = TRUE)
#std_dev_price <- sd(kings_county$kings_price, na.rm = TRUE)
#kings_county$predicted_price <- (kings_county$predicted_price * std_dev_price) + mean_price
# Combine the actual prices and predicted prices into a data frame
#comparison <- data.frame(Actual = kings_county$kings_price, Predicted = kings_county$predicted_price)

# Print the comparison side by side
#print(comparison)

# Visualization Plot actual and predicted prices with the chosen feature into one data frame
# I begin by taking 20 random sampling data only for better visualization
#kings_county <- kings_county[sample(nrow(kings_county), 30), ]

#kings_comparison_data <- data.frame(
#  kcd_feature = kings_county[["kings_sqftlivingroom"]],  
#  kcd_actual = kings_county$kings_price,  
#  kcd_predicted = kings_county$predicted_price  
#)

#kings_comparison_data <- gather(kings_comparison_data, key = "Price_Type", value = "Price", kcd_actual, kcd_predicted)

# Plot the actual vs predicted prices
#ggplot(kings_comparison_data, aes(x = kcd_feature, y = Price, color = Price_Type)) +
#  geom_line() +  # Line plot for both actual and predicted prices
#  labs(title = "Actual vs Predicted House Prices", 
#       x = "Square feet of Living Room", 
#       y = "Price") +
#  scale_color_manual(values = c("blue", "red")) +  # Customize line colors
#  theme_minimal()  # Clean theme
#-------------------------KC - COMPARISON MEDIAN OF SALE PRICE 2014 v 2024----------------------------
# Enable this if for 2024 Comparison
# Calculate the median price for the actual dataset
#median_kc_2014 <- median(kings_county$kings_price, na.rm = TRUE)
#median_kc_2024 <- 860700 # Get data from zillow website for house prices in kings county

# multiplying the data set house price with median increase
#kings_county$kings_price <- kings_county$kings_price * median_kc_2024/median_kc_2014

# Adding new house simulation
# Example input for the house you're predicting (single row of data)
# 1st house
#new_house <- data.frame(  kings_sqftlivingroom = 2920,  kings_bedroom = 6,  kings_bathroom = 3,  kings_latitude = 47.6161,  kings_longitude = -122.1223,  kings_zipcode = 98008,  kings_view = median(kings_county$kings_view, na.rm = TRUE),  kings_latitudeD2 = 47.6161^2,  kings_sqftbasementDUMMY = median(kings_county$kings_sqftbasementDUMMY, na.rm = TRUE))

# 2nd house
#new_house <- data.frame(  kings_sqftlivingroom = 5947,  kings_bedroom = 6,  kings_bathroom = 7,  kings_latitude = 47.6192,  kings_longitude = -122.1956,  kings_zipcode = 98004,  kings_view = median(kings_county$kings_view, na.rm = TRUE),  kings_latitudeD2 = 47.6192^2,  kings_sqftbasementDUMMY = median(kings_county$kings_sqftbasementDUMMY, na.rm = TRUE))

# 3rd house
#new_house <- data.frame(  kings_sqftlivingroom = 642,  kings_bedroom = 1,  kings_bathroom = 1,  kings_latitude = 47.6021,  kings_longitude = -122.1393,  kings_zipcode = 98007,  kings_view = median(kings_county$kings_view, na.rm = TRUE),  kings_latitudeD2 = 47.6021^2,  kings_sqftbasementDUMMY = median(kings_county$kings_sqftbasementDUMMY, na.rm = TRUE))

# Standardizing new house data
#means <- c(mean(kings_county$kings_sqftlivingroom), mean(kings_county$kings_bedroom),
#           mean(kings_county$kings_bathroom), mean(kings_county$kings_latitude),
#           mean(kings_county$kings_longitude), mean(kings_county$kings_zipcode),
#           mean(kings_county$kings_view), mean(kings_county$kings_latitudeD2),
#           mean(kings_county$kings_sqftbasementDUMMY))
#sds <- c(sd(kings_county$kings_sqftlivingroom), sd(kings_county$kings_bedroom),
#         sd(kings_county$kings_bathroom), sd(kings_county$kings_latitude),
#         sd(kings_county$kings_longitude), sd(kings_county$kings_zipcode),
#         sd(kings_county$kings_view), sd(kings_county$kings_latitudeD2),
#         sd(kings_county$kings_sqftbasementDUMMY))
#new_house$kings_sqftlivingroom <- (new_house$kings_sqftlivingroom - means[1]) / sds[1]
#new_house$kings_bedroom <- (new_house$kings_bedroom - means[2]) / sds[2]
#new_house$kings_bathroom <- (new_house$kings_bathroom - means[3]) / sds[3]
#new_house$kings_latitude <- (new_house$kings_latitude - means[4]) / sds[4]
#new_house$kings_longitude <- (new_house$kings_longitude - means[5]) / sds[5]
#new_house$kings_zipcode <- (new_house$kings_zipcode - means[6]) / sds[6]
#new_house$kings_view <- (new_house$kings_view - means[7]) / sds[7]
#new_house$kings_latitudeD2 <- (new_house$kings_latitudeD2 - means[8]) / sds[8]
#new_house$kings_sqftbasementDUMMY <- (new_house$kings_sqftbasementDUMMY - means[9]) / sds[9]

#head(new_house)

# Predict using the Ridge model
#predicted_price <- predict(ridge_model, newdata = new_house)

# Unstandardize the price
#means <- c(mean(kings_county$kings_price))
#sds <- c(sd(kings_county$kings_price))
#unstandardized_price <- (predicted_price * sds) + means

# Output the predicted price
#print(unstandardized_price)

#------------------------------------KC - HEATMAP -------------------------------------------------------
# Example dataset
#kings_heatmap <- data.frame(
#  latitude = kings_county$kings_latitude,
#  longitude = kings_county$kings_longitude,
#  price = kings_county$kings_price
#)

# Define a color palette (light red to dark red)
#color_palette <- colorNumeric(
#  palette = "Reds",  # Use the Reds color scale
#  domain = kings_heatmap$price  # Map it to the price column
#)

#leaflet(kings_heatmap) %>%
#  addTiles() %>%
#  addHeatmap(
#    lng = ~longitude, lat = ~latitude,
#    intensity = ~price, blur = 20, max = max(kings_heatmap$price), radius = 3
#  ) %>%
#  addCircleMarkers(
#    lng = ~longitude, lat = ~latitude,
#    radius = 3,  # Marker size
#    color = ~color_palette(price),  # Dynamic red intensity based on price
#    fillColor = ~color_palette(price),  # Fill color matches border
#    fillOpacity = 0.8,  # Transparency for the marker fill
#    popup = ~paste("Price: $", format(price, big.mark = ","))
#  ) %>%
#  setView(lng = mean(kings_heatmap$longitude), lat = mean(kings_heatmap$latitude), zoom = 10)

#--------------------------------CR - DATA PREPROCESSING----------------------------------------------------
# naming the rawdata
credit_risk <- rawdata

# Transforming data using one hot encoding
#head(credit_risk)
credit_risk$credit_historicaldefault <- ifelse(credit_risk$credit_historicaldefault == "Y", 1, 0)
credit_risk$credit_employment <- as.numeric(credit_risk$credit_employment)
credit_risk$credit_interestrate <- as.numeric(credit_risk$credit_interestrate)
#unique(credit_risk$credit_ownership)
#unique(credit_risk$credit_loanintent)
#unique(credit_risk$credit_loangrade)
ownership_ohe <- model.matrix(~ credit_ownership - 1, data = credit_risk)
loanintent_ohe <- model.matrix(~ credit_loanintent - 1, data = credit_risk)
loangrade_ohe <- model.matrix(~ credit_loangrade - 1, data = credit_risk)
credit_risk$credit_ownership <- NULL
credit_risk$credit_loanintent <- NULL
credit_risk$credit_loangrade <- NULL
# Adding the encoded column back to our original data file
credit_risk <- cbind(credit_risk, ownership_ohe, loanintent_ohe, loangrade_ohe)

# Saving 5000 data sets for prediction purpose
# Set seed for reproducibility 
set.seed(123)

# Randomly sample 5000 rows from the original dataset
prediction_rows <- sample(1:nrow(credit_risk), size = 5000)

# Create a new dataset with the sampled rows
credit_risk_prediction <- credit_risk[prediction_rows, ]

# Remove the 10,000 sampled rows from the original dataset
credit_risk <- credit_risk[-prediction_rows, ]

#-----------------------------------CR - CLASS IMBALANCES ------------------------------------------  
# Checking wether our response is imbalance
#count_ones <- sum(credit_risk$credit_historicaldefault == 1)
#count_zeros <- sum(credit_risk$credit_historicaldefault == 0)
# Print the result
#print(count_ones)
#print(count_zeros)

# EDA of data to see which method of Class imbalances handling will we use 
# Mean of the feature
#mean_value <- mean(credit_risk$credit_interestrate, na.rm = TRUE)

# Standard deviation of the feature
#var_value <- var(credit_risk$credit_interestrate, na.rm = TRUE) 

# Print the results
#print(paste("Mean: ", mean_value))
#print(paste("Variance: ", var_value))

#-----------------------------------CR - XG BOOST ------------------------------------------------  
# Define the features
#x_train <- as.matrix(credit_risk[, -which(names(credit_risk) == "credit_historicaldefault")])

# Define the target variable
#y_train <- credit_risk$credit_historicaldefault

# Train the Gradient Boosting model
#gbm_model <- xgboost(data = x_train, label = y_train, 
#                     nrounds = 100,              # Number of boosting rounds (iterations)
#                     objective = "binary:logistic", # Binary classification (logistic regression)
#                     eval_metric = "logloss",      # Evaluation metric
#                     max_depth = 6,                # Maximum depth of each tree
#                     eta = 0.3)                   # Learning rate (shrinkage)

# Make predictions on the test data
#x_test <- as.matrix(credit_risk_prediction[, -which(names(credit_risk_prediction) == "credit_historicaldefault")])
#gbm_pred <- predict(gbm_model, x_test)

# Display the first few predictions
#head(gbm_pred)

# Convert predictions to binary outcomes (0 or 1)
#pred_binary <- ifelse(gbm_pred > 0.5, 1, 0)

# Create confusion matrix
#confusion_matrix <- table(Predicted = pred_binary, Actual = credit_risk_prediction$credit_historicaldefault)
#print(confusion_matrix)

# Calculate accuracy
#accuracy <- mean(pred_binary == credit_risk_prediction$credit_historicaldefault)
#print(paste("Accuracy:", accuracy))

# Create a prediction object for AUC
#pred_obj <- prediction(gbm_pred, credit_risk_prediction$credit_historicaldefault)

# Calculate AUC
#auc <- performance(pred_obj, measure = "auc")
#print(paste("AUC:", auc@y.values[[1]]))

# Hyperparameter Tuning
#cv_model <- xgb.cv(data = x_train, label = y_train, 
#                   nrounds = 100, 
#                   objective = "binary:logistic", 
#                   eval_metric = "logloss", 
#                   max_depth = 6, 
#                   eta = 0.3, 
#                   nfold = 5,    # 5-fold cross-validation
#                   verbose = TRUE)

#--------------------------------CR - CORRELATION MATRIX-------------------------------------------
# Compute the correlation matrix
#correlation_matrix <- cor(credit_risk, use = "complete.obs", method = "pearson")

# Create a correlation plot
#corrplot(correlation_matrix, 
#      method = "circle",     # Visualization method (can also use "square", "number", etc.)
#      type = "upper",        # Show only the upper triangle
#      order = "hclust",      # Order correlations by hierarchical clustering
#      tl.col = "black",      # Set text label color
#      tl.srt = 45,           # Rotate text labels to 45 degrees
#      tl.cex = 0.4,          # text size
#      addCoef.col = "orange",  # Display correlation coefficients on the plot
#      diag = FALSE,          # Hide diagonal (1's) if needed
#      number.cex = 0.4)      #size of number

#--------------------------------CR - STRATIFIED SAMPLING--------------------------------------------
# Stratified sampling: Select 1000 default (1) and 1000 non-default (0) cases
default_yes <- credit_risk %>% filter(credit_historicaldefault == 1) %>% sample_n(1000)
default_no <- credit_risk %>% filter(credit_historicaldefault == 0) %>% sample_n(1000)

# Combine the two sampled datasets (balanced dataset)
credit_risk <- bind_rows(default_yes, default_no)

# Check the distribution in the balanced dataset
#table(credit_risk$credit_historicaldefault)

#---------------------------------- CR - PCA -----------------------------------------------------------
# Scale the dataset without the response variable
#credit_risk_scaled <- scale(credit_risk[, -which(names(credit_risk) == "credit_historicaldefault")])

# Apply PCA
#pca_model <- prcomp(credit_risk_scaled, center = TRUE, scale. = TRUE)

# Check variance get 95%
# summary(pca_model)

# Plot the cumulative variance explained by each principal component
#plot(cumsum(pca_model$sdev^2) / sum(pca_model$sdev^2), xlab = "Number of Components", 
#     ylab = "Cumulative Proportion of Variance Explained", type = "b", main = "Variance Explained by PCA")

# Select up to 95% principal components (example)
#credit_risk_pca <- as.data.frame(pca_model$x[, 1:17])  # Keep the first 10 components

# Add the target variable back to the PCA data
#credit_risk_pca$credit_historicaldefault <- credit_risk$credit_historicaldefault

# Check the structure of the PCA-transformed data
#  str(credit_risk_pca)

#---------------------CR - LOGISTIC R & HYPERPARAMETER TUNING  -----------------------------------------------
# Train logistic regression using the PCA-transformed features
#logistic_pca <- glm(credit_historicaldefault ~ ., data = credit_risk_pca, family = binomial())

# Summarize the model
#summary(logistic_pca)

# Predict on the PCA-transformed restratified dataset
#log_pred <- predict(logistic_pca, credit_risk_pca, type = "response")

# Convert probabilities to binary outcomes
#log_pred_binary <- ifelse(log_pred > 0.5, 1, 0)

# Create confusion matrix
#confusion_matrix <- table(Predicted = log_pred_binary, Actual = credit_risk_pca$credit_historicaldefault)
#print(confusion_matrix)

# Calculate accuracy
#accuracy <- mean(log_pred_binary == credit_risk_pca$credit_historicaldefault)
#print(paste("Accuracy:", accuracy))

# Calculate AUC
#pred_obj <- prediction(log_pred, credit_risk_pca$credit_historicaldefault)
#auc <- performance(pred_obj, measure = "auc")
#print(paste("AUC:", auc@y.values[[1]]))
#-------------------------- CR - PLOTTING LOGISTIC REGRESSION AGAINST PC1 -------------------------
# Fit the logistic regression model PC1 and Historical Default
#PC1_model <- glm(credit_historicaldefault ~ PC1, data = credit_risk_pca, family = "binomial")

# Generate a sequence of PC1 values for smooth plotting
#pc1_seq <- seq(min(credit_risk_pca$PC1), max(credit_risk_pca$PC1), length.out = 2000)

# Predict probabilities for the PC1 sequence
#predicted_probs <- predict(PC1_model, newdata = data.frame(PC1 = pc1_seq), type = "response")

# Create the plot
#ggplot(credit_risk_pca, aes(x = PC1, y = credit_historicaldefault)) +
#  geom_point(aes(color = as.factor(credit_historicaldefault)), size = 2, alpha = 0.7) +  
#  geom_line(data = data.frame(PC1 = pc1_seq, Predicted = predicted_probs), 
#            aes(x = PC1, y = Predicted), color = "blue", size = 1) +  # Sigmoid curve
#  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +  # Threshold line
#  labs(
#    title = "Logistic Regression with PC1",
#    x = "Principal Component 1 (PC1)",
#    y = "Probability of Historical Default",
#    color = "Actual Default"
#  ) +
#  theme_minimal()

# Fit the logistic regression model Interest rate and Historical Default
#interestrate_model <- glm(credit_historicaldefault ~ credit_interestrate, data = credit_risk, family = "binomial")

# Generate a sequence of InterestRate values for smooth plotting
#interestrate_seq <- seq(min(credit_risk$credit_interestrate), max(credit_risk$credit_interestrate), length.out = 2000)

# Predict probabilities for the InterestRate sequence
#predicted_probs <- predict(interestrate_model, newdata = data.frame(credit_interestrate = interestrate_seq), type = "response")

# Plotting with InterestRate for Interpretability
#ggplot(credit_risk, aes(x = credit_interestrate, y = credit_historicaldefault)) +
#  geom_point(aes(color = as.factor(credit_historicaldefault)), size = 2, alpha = 0.7) + 
#  geom_line(data = data.frame(credit_interestrate = interestrate_seq, Predicted = predicted_probs), 
#            aes(x = credit_interestrate, y = Predicted), color = "green", size = 1) +  # Sigmoid curve
#  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray") +  # Threshold line
#  labs(
#    title = "Logistic Regression: Historical Default vs Interest Rate",
#    x = "Interest Rate",
#    y = "Probability of Default"
#  ) +
#  scale_color_manual(values = c("yellow", "orange"), labels = c("No Default", "Default")) +
#  theme_minimal()

#---------------------------------------CR - CLUSTERING---------------------------------------------------
# Lets start by ensuring randomness by reproducibility
set.seed(123)
# This regards InterestRate as the only key factor
credit_risk_all <- credit_risk
kmeans_credit <- kmeans(credit_risk$credit_interestrate, centers = 3, nstart = 25)
credit_risk$Cluster <- as.factor(kmeans_credit$cluster)
# This uses all feature as the factor only activate 1
scaled_data <- scale(credit_risk_all)
kmeans_credit_all <- kmeans(scaled_data, centers = 3, nstart = 25)
credit_risk_all$Cluster_all <- as.factor(kmeans_credit_all$cluster)

# Define colors
#cluster_colors <- c("1" = "red", "2" = "green", "3" = "yellow")

# Plot for interest rate only
#p1 <- ggplot(credit_risk, aes(x = credit_interestrate, y = credit_historicaldefault, color = as.factor(Cluster))) +
#  geom_point() +
#  scale_color_manual(values = cluster_colors) +
#  labs(title = "Clustering with Interest Rate Only",
#       x = "Interest Rate",
#       y = "Historical Default",
#       color = "Cluster") +
#  theme_minimal()

# Plot for all features
#p2 <- ggplot(credit_risk_all, aes(x = credit_interestrate, y = credit_historicaldefault, color = as.factor(Cluster_all))) +
#  geom_point() +
#  scale_color_manual(values = cluster_colors) +
#  labs(title = "Clustering with All Features",
#       x = "Interest Rate",
#       y = "Historical Default",
#       color = "Cluster") +
#  theme_minimal()

# Create a mapping vector based on the summaries
#cluster_mapping <- c("1" = "3", "2" = "1", "3" = "2")  # Adjust mapping based on analysis

# Apply the mapping to align Cluster_all
#credit_risk_all$Cluster_aligned <- as.factor(cluster_mapping[as.character(credit_risk_all$Cluster_all)])

# Arrange plots side by side for comparison
#p1 + p2

# Compare historical default rates across clusters
#cluster_summary <- aggregate(cbind(credit_interestrate, credit_historicaldefault) ~ Cluster, data = credit_risk, mean)
#cluster_all_summary <- aggregate(cbind(credit_interestrate, credit_historicaldefault) ~ Cluster_all, data = credit_risk_all, mean)
#print(cluster_summary)
#print(cluster_all_summary)

#------------------------------- CR - SEMI SUPERVISED LEARNING ----------------------------------------------
# Logistic model using Cluster only interest rate
#logistic_model_rate <- glm(credit_historicaldefault ~ Cluster, 
#                      data = credit_risk, 
#                      family = "binomial")

# Prediction on test set
#credit_risk$predicted_prob <- predict(logistic_model_rate, newdata = credit_risk, type = "response")

# Evaluate the model
#pred <- prediction(credit_risk$predicted_prob, credit_risk$credit_historicaldefault)
#perf <- performance(pred, "tpr", "fpr")
#auc <- performance(pred, "auc")@y.values[[1]]
#cat("AUC:", auc, "\n")

# Logistic regression using all features
#logistic_model_all <- glm(credit_historicaldefault ~ Cluster_all, 
#                          data = credit_risk_all, 
#                          family = "binomial")

# Predict probabilities
#credit_risk_all$predicted_prob_all <- predict(logistic_model_all, newdata = credit_risk_all, type = "response")

# Evaluate the model
#pred_all <- prediction(credit_risk_all$predicted_prob_all, credit_risk_all$credit_historicaldefault)
#perf_all <- performance(pred_all, "tpr", "fpr")
#auc_all <- performance(pred_all, "auc")@y.values[[1]]
#cat("AUC for all-features clusters:", auc_all, "\n")

# Combine predicted probabilities into a single data frame for comparison
#comparison_data <- data.frame(
#  Default = credit_risk$credit_historicaldefault,
#  Predicted_Prob_Rate = credit_risk$predicted_prob,
#  Predicted_Prob_All = credit_risk_all$predicted_prob_all
#)

# Plot histograms for both clustering approaches
#p1 <- ggplot(comparison_data, aes(x = Predicted_Prob_Rate, fill = as.factor(Default))) +
#  geom_histogram(binwidth = 0.05, position = "identity", alpha = 0.7) +
#  labs(
#    title = "Interest Rate Clusters",
#    x = "Predicted Probability",
#    fill = "Default (0/1)"
#  ) +
#  theme_minimal()

#p2 <- ggplot(comparison_data, aes(x = Predicted_Prob_All, fill = as.factor(Default))) +
#  geom_histogram(binwidth = 0.05, position = "identity", alpha = 0.7) +
#  labs(
#    title = "All Features Clusters",
#    x = "Predicted Probability",
#    fill = "Default (0/1)"
#  ) +
#  theme_minimal()
#p1 + p2

#random_sample_rate <- credit_risk[sample(1:nrow(credit_risk[credit_risk$credit_historicaldefault == 1, ]
#), 10), ] 
#random_sample_all <- credit_risk_all[sample(1:nrow(credit_risk_all[credit_risk_all$credit_historicaldefault == 1, ]
#), 10), ] 
#View(random_sample_rate)
#View(random_sample_all)

# overlapping visualization
#distance_matrix <- dist(credit_risk_all[, !names(credit_risk_all) %in% c("Cluster_all")])
#silhouette_scores <- silhouette(as.numeric(credit_risk_all$Cluster_all), distance_matrix)
#summary(silhouette_scores)