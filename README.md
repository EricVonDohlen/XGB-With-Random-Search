# XGBoost With Random Search
R XGboost With Random Search for Hyperparameters

This file contains what I call a 'slug' for estimation of an XG Boost algorithm in R that uses random search to tune the various parameters. Be warned that it is very brute force; no effort is made to update hyperparameter distributions based on validation results. Moreover, no crossvalidation is performed. I view this slug as a way to develop priors about hyperparameter values so more elegant tuning may be performed. 

library(dplyr)
require(zoo)
require(reshape)
require(tidyr)
require(readr)
library(lubridate)
library(dint)
library(RODBC)
library(xgboost)
memory.limit(99999)




linear <- glm(Y1 ~ X1 + X2 + X3 + X4 + X4 + X5 + X6 + X7 + X8 + X9 + X10, family = binomial, data=model_data )
summary(linear)





#SPLIT INTO TEST AND TRAIN
smp_size <- floor(0.67 * nrow(model_data))

#SET THE TRAINING DATA
train_ind <- sample(seq_len(nrow(model_data)), size = smp_size)

#TEST AND TRAIN DATASET
xgb_train <- model_data[train_ind, ]
xgb_test <- model_data[-train_ind, ]

#CREATE XGB DM MATRIX
dm_matrix_train <- xgb.DMatrix(data=as.matrix(xgb_train %>% select(-Y1)), label=as.numeric(xgb_train$Y1), missing=NA)
dm_matrix_test <- xgb.DMatrix(data=as.matrix(xgb_test %>% select(-Y1)), label=as.numeric(xgb_test$Y1), missing=NA)





# Take start time to measure time of random search algorithm
start.time <- Sys.time()

# Create empty lists
best_auc_list = list()
parameters_list = list()

# Create 10000 rows with random hyperparameters
set.seed(20)
for (iter in 1:10){
  param <- list(booster = "gbtree",
                objective = "binary:logistic",
                max_depth = sample(3:8, 1),
                nrounds = sample(c(150,200,250,300,500,600,700,1000),1),
                eta =  runif(1, .001, .1),
                gamma = sample(c(0,0.01,0.99),1),
                lambda = sample(c(.5,.6,1),1),
                alpha = sample(c(0.5,1),1),
                subsample = runif(1, .7, 1),
                colsample_bytree = runif(1, .5, 1),
                colsample_bylevel = runif(1, .5, 1),
                colsample_bynode =  runif(1, .5, 1),
                min_child_weight =  sample(0:15, 1)
  )
  parameters <- as.data.frame(param)
  parameters_list[[iter]] <- parameters
}

# Create object that contains all randomly created hyperparameters
parameters_df = do.call(rbind, parameters_list)

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(parameters_df)){
  set.seed(20)
  mdcv <- xgb.train(data=dm_matrix_train,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    max_depth = parameters_df$max_depth[row],
                    eta = parameters_df$eta[row],
                    subsample = parameters_df$subsample[row],
                    gamma = parameters_df$gamma[row],
                    alpha = parameters_df$alpha[row],
                    lambda = parameters_df$lambda[row],
                    colsample_bytree = parameters_df$colsample_bytree[row],
                    colsample_bylevel = parameters_df$colsample_bylevel[row],
                    colsample_bynode = parameters_df$colsample_bynode[row],
                    min_child_weight = parameters_df$min_child_weight[row],
                    nrounds= parameters_df$nrounds[row],
                    eval_metric = "auc",
                    early_stopping_rounds= 30,
                    print_every_n = 10,
                    watchlist = list(train= dm_matrix_train, val= dm_matrix_test)
  )
  best_auc <- as.data.frame(max(mdcv$evaluation_log$val_auc))
  best_auc_list[[row]] <- best_auc
  print(mdcv$nfeatures)
  print(row)
}

# Create object that contains all accuracies
best_auc_df = do.call(rbind, best_auc_list)

# Bind columns of accuracy values and random hyperparameter values
randomsearch = cbind(best_auc_df, parameters_df)


# Stop time and calculate difference
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

setwd('C:/Users/evd19/OneDrive/documents')
write.csv(randomsearch, 'randomsearch.csv', row.names=F)

# Load random search output
randomsearch <- read.csv("randomsearch.csv")




# Isolate nrounds since there was a small range used before -----------------------------------------------------------------------------------------------------------------

randomsearch[1,]





# Final tuned model ---------------------------------------------------------------------------------------------
set.seed(20)
params <- list(booster = "gbtree", 
               objective = "binary:logistic",
               max_depth = randomsearch[1,]$max_depth,
               eta = randomsearch[1,]$eta,
               gamma = randomsearch[1,]$gamma,
               lambda = randomsearch[1,]$lambda,
               alpha = randomsearch[1,]$alpha,
               subsample = randomsearch[1,]$subsample,
               colsample_bytree = randomsearch[1,]$colsample_bytree,
               colsample_bylevel = randomsearch[1,]$colsample_bylevel,
               colsample_bynode = randomsearch[1,]$colsample_bynode,
               min_child_weight = randomsearch[1,]$min_child_weight)
xgb_tuned_final <- xgb.train(params = params,
                             data = dm_matrix_train,
                             nrounds = randomsearch[1,]$nrounds,
                             print_every_n = 10,
                             eval_metric = "auc",
                             early_stopping_rounds = 30,
                             watchlist = list(train= dm_matrix_train, val= dm_matrix_test))

xgb.save(xgb_tuned_final,'whateveryouwanttocallit')

# Analysis -----------------------------------------------------------------


predtest <- cbind(predicted=predict(xgb_tuned_final,dm_matrix_test),actual=xgb_test$Y1)
predtrain <- cbind(predicted=predict(xgb_tuned_final,dm_matrix_train),actual=xgb_train$Y1)

pred_final <- as.data.frame(rbind(predtest,predtrain))


pred_final$decile <- dplyr::ntile(pred_final$predicted,10)
lift <- pred_final %>%  group_by(decile) %>% summarise(n=n(),mpred=mean(predicted),mact=mean(actual))


final_model_importance <- as.data.frame(xgb.importance(colnames(dm_matrix_test), model = xgb_tuned_final))



