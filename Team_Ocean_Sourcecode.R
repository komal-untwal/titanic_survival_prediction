library(tidyverse)
library(data.table)
library(lightgbm)

dat <- fread("/Users/rajesh/MS_Masters_docs/STUDY_Material/Sem_1/R_Programming/Project/train.csv", data.table = F) %>%
  mutate(is_test = 0) %>%
  bind_rows(
    fread("/Users/rajesh/MS_Masters_docs/STUDY_Material/Sem_1/R_Programming/Project/test.csv", data.table = F) %>% 
      mutate(is_test = 1))

get_cabin_data <- function(x, type = "alpha"){
  if(type == "alpha") return(substr(x,1,1))
  if(type == "number") return(as.numeric(substr(x,2,nchar(x))))  
}
get_cabin_data <- Vectorize(get_cabin_data)

age_na_fill <- dat %>% group_by(Pclass, Sex) %>% 
  summarise(Age_na = mean(Age, na.rm = T))

fare_na_fill <- dat %>% group_by(Pclass) %>% 
  summarise(Fare_na = mean(Fare, na.rm = T))

dat <- dat %>%
  left_join(age_na_fill, by = c("Pclass", "Sex")) %>%
  mutate(Age = ifelse(is.na(Age), Age_na, Age)) %>%
  select(-Age_na) %>%
  left_join(fare_na_fill, by = c("Pclass")) %>%
  mutate(Fare = ifelse(is.na(Fare), Fare_na, Fare)) %>%
  select(-Fare_na) %>%
  separate(col = Name, into = c("Name1", "Name2"), sep = ", ") %>%
  #mutate(Ticket = str_replace_all(Ticket, "\\.|\\/", "")) %>%
  add_count(Ticket) %>%
  rename(Ticket_N = n) %>%
  mutate(Cabin_number = get_cabin_data(Cabin, type = "number"),
         Cabin_alpha = get_cabin_data(Cabin, type = "alpha"),
         Cabin_alpha = ifelse(Cabin_alpha == "", "missing", Cabin_alpha),
         Ticket_alpha = str_replace_all(string = Ticket, pattern = regex("[0-9]"), replacement = ""),
         Ticket_alpha = str_replace_all(string = Ticket_alpha, pattern = " ", replacement = ""),
         Ticket_alpha = ifelse(Ticket_alpha == "", "missing", Ticket_alpha),
         Embarked = ifelse(Embarked == "", "S", Embarked)
  ) %>%
  select(-Cabin, -Name1, -Name2, -Ticket, -Cabin_number) %>%
  mutate_if(negate(is.numeric), function(x){as.numeric(as.factor(x))})

non_feats <- c("PassengerId", "Survived", "is_test")
feats <- setdiff(colnames(dat), non_feats)

set.seed(2710)
NFOLDS <- 10
val_ind <- caret::createFolds(y = dat %>% filter(is_test == 0) %>% .$Survived, k = NFOLDS)

p_test_bin <<- matrix(0, nrow = dat %>% filter(is_test == 1) %>% nrow, ncol = NFOLDS)
p_test_prob <<- matrix(0, nrow = dat %>% filter(is_test == 1) %>% nrow, ncol = NFOLDS)
train_fold <- function(fold){
  
  x_train <- dat[dat$is_test == 0,][-val_ind[[fold]],feats]
  y_train <- dat[dat$is_test == 0,][-val_ind[[fold]],]$Survived
  x_val <-  dat[dat$is_test == 0,][val_ind[[fold]],feats]
  y_val <- dat[dat$is_test == 0,][val_ind[[fold]],]$Survived
  
  params <- list(objective = "binary", 
                 metric = "auc",
                 boosting = "gbdt", 
                 learning_rate = 0.01,
                 sub_feature = 0.8,
                 sub_row = 0.8,
                 num_leaves = 20,
                 reg_alpha = 3e-5,
                 reg_lambda = 9e-2,
                 seed = 27)
  
  xtr <- lgb.Dataset(as.matrix(x_train), label = y_train, free_raw_data = F)
  xval <- lgb.Dataset(as.matrix(x_val), label = y_val, free_raw_data = F)
  
  m_lgb <- lgb.train(params = params,
                     data = xtr,
                     nrounds = 5000,
                     valids = list(val = xval),
                     early_stopping_rounds = 100, 
                     eval_freq = 250, 
                     verbose = -1)
  
  imp = lightgbm::lgb.importance(m_lgb)
  head(imp)
  
  p <- lightgbm:::predict.lgb.Booster(m_lgb, as.matrix(x_val), num_iteration = m_lgb$best_iter)
  
  find_thresh <- function(thresh, p){
    t <- quantile(p, thresh)
    p_bin <- ifelse(p >= t, 1,0)
    score <- MLmetrics::Accuracy(y_pred = p_bin, y_true = y_val)
    return(tibble(score, thresh = t))
  }
  
  thresh_df <- purrr::map_dfr(seq(.4,.7,.0001), find_thresh, p = p)
  best_thresh <- arrange(thresh_df, desc(score)) %>% head(1) %>% .$thresh
  best_acc <- arrange(thresh_df, desc(score)) %>% head(1) %>% .$score
  
  cat("fold:", fold, "thresh:", best_thresh,"ACC:", best_acc, "AUC:", m_lgb$best_score, "iter:", m_lgb$best_iter,"\n")
  
  p_te <- lightgbm:::predict.lgb.Booster(m_lgb, as.matrix(dat %>% filter(is_test == 1) %>% select(feats), num_iteration = m_lgb2$best_iter))
  
  p_test_bin[,fold] <<- ifelse(p_te >= best_thresh, 1,0)
  p_test_prob[,fold] <<- p_te
  
  return(tibble(auc_scores = m_lgb$best_score, acc_scores = best_acc, thresh = best_thresh))
}

out <- purrr::map_dfr(1:NFOLDS, train_fold)

cat("final model scores:", "ACC:", mean(out$acc_scores), "AUC:", mean(out$auc_scores), "\n")

p_majority <- apply(p_test_bin,1,function(x) as.numeric(names(which.max(table(x)))) )

p_test_prob <- rowMeans(p_test_prob)
p_prob <- ifelse(p_test_prob >= mean(unique(out$thresh)), 1,0)

#--- majority voting
majority <- fread("/Users/rajesh/MS_Masters_docs/STUDY_Material/Sem_1/R_Programming/Project/sample_submission.csv") %>%
  mutate(Survived = p_majority)
fwrite(majority, "majority.csv")
#--- proba mean
surv_prob <- fread("/Users/rajesh/MS_Masters_docs/STUDY_Material/Sem_1/R_Programming/Project/sample_submission.csv") %>%
  mutate(Survived = p_prob)
fwrite(surv_prob, "survival_pred.csv")