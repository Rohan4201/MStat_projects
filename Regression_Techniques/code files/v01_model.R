
library(discrim)
library(MASS)
library(klaR)

library(tidyverse)
library(tidymodels)
library(glmnet)
library(patchwork)

################ Fonts

sysfonts::font_add_google(name = "Ubuntu", family = "ubuntu")
sysfonts::font_add_google(name = "Noto Serif", family = "noto")
sysfonts::font_add_google("IBM Plex Serif", "Regular 400")


not <- "noto"
ubun <- "ubuntu"
plex_serif <- "Regular 400"

showtext::showtext_auto()

###########################################
############################################
############################################

#### Input data


fara_df <- 
  read_csv("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/farad.csv",
           na = c("", "NA", "NULL"))

to_be_removed <- (fara_df %>%
                    select(-(ends_with("20") | ends_with("20share"))) %>%
                    select(ends_with("10") | ends_with("10share")) %>%
                    colnames())[13:37]

fara_df_removed <- fara_df %>%
  select(-(ends_with("20") | ends_with("20share"))) %>%
  select(-all_of(to_be_removed))


rm(to_be_removed)

fara_df_removed %>% 
          select(fara_df_removed %>%
                   select(lapophalf:TractSNAP) %>%
                   select((ends_with("1") | ends_with("1share"))) %>%
                   colnames()) %>%
          filter(!is.na(lahunv1share), is.na(lawhite1share)) %>%
  sapply(., function(column) {sum(is.na(column))})
  



fara_df_1_removed <- fara_df_removed %>%
  select(-(fara_df_removed %>%
           select(lapophalf:TractSNAP) %>%
           select((ends_with("1") | ends_with("1share"))) %>%
           colnames())) %>%
  select(-lapop10, -LAPOP1_10, -LALOWI1_10) %>%
  mutate_if(is.character, factor) %>%
  mutate(Urban = factor(Urban),
         GroupQuartersFlag = factor(GroupQuartersFlag),
         LILATracts_1And10 = factor(LILATracts_1And10),
         LILATracts_halfAnd10 = factor(LILATracts_halfAnd10),
         LILATracts_Vehicle = factor(LILATracts_Vehicle),
         HUNVFlag = factor(HUNVFlag),
         LowIncomeTracts = factor(LowIncomeTracts),
         LA1and10 = factor(LA1and10),
         LAhalfand10 = factor(LAhalfand10),
         LATracts_half = factor(LATracts_half),
         LATracts1 = factor(LATracts1),
         LATracts10 = factor(LATracts10)) %>%
  na.omit()


rm(fara_df)
rm(fara_df_removed)

###### Modeling

model_data <- fara_df_1_removed %>%
  dplyr::select(LILATracts_halfAnd10, LAPOP05_10:TractSNAP, Urban:GroupQuartersFlag,
         PCTGQTRS, HUNVFlag:MedianFamilyIncome, -LowIncomeTracts)

rm(fara_df_1_removed)

#model_data <- model_data %>%
#  dplyr::select(LILATracts_halfAnd10:LALOWI05_10, TractLOWI:MedianFamilyIncome,
#         (model_data %>%
#             dplyr::select(lapophalf:lasnaphalfshare) %>%
#             dplyr::select(ends_with("share")) %>%
#             colnames())) 


set.seed(101)
fara_test_train_split <- initial_split(model_data, strata = LILATracts_halfAnd10)

fara_train <- training(fara_test_train_split)
fara_test <- testing(fara_test_train_split)


set.seed(111)
cv_folds <- vfold_cv(fara_train, v = 10)

#######################################################################################


glmnet_recipe <- 
  recipe(formula = LILATracts_halfAnd10 ~ ., data = model_data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 
  

glmnet_spec <- 
  logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet") 

glmnet_workflow <- 
  workflow() %>% 
  add_recipe(glmnet_recipe) %>% 
  add_model(glmnet_spec) 

glmnet_grid <- tidyr::crossing(penalty = 10^seq(-6, -1, length.out = 9), 
                               mixture = c(0.05, seq(0.1, 1, 0.1))) 


doParallel::registerDoParallel()
glmnet_tune <- 
  tune_grid(glmnet_workflow, resamples = cv_folds, grid = glmnet_grid) 





# 



show_best(glmnet_tune, metric = "roc_auc")

best_regualarization_tune <- select_best(glmnet_tune, metric = "roc_auc")

best_pct_loss_regularization <- select_by_pct_loss(glmnet_tune, desc(penalty),metric = "roc_auc")


### penalty: 0.000001 , mixture = 1
############################### 

## Fitting logistic regression model and predicting probablities


final_logis_wf <- finalize_workflow(glmnet_workflow, best_regualarization_tune)

logis_fit <- fit(final_logis_wf, data = fara_train)

logistic_predictions <- augment(logis_fit, data = fara_train, new_data = fara_train,
                                type.predict = "response")

final_logis_wf_pct <- finalize_workflow(glmnet_workflow, best_pct_loss_regularization)

logis_fit_pct <- fit(final_logis_wf_pct, data = fara_train)

logistic_predictions_pct <- augment(logis_fit_pct, data = fara_train, new_data = fara_train,
                                type.predict = "response")


write_csv(logistic_predictions, 
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/logistic_predictions.csv")

logistic_predictions %>%
  probably::threshold_perf(estimate = .pred_0, truth = LILATracts_halfAnd10,
                           thresholds = seq(0.1, 0.95, by = 0.005)) %>%
  filter(.metric != "distance") %>%
  ggplot(aes(x = .threshold, y  = .estimate, color = .metric)) +
  geom_line()+
  theme_bw()


### ROC curve

roc_auc(logistic_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second")
  autoplot()

write_csv(roc_curve(logistic_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
                  event_level = "second"),
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/logistic_roc_data.csv")


## Precision

accuracy(logistic_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class, 
          event_level = "second") 
  autoplot()

#####################################################################################
######################################################################################
  
## Threshold selection

all_metrics_threshold <- function(threshold){
  logistic_predictions <- logistic_predictions %>%
    mutate(.pred_class_threshold = factor(ifelse(.pred_1 > threshold , 1, 0)))
  
  conf_mat(logistic_predictions,
         truth = LILATracts_halfAnd10, estimate = .pred_class_threshold, 
         event_level = "second")%>%
  summary()

}



threshold_vec <- seq(0.05, 0.9, 0.01)

metric_logis_all <- map_dfr(threshold_vec, ~ all_metrics_threshold(.x)) %>%
  mutate(threshold_val = rep(seq(0.05,0.9,0.01), each = 13))


logistic_metric_plot <- metric_logis_all %>%
  filter(.metric == c("accuracy", "precision", "sens", "j_index", "bal_accuracy","recall"
  )) %>%
  ggplot(aes(x = threshold_val, y = .estimate, color = .metric)) +
  geom_line()+
  labs(title = "Logistic Regression: Metric estimates for different threshold values")+
  geom_vline(aes(xintercept = 0.443))+
  geom_vline(aes(xintercept = 0.5), color = "grey50")+
  geom_vline(aes(xintercept = 0.35), color = "grey50")+
  ggthemes::scale_color_wsj()+
  theme_bw()+
  theme(axis.title = element_text(size = 23, family = ubun),
        axis.text = element_text(size = 23, family = ubun),
        legend.position = "right",
        legend.text = element_text(size = 23, family = ubun),
        legend.title =  element_text(size = 23, family = ubun),
        plot.title = element_text(size = 35, family = plex_serif, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 22.5, family = ubun, hjust = 0))


ggsave(logistic_metric_plot, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/logistic_metric_plot.jpg",
       width = 22, height = 13, units = "cm")


#### Threshold selected is 0.44

logistic_test_predictions <- augment(logis_fit, new_data = fara_test,
                                     type.predict = "response")


logistic_test_predictions <- logistic_test_predictions %>%
  mutate(.pred_class_0.44 = factor(ifelse(.pred_0 > 0.44, 0, 1)))

conf_mat(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44)


write_csv(logistic_test_predictions, 
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/logistic_test_predictions.csv")


####### Threshold_value set to 0.44

accuracy(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44, 
         event_level = "second") 

roc_auc(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
         event_level = "second") 
####### Variable importance

variable_importance_logistic <- logis_fit %>% extract_fit_parsnip() %>% vip::vi() %>% 
  mutate(Importance = abs(Importance)) %>%
  arrange(desc(Importance)) %>%
  head(9) %>%
  ggplot(aes(y = Importance, x = reorder(factor(Variable), desc(Importance)), 
             fill = factor(Variable)))+
  geom_col()+
  theme_bw()+
  labs(y = "Importance")+
  hrbrthemes::scale_fill_ipsum()+
  theme(axis.title.x = element_text(size = 23, family = ubun),
        axis.title.y = element_blank(),
        axis.text = element_text(size = 23, family = ubun),
        legend.position = "none",
        legend.text = element_text(size = 23, family = ubun),
        legend.title =  element_text(size = 23, family = ubun),
        plot.title = element_text(size = 35, family = plex_serif, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 22.5, family = ubun, hjust = 0))+
  coord_flip()


ggsave(variable_importance_logistic, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/variable_importance_logistic.jpg",
       width = 22, height = 13, units = "cm")


roc_curve_logistic<- roc_curve(logistic_test_predictions, 
                               truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity))+
  geom_smooth(stat="identity")+
  annotate("curve",curvature=0,
           x=0,xend=1,y=0,yend=1,linetype="dotted")+
  
  labs(title="ROC for the logistic regression model",
       x="1 - Specificity", y="Sensitivity"
       #       subtitle="Area under this ROC curve is about 0.949 which suggests that the\nLogistic model with aforementioned variables provides a good fit for the data")+
  )+  
  hrbrthemes::theme_tinyhand()+
  theme(plot.title = element_text(family=ubun,size=20,hjust=0.2),
        plot.subtitle = element_text(family = not, size = 17.5, hjust = 0.5, color = "white"),
        axis.title.x = element_text(family=not,size=12),
        axis.title.y = element_text(family=not,size=12))

ggsave(roc_curve_logistic, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/roc_curve_logistic.jpg",
       width = 22, height = 13, units = "cm")





#####################################################################################

#### Extras

fara_df_1_removed %>%
  ggplot(aes(x = 10 ^(-(PovertyRate)), y = LILATracts_halfAnd10)) +
  geom_point(color = "midnightblue", alpha = 0.3) +
  theme_bw()




######################################################################################
#############################################################################################
#############################################################################################


glmnet_recipe_poly <- 
  recipe(formula = LILATracts_halfAnd10 ~ ., data = model_data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_ns(c(lasnaphalf, lasnaphalfshare, TractSNAP, PovertyRate, OHU2010, MedianFamilyIncome,
          lapophalf, lapophalfshare))
  


glmnet_spec <- 
  logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet") 

glmnet_workflow_poly <- 
  workflow() %>% 
  add_recipe(glmnet_recipe_poly) %>% 
  add_model(glmnet_spec) 

glmnet_grid <- tidyr::crossing(penalty = 10^seq(-6, -1, length.out = 10), 
                               mixture = c(0.05, seq(0.1, 1, by = 0.1))) 



doParallel::registerDoParallel()
glmnet_tune_poly <- 
  tune_grid(glmnet_workflow_poly, resamples = cv_folds, grid = glmnet_grid,
            control = control_grid(verbose = T)) 





# 



show_best(glmnet_tune_poly, metric = "roc_auc")

best_regualarization_tune_poly <- select_best(glmnet_tune_poly, metric = "roc_auc")

select_by_one_std_err(glmnet_tune_poly, desc(penalty),metric = "roc_auc")



############################### 

## Fitting logistic regression model and predicting probablities


final_logis_wf_poly <- finalize_workflow(glmnet_workflow_poly, best_regualarization_tune_poly)

logis_fit_poly <- fit(final_logis_wf_poly, data = fara_train)

logistic_predictions_poly <- augment(logis_fit_poly, data = fara_train, new_data = fara_train,
                                type.predict = "response")


logistic_predictions_poly %>%
  probably::threshold_perf(estimate = .pred_0, truth = LILATracts_halfAnd10,
                           thresholds = seq(0.1, 0.95, by = 0.005)) %>%
  filter(.metric != "distance") %>%
  ggplot(aes(x = .threshold, y  = .estimate, color = .metric)) +
  geom_line()+
  theme_bw()


### ROC curve

roc_curve(logistic_predictions_poly, truth = LILATracts_halfAnd10, estimate = .pred_1, 
        event_level = "second") %>%
autoplot()




## Precision

accuracy(logistic_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class, 
         event_level = "second") 
autoplot()

#####################################################################################
######################################################################################

## Threshold selection

all_metrics_threshold_poly <- function(threshold){
  logistic_predictions_poly <- logistic_predictions_poly %>%
    mutate(.pred_class_threshold = factor(ifelse(.pred_1 > threshold , 1, 0)))
  
  conf_mat(logistic_predictions_poly,
           truth = LILATracts_halfAnd10, estimate = .pred_class_threshold, 
           event_level = "second")%>%
    summary()
  
}



threshold_vec <- seq(0.05, 0.9, 0.01)

metric_logis_all <- map_dfr(threshold_vec, ~ function(.x){
  logistic_predictions_poly <- logistic_predictions_poly %>%
    mutate(.pred_class_threshold = factor(ifelse(.pred_1 > .x, 1, 0)))
  
  conf_mat(logistic_predictions_poly,
           truth = LILATracts_halfAnd10, estimate = .pred_class_threshold, 
           event_level = "second")%>%
    summary()
  
}) %>%
  mutate(threshold_val = rep(seq(0.05,0.9,0.01), each = 13))


metric_logis_all <- map_dfr(threshold_vec, ~ all_metrics_threshold_poly(.x)) %>%
  mutate(threshold_val = rep(seq(0.05,0.9,0.01), each = 13))




logistic_metric_plot <- metric_logis_all %>%
  filter(.metric == c("accuracy", "precision", "sens", "j_index", "bal_accuracy","recall"
  )) %>%
  ggplot(aes(x = threshold_val, y = .estimate, color = .metric)) +
  geom_line()+
  labs(title = "Logistic Regression: Metric estimates for different threshold values")+
  geom_vline(aes(xintercept = 0.443))+
  geom_vline(aes(xintercept = 0.5))+
  geom_vline(aes(xintercept = 0.35))+
  ggthemes::scale_color_wsj()+
  theme_bw()+
  theme(plot.title = element_text(family = ubun, face = "bold", hjust = 0.5, size = 20))



logistic_test_predictions_poly <- augment(logis_fit_poly, new_data = fara_test,
                                     type.predict = "response")


logistic_test_predictions_poly <- logistic_test_predictions_poly %>%
  mutate(.pred_class_0.44 = factor(ifelse(.pred_0 > 0.44, 0, 1)))

conf_mat(logistic_test_predictions_poly, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44)



write_csv(logistic_test_predictions_poly, 
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/logistic_test_predictions_poly.csv")

write_csv(logistic_predictions_poly, 
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/logistic_predictions_poly.csv")

write_csv(logis_fit_poly %>% extract_fit_parsnip() %>% vip::vi(), 
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/variable_importance_poly.csv")


write_csv(roc_curve(logistic_predictions_poly, truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second"),
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/roc_data_logistic_poly.csv")
####### Threshold_value set to 0.44


####### Variable importance

variable_importance_logistic <- logis_fit_poly %>% extract_fit_parsnip() %>% vip::vip()




#############################################################################################
#############################################################################################
#############################################################################################






#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################







set.seed(101)
fara_test_train_split_lda <- initial_split(model_data, strata = LILATracts_halfAnd10)

lda_fara_train <- training(fara_test_train_split_lda)
lda_fara_test <- testing(fara_test_train_split_lda)



lda_recipe <- 
  recipe(formula = LILATracts_halfAnd10 ~ ., data = model_data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

lda_spec <- 
  discrim_linear(regularization_method = "diagonal") %>% 
  set_mode("classification") %>% 
  set_engine("MASS") 

lda_workflow <- 
  workflow() %>% 
  add_recipe(lda_recipe) %>% 
  add_model(lda_spec) 


lda_fit <- fit(lda_workflow, data = lda_fara_train)
lda_predictions <- augment(lda_fit, new_data = lda_fara_train)


write_csv(lda_predictions,
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/lda_predictions.csv")


lda_all_metrics_threshold <- function(threshold){
  lda_predictions <- lda_predictions %>%
    mutate(.pred_class_threshold = factor(ifelse(.pred_1 > threshold , 1, 0)))
  
  conf_mat(lda_predictions,
           truth = LILATracts_halfAnd10, estimate = .pred_class_threshold, 
           event_level = "second")%>%
    summary()
  
}



threshold_vec <- seq(0.05, 0.9, 0.01)

lda_metric_logis_all <- map_dfr(threshold_vec, ~ lda_all_metrics_threshold(.x)) %>%
  mutate(threshold_val = rep(seq(0.05,0.9,0.01), each = 13))



lda_metric_plot <- lda_metric_logis_all %>%
  filter(.metric == c("accuracy", "precision", "sens", "j_index", "bal_accuracy", "recall"
                      )) %>%
  ggplot(aes(x = threshold_val, y = .estimate, color = .metric)) +
  geom_line()+
  labs(title = "LDA (without PCA): Metric estimates for different threshold values")+
#  geom_vline(aes(xintercept = 0.35))+
  geom_vline(aes(xintercept = 0.36))+
  geom_vline(aes(xintercept = 0.265), color = "grey50")+
  ggthemes::scale_color_wsj()+
  theme_bw()+
  theme(axis.title = element_text(size = 23, family = ubun),
        axis.text = element_text(size = 23, family = ubun),
        legend.position = "right",
        legend.text = element_text(size = 23, family = ubun),
        legend.title =  element_text(size = 23, family = ubun),
        plot.title = element_text(size = 35, family = plex_serif, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 22.5, family = ubun, hjust = 0))


ggsave(lda_metric_plot, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/lda_metric_plot.jpg",
       width = 22, height = 13, units = "cm")



logistic_metric_plot+lda_metric_plot


lda_test_predictions <- augment(lda_fit, new_data = lda_fara_test,
                                type.predict = "response")


lda_test_predictions <- lda_test_predictions %>%
  mutate(.pred_class_0.36 = factor(ifelse(.pred_0 > 0.36, 0, 1)))

write_csv(lda_test_predictions,
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/lda_test_predictions.csv")


write_csv(roc_curve(lda_test_predictions, 
                    truth = LILATracts_halfAnd10, estimate = .pred_1, 
                    event_level = "second"),
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/lda_roc_data.csv")

conf_mat(lda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36)


lda_predictions %>%
  probably::threshold_perf(estimate = .pred_0, truth = LILATracts_halfAnd10,
                           thresholds = seq(0.1, 0.95, by = 0.005)) %>%
  filter(.metric != "distance") %>%
  ggplot(aes(x = .threshold, y  = .estimate, color = .metric)) +
  geom_line()+
  theme_bw()



roc_curve_lda<- roc_curve(lda_test_predictions, 
                               truth = LILATracts_halfAnd10, estimate = .pred_1, 
                               event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity))+
  geom_smooth(stat="identity")+
  annotate("curve",curvature=0,
           x=0,xend=1,y=0,yend=1,linetype="dotted")+
  
  labs(x="1 - Specificity", y="Sensitivity"
       #       subtitle="Area under this ROC curve is about 0.949 which suggests that the\nLogistic model with aforementioned variables provides a good fit for the data")+
  )+  
  hrbrthemes::theme_tinyhand()+
  theme(axis.text.x  = element_text(family=not,size=21),
        axis.text.y  = element_text(family=not,size=21),
        axis.title.x = element_text(family=not,size=24),
        axis.title.y = element_text(family=not,size=24))

ggsave(roc_curve_lda, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/roc_curve_lda.jpg",
       width = 22, height = 13, units = "cm")



############################################################################################
############################################################################################






lda_recipe2 <- 
  recipe(formula = LILATracts_halfAnd10 ~ ., data = model_data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), num_comp = 10)

lda_spec <- 
  discrim_linear(regularization_method = "diagonal") %>% 
  set_mode("classification") %>% 
  set_engine("MASS") 

lda_workflow2 <- 
  workflow() %>% 
  add_recipe(lda_recipe2) %>% 
  add_model(lda_spec) 


lda_fit2 <- fit(lda_workflow2, data = lda_fara_train)
lda_predictions2 <- augment(lda_fit2, new_data = lda_fara_train)

lda_all_metrics_threshold2 <- function(threshold){
  lda_predictions2 <- lda_predictions2 %>%
    mutate(.pred_class_threshold = factor(ifelse(.pred_1 > threshold , 1, 0)))
  
  conf_mat(lda_predictions2,
           truth = LILATracts_halfAnd10, estimate = .pred_class_threshold, 
           event_level = "second")%>%
    summary()
  
}



threshold_vec <- seq(0.05, 0.9, 0.01)

lda_metric_logis_all2 <- map_dfr(threshold_vec, ~ lda_all_metrics_threshold2(.x)) %>%
  mutate(threshold_val = rep(seq(0.05,0.9,0.01), each = 13))


lda_metric_plot2 <- lda_metric_logis_all2 %>%
  filter(.metric == c("accuracy", "precision", "sens", "j_index", "bal_accuracy", "recall"
  )) %>%
  ggplot(aes(x = threshold_val, y = .estimate, color = .metric)) +
  geom_line()+
  labs(title = "LDA (with PCA): Metric estimates for different threshold values")+
  #  geom_vline(aes(xintercept = 0.35))+
  geom_vline(aes(xintercept = 0.35))+
  geom_vline(aes(xintercept = 0.265), color = "grey50")+
  ggthemes::scale_color_wsj()+
  theme_bw()+
  theme(axis.title = element_text(size = 23, family = ubun),
        axis.text = element_text(size = 23, family = ubun),
        legend.position = "right",
        legend.text = element_text(size = 23, family = ubun),
        legend.title =  element_text(size = 23, family = ubun),
        plot.title = element_text(size = 35, family = plex_serif, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 22.5, family = ubun, hjust = 0))


ggsave(lda_metric_plot2, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/lda_metric_plot2.jpg",
       width = 22, height = 13, units = "cm")



logistic_metric_plot+lda_metric_plot


lda_test_predictions2 <- augment(lda_fit2, new_data = lda_fara_test,
                                type.predict = "response")


lda_test_predictions2 <- lda_test_predictions2 %>%
  mutate(.pred_class_0.35 = factor(ifelse(.pred_0 > 0.35, 0, 1)))

conf_mat(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_class_0.35)


write_csv(lda_test_predictions2,
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/lda_test_predictions2.csv")


write_csv(lda_predictions2,
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/lda_predictions2.csv")



write_csv(roc_curve(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second"),
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/lda_roc_data2.csv")



roc_curve_lda2<- roc_curve(lda_test_predictions2, 
                          truth = LILATracts_halfAnd10, estimate = .pred_1, 
                          event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity))+
  geom_smooth(stat="identity")+
  annotate("curve",curvature=0,
           x=0,xend=1,y=0,yend=1,linetype="dotted")+
  
  labs(x="1 - Specificity", y="Sensitivity"
       #       subtitle="Area under this ROC curve is about 0.949 which suggests that the\nLogistic model with aforementioned variables provides a good fit for the data")+
  )+  
  hrbrthemes::theme_tinyhand()+
  theme(axis.text.x  = element_text(family=not,size=21),
        axis.text.y  = element_text(family=not,size=21),
        axis.title.x = element_text(family=not,size=24),
        axis.title.y = element_text(family=not,size=24))

ggsave(roc_curve_lda2, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/roc_curve_lda2.jpg",
       width = 22, height = 13, units = "cm")





lda_predictions %>%
  probably::threshold_perf(estimate = .pred_0, truth = LILATracts_halfAnd10,
                           thresholds = seq(0.1, 0.95, by = 0.005)) %>%
  filter(.metric != "distance") %>%
  ggplot(aes(x = .threshold, y  = .estimate, color = .metric)) +
  geom_line()+
  theme_bw()








































############################################################################################
############################################################################################

set.seed(101)
fara_test_train_split_knn <- initial_split(model_data, strata = LILATracts_halfAnd10)

knn_fara_train <- training(fara_test_train_split_knn)
knn_fara_test <- testing(fara_test_train_split_knn)


set.seed(105)
cv_folds_knn <- vfold_cv(knn_fara_train, v = 10)


kknn_recipe <- 
  recipe(formula = LILATracts_halfAnd10 ~ ., data = model_data) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) 
#  step_pca(lawhitehalf:lahisphalfshare)

kknn_spec <- 
  nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn") 

kknn_workflow <- 
  workflow() %>% 
  add_recipe(kknn_recipe) %>% 
  add_model(kknn_spec) 

 knn_grid <- grid_regular(neighbors(range = c(3L, 9L)), levels = 7)


 doParallel::registerDoParallel()
 kknn_tune <-
  tune_grid(kknn_workflow, resamples = cv_folds_knn, grid = knn_grid)


knn_fit <- fit(kknn_workflow, data = knn_fara_train)


knn_predictions <- augment(knn_fit, new_data = knn_fara_test)


accuracy(knn_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class)

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################



set.seed(101)
fara_test_train_split_rf <- initial_split(model_data, strata = LILATracts_halfAnd10)

rf_fara_train <- training(fara_test_train_split_rf)
rf_fara_test <- testing(fara_test_train_split_rf)


set.seed(105)
cv_folds_rf <- vfold_cv(rf_fara_train, v = 10)





ranger_recipe <- 
  recipe(formula = LILATracts_halfAnd10 ~ ., data = model_data) 

ranger_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 650) %>% 
  set_mode("classification") %>% 
  set_engine("ranger") 

ranger_workflow <- 
  workflow() %>% 
  add_recipe(ranger_recipe) %>% 
  add_model(ranger_spec) 

ranger_grid <- tidyr::crossing(mtry = seq(1, 10, by = 1), 
                              min_n = seq(2, 25, 1)) 


doParallel::registerDoParallel()
set.seed(28580)
ranger_tune <-
  tune_grid(ranger_workflow, resamples = cv_folds_rf, grid = ranger_grid)




#######################################################################################
###### RDA




set.seed(101)
fara_test_train_split_rda <- initial_split(model_data, strata = LILATracts_halfAnd10)

rda_fara_train <- training(fara_test_train_split_rda)
rda_fara_test <- testing(fara_test_train_split_rda)

set.seed(105)
cv_folds_rda <- vfold_cv(rda_fara_train, v = 10)




rda_recipe <- 
  recipe(formula = LILATracts_halfAnd10 ~ ., data = model_data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

rda_spec <- 
  discrim_regularized(frac_identity = tune(), frac_common_cov = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("klaR") 

rda_workflow <- 
  workflow() %>% 
  add_recipe(rda_recipe) %>% 
  add_model(rda_spec) 


rda_grid <- tidyr::crossing(frac_identity = seq(0, 1, by = 0.2), 
                               frac_common_cov = seq(0, 1, by = 0.1)) 


ctrl_grid <- control_grid(verbose = T)

doParallel::registerDoParallel()
rda_tune <- 
  tune_grid(rda_workflow, resamples = cv_folds_rda, grid = rda_grid,
            control = ctrl_grid) 



bind_rows(rda_tune$.metrics) %>% 
  bind_cols(fold_id = rep(str_c("Fold_", 1:10, sep = ""), each = 110)) %>%
  filter(.metric == "accuracy")%>%
  ggplot(aes(x = .estimate))+
  geom_point(aes(y = mean(.estimate)), size = 3)+
  geom_segment(aes(x = fold_id, xend = fold_id, y = max(.estimate), yend = min(.estimate)), 
               width = 0.8, color = "midnightblue") +
  theme_bw()




bind_rows(rda_tune$.metrics) %>% 
  bind_cols(fold_id = rep(str_c("Fold_", 1:10, sep = ""), each = 110)) %>%
  write_csv(file = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/rda_tune_metrics.csv")



show_best(rda_tune, metric = "roc_auc")

best_regualarization_tune_rda <- select_best(rda_tune, metric = "roc_auc")

select_by_pct_loss(rda_tune, desc(frac_identity), metric = "roc_auc")


############################### 

## Fitting logistic regression model and predicting probablities


final_rda_wf <- finalize_workflow(rda_workflow, best_regualarization_tune_rda)

rda_fit <- fit(final_rda_wf, data = rda_fara_train)

rda_predictions <- augment(rda_fit, data = rda_fara_train, new_data = rda_fara_train)


rda_predictions %>%
  probably::threshold_perf(estimate = .pred_0, truth = LILATracts_halfAnd10,
                           thresholds = seq(0.1, 0.95, by = 0.005)) %>%
  filter(.metric != "distance") %>%
  ggplot(aes(x = .threshold, y  = .estimate, color = .metric)) +
  geom_line()+
  theme_bw()


### ROC curve

roc_auc(rda_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
        event_level = "second")
autoplot()


roc_curve_rda <- roc_curve(rda_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity))+
  geom_smooth(stat="identity")+
  annotate("curve",curvature=0,
           x=0,xend=1,y=0,yend=1,linetype="dotted")+
  
  labs(tx="1 - Specificity", y="Sensitivity"
       #       subtitle="Area under this ROC curve is about 0.949 which suggests that the\nLogistic model with aforementioned variables provides a good fit for the data")+
  )+  
  hrbrthemes::theme_tinyhand()+
  theme(plot.title = element_blank(),
        plot.subtitle = element_text(family = not, size = 17.5, hjust = 0.5, color = "white"),
        axis.title.x = element_text(family=not,size=12),
        axis.title.y = element_text(family=not,size=12)) 



## Precision

accuracy(rda_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class, 
         event_level = "second") 
autoplot()

#####################################################################################
######################################################################################

## Threshold selection

rda_all_metrics_threshold <- function(threshold){
  rda_predictions <- rda_predictions %>%
    mutate(.pred_class_threshold = factor(ifelse(.pred_1 > threshold , 1, 0)))
  
  conf_mat(rda_predictions,
           truth = LILATracts_halfAnd10, estimate = .pred_class_threshold, 
           event_level = "second")%>%
    summary()
  
}



threshold_vec <- seq(0.05, 0.9, 0.01)

rda_metric_logis_all <- map_dfr(threshold_vec, ~ rda_all_metrics_threshold(.x)) %>%
  mutate(threshold_val = rep(seq(0.05,0.9,0.01), each = 13))


rda_metric_plot <- rda_metric_logis_all %>%
  filter(.metric == c("accuracy", "precision", "sens", "j_index", "bal_accuracy","recall"
  )) %>%
  ggplot(aes(x = threshold_val, y = .estimate, color = .metric)) +
  geom_line()+
  labs(title = "RDA: Metric estimates for different threshold values")+
#  geom_vline(aes(xintercept = 0.443))+
#  geom_vline(aes(xintercept = 0.5))+
  geom_vline(aes(xintercept = 0.36))+
  ggthemes::scale_color_wsj()+
  theme_bw()+
  theme(axis.title = element_text(size = 23, family = ubun),
        axis.text = element_text(size = 23, family = ubun),
        legend.position = "right",
        legend.text = element_text(size = 23, family = ubun),
        legend.title =  element_text(size = 23, family = ubun),
        plot.title = element_text(size = 35, family = plex_serif, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 22.5, family = ubun, hjust = 0))


ggsave(rda_metric_plot, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/rda_metric_plot.jpg",
       width = 22, height = 13, units = "cm")



rda_test_predictions <- augment(rda_fit, new_data = rda_fara_test,
                                     type.predict = "response")


rda_test_predictions <- rda_test_predictions %>%
  mutate(.pred_class_0.36 = factor(ifelse(.pred_0 > 0.36, 0, 1)))

conf_mat(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36)

write_csv(rda_test_predictions,
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/rda_test_predictions.csv")


write_csv(rda_predictions,
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/rda_predictions.csv")

write_csv(roc_curve(rda_test_predictions, 
                    truth = LILATracts_halfAnd10, estimate = .pred_1, 
                    event_level = "second"),
          "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/rda_roc_data.csv")


roc_curve_rda<- roc_curve(rda_test_predictions, 
                               truth = LILATracts_halfAnd10, estimate = .pred_1, 
                               event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity))+
  geom_smooth(stat="identity")+
  annotate("curve",curvature=0,
           x=0,xend=1,y=0,yend=1,linetype="dotted")+
  
  labs(x="1 - Specificity", y="Sensitivity"
       #       subtitle="Area under this ROC curve is about 0.949 which suggests that the\nLogistic model with aforementioned variables provides a good fit for the data")+
  )+  
  hrbrthemes::theme_tinyhand()+
  theme(plot.title = element_text(family=ubun,size=20,hjust=0.2),
        plot.subtitle = element_text(family = not, size = 17.5, hjust = 0.5, color = "white"),
        axis.title.x = element_text(family=not,size=12),
        axis.title.y = element_text(family=not,size=12))

ggsave(roc_curve_rda, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/roc_curve_rda.jpg",
       width = 22, height = 13, units = "cm")






rda_roc_data <- roc_curve(rda_test_predictions, truth = LILATracts_halfAnd10, 
                          estimate = .pred_1, event_level = "second")

pr_auc(rda_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
       event_level = "second")
####### Threshold_value set to 0.35 (high sensitivity and balanced )



roc_curve(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity))+
  geom_smooth(stat="identity")+
  annotate("curve",curvature=0,
           x=0,xend=1,y=0,yend=1,linetype="dotted")+
  
  labs(title="ROC for the logistic regression model",
       x="1 - Specificity", y="Sensitivity"
       #       subtitle="Area under this ROC curve is about 0.949 which suggests that the\nLogistic model with aforementioned variables provides a good fit for the data")+
  )+  
  hrbrthemes::theme_tinyhand()+
  theme(plot.title = element_text(family=ubun,size=20,hjust=0.2),
        plot.subtitle = element_text(family = not, size = 17.5, hjust = 0.5, color = "white"),
        axis.title.x = element_text(family=not,size=12),
        axis.title.y = element_text(family=not,size=12)) 




final_roc <- roc_curve(logistic_test_predictions, 
          truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second") %>% 
  bind_rows(roc_curve(lda_test_predictions, 
          truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second") ) %>%
  bind_rows(roc_curve(lda_test_predictions2, 
          truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second")) %>%
  rbind(roc_curve(rda_test_predictions, 
          truth = LILATracts_halfAnd10, estimate = .pred_1, 
          event_level = "second")) %>%
  mutate(model = c(rep("LR",14350),
                   rep("LDA (without PCA)",14350), 
                   rep("LDA (with PCA)",14350), 
                   rep("RDA", 14340))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = model))+
  geom_smooth(stat="identity")+
  annotate("curve",curvature=0,
           x=0,xend=1,y=0,yend=1,linetype="dotted")+
  
  labs(x="1 - Specificity", y="Sensitivity"
       #       subtitle="Area under this ROC curve is about 0.949 which suggests that the\nLogistic model with aforementioned variables provides a good fit for the data")+
  )+  
  ggthemes::scale_color_wsj()+
  hrbrthemes::theme_tinyhand()+
  theme(plot.title = element_text(family=ubun,size=29,hjust=0.2),
        plot.subtitle = element_text(family = not, size = 26.5, hjust = 0.5, color = "white"),
        axis.title.x = element_text(family=not,size=21),
        axis.title.y = element_text(family=not,size=21))




ggsave(final_roc, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/final_roc.jpg",
       width = 22, height = 13, units = "cm")





#### Threshold selected is 0.44

logistic_test_predictions <- augment(logis_fit, new_data = fara_test,
                                     type.predict = "response")


logistic_test_predictions <- logistic_test_predictions %>%
  mutate(.pred_class_0.44 = factor(ifelse(.pred_0 > 0.44, 0, 1)))

conf_mat(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44)




####### Threshold_value set to 0.44

#########################################################################################


roc












############################# RDA metrics ###############################################

accuracy(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
         event_level = "second") # 0.851

roc_auc(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
        event_level = "second") # 0.938

j_index(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
        event_level = "second") # 0.606

precision(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
          event_level = "second") # 0.897

recall(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
       event_level = "second") # 0.646

sens(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
     event_level = "second") # 0.646

spec(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
     event_level = "second") # 0.960

bal_accuracy(rda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
             event_level = "second") # 0.803




############################ Logistic


accuracy(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44, 
         event_level = "second") # 0.884

roc_auc(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
        event_level = "second") # 0.949

j_index(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44, 
        event_level = "second") # 0.716

precision(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44, 
          event_level = "second") # 0.880

recall(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44, 
       event_level = "second") # 0.772


spec(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44, 
     event_level = "second") # 0.943

bal_accuracy(logistic_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.44, 
             event_level = "second") # 0.858




##################################################################################




accuracy(lda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
         event_level = "second") # 0.853

roc_auc(lda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_1, 
        event_level = "second") # 0.939

j_index(lda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
        event_level = "second") # 0.611

precision(lda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
          event_level = "second") # 0.905

recall(lda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
       event_level = "second") # 0.648


spec(lda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
     event_level = "second") # 0.963

bal_accuracy(lda_test_predictions, truth = LILATracts_halfAnd10, estimate = .pred_class_0.36, 
             event_level = "second") # 0.806



########################################################

accuracy(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_class_0.35, 
         event_level = "second") # 0.853

roc_auc(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_1, 
        event_level = "second") # 0.939

j_index(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_class_0.35, 
        event_level = "second") # 0.611

precision(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_class_0.35, 
          event_level = "second") # 0.905

recall(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_class_0.35, 
       event_level = "second") # 0.648


spec(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_class_0.35, 
     event_level = "second") # 0.963

bal_accuracy(lda_test_predictions2, truth = LILATracts_halfAnd10, estimate = .pred_class_0.35, 
             event_level = "second") # 0.806






logis_fit %>% glance() %>% View()















