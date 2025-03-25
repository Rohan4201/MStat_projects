library(tidyverse)
library(tidymodels)
library(glmnet)

##################################33

sysfonts::font_add_google(name = "Ubuntu", family = "ubuntu")
sysfonts::font_add_google(name = "Noto Serif", family = "noto")
sysfonts::font_add_google("IBM Plex Serif", "Regular 400")


not <- "noto"
ubun <- "ubuntu"
plex_serif <- "Regular 400"

showtext::showtext_auto()


################################################

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





###########################################################################





bootstrap_resamples <- bootstraps(model_data, times = 50, strata = LILATracts_halfAnd10,
                                  apparent = TRUE)






glmnet_recipe <- 
  recipe(formula = LILATracts_halfAnd10 ~ ., data = model_data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_ns(c(lasnaphalf, lasnaphalfshare, TractSNAP, PovertyRate, MedianFamilyIncome))


glmnet_spec <- 
  logistic_reg(penalty = 0.000001, mixture = 0.8) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet") 

glmnet_workflow <- 
  workflow() %>% 
  add_recipe(glmnet_recipe) %>% 
  add_model(glmnet_spec) 


fit_fun <- function(splits) {
  # We could check for convergence, make new parameters, etc.
  fit(glmnet_workflow, data = analysis(splits)) %>%
    tidy()
}




bootstrap_logistic <- bootstrap_resamples %>%
  mutate(model = map(splits, ~ fit_fun(.x)))




bootstrap_estimate_plot <- bootstrap_logistic %>%
  unnest(model) %>%
  group_by(term) %>%
  summarise(estimate_min = min(estimate, na.rm = T),
            estimate_max = max(estimate, na.rm = T),
            estimate_mean = mean(estimate, na.rm = T),
            estimate_sd = sd(estimate, na.rm = T),
            estimate_median = median(estimate, na.rm = T)) %>%
  filter(term != "(Intercept)") %>%
 # filter((estimate_max < 0 & estimate_min < 0) | (estimate_min >0 & estimate_max >0)) %>%
  #View()
  ungroup() %>%
  ggplot(aes(x = reorder(factor(term), abs(estimate_mean))))+
  geom_point(aes(y = estimate_mean), size = 1.5)+
#  geom_point(aes(y = estimate_mean + estimate_sd), color = "midnightblue")+
#  geom_point(aes(y = estimate_mean - estimate_sd), color = "midnightblue")+
  geom_point(aes(y = estimate_min), size = 0.5)+
  geom_point(aes(y = estimate_max), size = 0.5)+
  geom_hline(aes(yintercept = 0), color = "red")+
  labs(y = "Coefficient Estimate",
       x = "Variables")+
  geom_segment(aes(y = estimate_min, yend = estimate_max,
                   xend = factor(term)), color = "grey50",
                size = 0.6)+
  theme_bw()+
  theme(
        axis.title.x = element_text(family=not, size=28),
        axis.title.y = element_text(family=not, size=28),
        axis.text = element_text(family=not, size=25))+
  coord_flip()


ggsave(bootstrap_estimate_plot, width = 25, height = 16, units = "cm", 
       filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/Initial plots/bootstrap_estimate_plot_poly_best_ohu_removed.jpg")  
  





#####################################################################################################









lda_roc_data <- read_csv("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/lda_roc_data.csv")

lda_roc_data2 <- read_csv("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/lda_roc_data2.csv")

logistic_roc_data <- read_csv("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/logistic_roc_data.csv")

rda_roc_data <- read_csv("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/rda_roc_data.csv")

logistic_poly_roc_data <- read_csv("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/roc_data_logistic_poly.csv")





final_roc <- tibble(logistic_roc_data, 
       id = rep("LR", nrow(logistic_roc_data))) %>%
  bind_rows(tibble(logistic_poly_roc_data, 
         id = rep("LR (Poly)", nrow(logistic_poly_roc_data)))) %>%
  bind_rows(tibble(lda_roc_data, 
                   id = rep("LDA", nrow(lda_roc_data)))) %>%
  bind_rows(tibble(lda_roc_data2, 
                   id = rep("LDA (with PCA)", nrow(lda_roc_data2)))) %>%
  bind_rows(tibble(rda_roc_data, 
                   id = rep("RDA", nrow(rda_roc_data)))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id))+
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
        axis.title.x = element_text(family=not,size=23),
        axis.title.y = element_text(family=not,size=23),
        axis.text.x = element_text(family=not,size=23),
        axis.text.y = element_text(family=not,size=23),
        legend.text = element_text(family=not,size=23),
        legend.title = element_blank(),
        legend.background = element_rect(fill = "white"),
        legend.box.background = element_rect(fill = "white"))




ggsave(final_roc, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/Initial plots/final_roc2.jpg",
       width = 21, height = 13, units = "cm")




#####################################################################################################



lda_all_roc <- tibble(lda_roc_data, 
       id = rep("LDA", nrow(lda_roc_data))) %>%
  bind_rows(tibble(lda_roc_data2, 
                   id = rep("LDA (with PCA)", nrow(lda_roc_data2)))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id))+
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
        axis.title.x = element_text(family=not,size=23),
        axis.title.y = element_text(family=not,size=23),
        axis.text.x = element_text(family=not,size=23),
        axis.text.y = element_text(family=not,size=23),
        legend.text = element_text(family=not,size=23),
        legend.title = element_blank(),
        legend.background = element_rect(fill = "white"),
        legend.box.background = element_rect(fill = "white"))




ggsave(lda_all_roc, filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/Initial plots/lda_all_roc.jpg",
       width = 21, height = 13, units = "cm")


  







