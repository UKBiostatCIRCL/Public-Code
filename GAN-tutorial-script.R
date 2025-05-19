

###########################
## GAN Tutorial Script ####
###########################

# This script is intended to be a singular walkthrough of the procedures
# described by Mangino and colleagues in their tutorial paper [TITLE].
# This script contains the entirety of the GAN fitting, dataset evaluation,
# and model fitting. If you are using RStudio as your IDE, using the 
# document outline feature in the top right corner or its analogue in the 
# bottom left corner of the script pane will make navigation of the full
# procedure simpler.

# We will use coding conventions that facilitate easy navigation throughout
# this tutorial script. Because of the confidentiality of the source data,
# the paper itself only uses GAN-generated data in the code and output while
# this script has the luxury of consisting ONLY of code. As such, we used the
# original source data here (sans output), but generated data in the paper.




#####################
## Preliminaries ####
#####################

## Set a random seed for the project
set.seed(seed = 8675309)

## Importing libraries

# List of packages to load
pkgs = c("rio",
         "devtools",
         "summarytools",
         "stringr",
         "ggplot2",
         "RColorBrewer",
         "DescTools",
         "rcompanion",
         "tidyr",
         "plyr",
         "MASS",
         "sjPlot",
         "table1",
         "dplyr",
         "Hmisc",
         "sjlabelled",
         "gtsummary",
         "car",
         "caret",
         "naniar",
         "ROCR",
         "psych",
         "RGAN",
         "torch",
         "broom",
         "pROC"
         )

# Load packages
for(p in pkgs){
  suppressPackageStartupMessages( library(p,
                                          quietly = TRUE,
                                          character.only = TRUE) )
}


rm(p, pkgs)


## Importing Source Dataset
dat_source = import(file = "tutorial-dataset.csv")

# Evaluating characteristics of source data
str(dat_source)

# Creating univariate summary report of source data
summarytools::view(
  dfSummary(x = dat_source),
  file = "dat_source-raw_summary.html"
)





########################
## Data Preparation ####
########################

# Because this dataset has already been pre-processed and your data will likely
# differ substantially from ours, we will not conduct the full data cleaning
# procedure here. However, note that we do convert our two categorical
# variables to numeric here and assess our data to ensure no missingness: These
# are two important steps that MUST be completed prior to any other steps.


# Converting categorical variables to numeric
dat_source2 = dat_source %>%
  mutate(
    sex = case_when(
      sex == "Male" ~ 1,
      sex == "Female" ~ 0,
      TRUE ~ NA_real_
    ),
    
    diagnosis = case_when(
      diagnosis == "acs" ~ 0,
      diagnosis == "tts" ~ 1,
      TRUE ~ NA_real_
    )
  ) %>%
  
  # Setting all variables in dataset to numeric
  mutate_all(
    as.numeric
  )


## Assessing Missingness

# All cells
is.na(dat_source2) %>% table()

# Visualize missingness
naniar::vis_miss(dat_source2)

# Case-level missingness
naniar::miss_case_table(dat_source2)




#######################################
## Fitting the Transformer and GAN ####
#######################################

## Initialize the transformer
transformer = data_transformer()

## Prepare the source data for input to the transformer fitting function

# Note that there is some redundancy here. We transform the dataset into a matrix,
# then back-transform it into a dataframe in the transformer fitting function. In
# our experience, this is the only functional workflow, as simply inputting the
# original dataframe to the transformer fitting function repeatedly threw errors.
# Your mileage may vary.

dat_source2 = as.matrix(dat_source2)

colnames(dat_source2) = c(
  "mrn", 
  "sex", 
  "age", 
  "inferior_hinge", 
  "anterior_hinge", 
  "ahp_ihp_ratio2", 
  "diagnosis"
)

# Alternatively, the re-naming can be done with the following function:
# colnames(dat_source2) = dput(names(dat_source))

# We specified this manually for this tutorial to illustrate the full process.


## Fit the transformer
transformer$fit(
  as.data.frame(dat_source2),
  discrete_columns = c(
    "sex",
    "diagnosis"
  )
)

transformer_dat = transformer$transform(dat_source2)

# view transformed dataframe
transformer_dat

# Ensure categorical variables are consistent
table(
  transformer_dat[,3],
  transformer_dat[,9]
)

table(
  transformer_dat[,2],
  transformer_dat[,8]
)

table(
  dat_source$sex,
  dat_source$diagnosis
)





######################################################
## Specify G and D Network, and GAN Architectures ####
######################################################


## Specifying G and D architectures ####

## Discriminator architecture
d_network = 
  Discriminator(data_dim = ncol(transformer_dat),
                hidden_units = list(128, 128),
                dropout_rate = 0.5,
                sigmoid = FALSE)


## Generator architecture
g_network = 
  Generator(noise_dim = 5,
            data_dim = ncol(transformer_dat),
            hidden_units = list(128, 128),
            dropout_rate = 0.5)


## Fitting the GAN
gan = gan_trainer(data = transformer_dat,             # transformed dataset
                  generator = g_network,              # Generator network input
                  discriminator = d_network,          # Discriminator network input
                  noise_dim = 5,                      # dimensions of "noise" data/random error
                  noise_distribution = "normal",      # distribution of "noise" data/random error
                  value_function = "wasserstein",     # type of loss function
                  data_type = "tabular",              # type of data
                  base_lr = 0.0001,                   # GAN learning rate
                  ttur_factor = 8,                    # Multiplier for learning rate
                  weight_clipper = NULL,              # Wasserstein GAN limits on D network weights
                  batch_size = 35,                    # Number of training samples included in minibatch for training
                  epochs = 400,                       # Total number of training cycles
                  plot_progress = TRUE,               # Plot data points periodically
                  plot_interval = 50,                 # How often to plot data points
                  eval_dropout = TRUE,                # Drop cases when sampling from synthetic data?
                  synthetic_examples = nrow(dat)*10,  # Number of synthetic cases to generate
                  plot_dimensions = c(5, 6),          # Columns in data to plot
                  device = "cpu")                     # On which device training should be done


## Fitting the GAN
gan = gan_trainer(
  # transformed dataset
  data = transformer_dat,
  # Generator network input
  generator = g_network,        
  # Discriminator network input
  discriminator = d_network,    
  # dimensions of "noise" data/random error
  noise_dim = 5,                      
  # distribution of "noise" data/random error
  noise_distribution = "normal",      
  # type of loss function
  value_function = "wasserstein",     
  # type of data
  data_type = "tabular",         
  # GAN learning rate
  base_lr = 0.0001,                   
  # Multiplier for learning rate
  ttur_factor = 8,                    
  # Wasserstein GAN limits on D network weights
  weight_clipper = NULL,              
  # Number of training samples included in minibatch for training
  batch_size = 35,                    
  # Total number of training cycles
  epochs = 400,                       
  # Plot data points periodically
  plot_progress = TRUE,              
  # How often to plot data points
  plot_interval = 50,                 
  # Drop cases when sampling from synthetic data?
  eval_dropout = TRUE,                
  # Number of synthetic cases to generate
  synthetic_examples = nrow(dat)*10,  
  # Columns in data to plot
  plot_dimensions = c(5, 6),          
  # On which device training should be done
  device = "cpu")                     


## Extracting generated data
dat_gan = sample_synthetic_data(
  gan,
  transformer = transformer
)

## Convert synthetic data to dataframe
dat_gan = as.data.frame(dat_gan)

## Harmonizing Patient sex and diagnosis variables with source data
dat_gan$sex = factor(
  dat_gan$sex,
  levels = c("0",
             "1"),
  labels = c("Female",
             "Male")
)


dat_gan$diagnosis = factor(
  dat_gan$diagnosis,
  levels = c("0",
             "1"),
  labels = c("ACS",
             "TTS")
)

## Create flag variables for real v synthetic data
dat_source$source = rep("Real",
                        nrow(dat_source))


dat_gan$source = rep("GAN",
                     nrow(dat_gan))



complete_data = rbind(dat_source, dat_gan)





#########################################
## Assessing Generated v Source Data ####
#########################################

# This process is lengthy and will differ by your exact dataset, variable types,
# and the nature of the topographies of annd relationships among variables in
# your data. We detail our process here, but acknowledge that this is a limited
# and niche context.


## Extracting univariate summary for generated data
summarytools::view(
  dfSummary(x = dat_gan),
  file = "gan-data_summary.html"
)


## Descriptive Table Stratified by Data Source
complete_data %>%
  select(-mrn) %>%
  tbl_summary(
    by = "source",
    
    type = list(
      all_continuous() ~ "continuous2",
      all_categorical() ~ "categorical"
      
    ),
    statistic = list(
      all_continuous() ~ c(
        "{mean} ({sd})",
        "{min} - {max}"
      ),
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = list(
      all_continuous() ~ 2,
      all_categorical() ~ c(0,2)
    ),
    missing = "no"
  ) %>%
  
  # Add significance tests
  add_p(
    test = list(
      all_continuous() ~ "t.test",
      all_categorical() ~ "chisq.test"
    ),
    pvalue_fun = function(x) {
      if_else(
        is.na(x), 
        NA_character_,
        if_else(x < 0.001, "< 0.001", format(round(x, 3), scientific = FALSE))
      )
    } 
  ) %>% 
  # Bold labels
  bold_labels() %>%
  bold_p()





 ## Assessing Correlations Among Continuous Variables

# Source data
dat_source %>%
  select(
    age,
    inferior_hinge,
    anterior_hinge,
    ahp_ihp_ratio2
  ) %>%
  pairs.panels()


# Generated data
dat_gan %>%
  select(
    age,
    inferior_hinge,
    anterior_hinge,
    ahp_ihp_ratio2
  ) %>%
  pairs.panels()



## Assessing Bivariate Proportions for Categorical Variables

# Source data
dat_source %>%
  select(
    sex,
    diagnosis
  ) %>%
  tbl_cross(
    row = sex,
    col = diagnosis,
    percent = "cell"
  )


# Generated data
dat_gan %>%
  select(
    sex,
    diagnosis
  ) %>%
  tbl_cross(
    row = sex,
    col = diagnosis,
    percent = "cell"
  )





#################################################################
## Fitting and Evaluating Logistic Regression on Source Data ####
#################################################################

# Again, we recognize that this is an extremely limited and niche setting, but
# we showcase our model fitting and evaluation procedure here to provide you
# with a realistic application of this process. Your models will demand 
# different evaluation procedures and, consequently, may look substantially
# different when fit to your generated data than was the case here.

# In this specific case, the model we fit here was published in its entirety
# in Ahmed et al., 2024; https://doi.org/10.1016/j.cpcardiol.2024.102731. 


## Fitting source model
mod_source = glm(diagnosis ~ sex : ahp_ihp_ratio2,
                 data = dat_source,
                 family = binomial(link = "logit"))

summary(mod_source)

tab_model(mod_source)

## Assessing predicted probabilities
predict(mod_source, newdata = dat_source, "response") %>% hist()

predictions_source = predict(mod_source, newdata = dat_source, "response")

## Visualizing predicted probabilities by Patient sex and diagnosis
ggplot(dat_source) +
  geom_point(aes(x = ahp_ihp_ratio2, 
                 y = predictions_source, 
                 color = diagnosis, 
                 shape = sex),
             size = 3) +
  labs(x = "AHP/IHP Ratio",
       y = "Predicted Probability of TTS Diagnosis",
       color = "Clinician\nDiagnosis",
       shape = "Sex") +
  scale_x_continuous(n.breaks = 10) +
  theme_classic(base_size = 24)


## Assessing predicted diagnosis baseline cutpoint (0.5)
confusionMatrix(
  data =  as.factor(
    ifelse(predictions_source > 0.5, 
           "TTS", 
           "ACS")),
  reference = dat_source$diagnosis
)

## Calculating AUC & 95% CI
auc(
  roc(
    dat_source$diagnosis,
    predictions_source
  )
)

auc(
  roc(
    dat_source$diagnosis,
    predictions_source
  )
) %>%
  ci.auc()








####################################################################
## Fitting and Evaluating Logistic Regression on Generated Data ####
####################################################################

# Functionally, you will follow exactly the same procedure and use the same 
# sequence and evaluation metrics for your model fit to generated data. The
# idea is to ensure that your same model specification will yield similar
# results when fit to generated data relative to your source data. In the
# event your model built on generated data looks substantially different than
# that fit to your source data, you must evaluate either the model specification,
# your sample size, bivariate associations, or your GAN hyperparameters.

# We recommend keeping several scripts or other record to ensure you can
# track your GAN hyperparameter specification.


## Fitting and evaluating synthetic model
mod_gan = glm(diagnosis ~ sex : ahp_ihp_ratio2,
                 data = dat_gan,
                 family = binomial(link = "logit"))

summary(mod_gan)

tab_model(mod_gan)

## Assessing predicted probabilities
predict(mod_gan, newdata = dat_gan, "response") %>% hist()

predictions_gan = predict(mod_gan, newdata = dat_gan, "response")

## Visualizing predicted probabilities by Patient sex and diagnosis
dat_gan %>%
  ggplot() +
  geom_point(aes(x = ahp_ihp_ratio2, 
                 y = predictions_gan, 
                 color = diagnosis, 
                 shape = sex),
             size = 3) +
  labs(x = "AHP/IHP Ratio",
       y = "Predicted Probability of TTS Diagnosis",
       color = "Clinician\nDiagnosis",
       shape = "Sex") +
  scale_x_continuous(n.breaks = 10) +
  theme_classic(base_size = 24)


## Assessing predicted diagnosis baseline cutpoint (0.5)
confusionMatrix(
  data =  as.factor(
    ifelse(predictions_gan > 0.5, 
           "TTS", 
           "ACS")),
  reference = dat_gan$diagnosis
)

## Calculating AUC & 95% CI
auc(
  roc(
    dat_gan$diagnosis,
    predictions_gan
  )
)

auc(
  roc(
    dat_gan$diagnosis,
    predictions_gan
  )
) %>%
  ci.auc()







#############################################
## Evaluating Synchrony with Source Data ####
#############################################

AIC(mod_source)
AIC(mod_gan)

BIC(mod_source)
BIC(mod_gan)

PseudoR2(mod_source)
PseudoR2(mod_gan)



gridExtra::grid.arrange(
dat_gan %>%
  ggplot() +
  geom_point(aes(x = ahp_ihp_ratio2, 
                 y = predictions_gan, 
                 color = diagnosis, 
                 shape = sex),
             size = 3) +
  labs(x = "AHP/IHP Ratio",
       y = "Predicted Probability of TTS Diagnosis",
       color = "Clinician\nDiagnosis",
       shape = "Sex") +
  scale_x_continuous(n.breaks = 10) +
  theme_classic(base_size = 24),
ggplot(dat_source) +
  geom_point(aes(x = ahp_ihp_ratio2, 
                 y = predictions_source, 
                 color = diagnosis, 
                 shape = sex),
             size = 3) +
  labs(x = "AHP/IHP Ratio",
       y = "Predicted Probability of TTS Diagnosis",
       color = "Clinician\nDiagnosis",
       shape = "Sex") +
  scale_x_continuous(n.breaks = 10) +
  theme_classic(base_size = 24),
nrow = 1) %>%
  ggsave(filename = "Figure 3.tiff", path = "output/Images for Manuscript", width = 21, height = 9, device='tiff', dpi=800)


