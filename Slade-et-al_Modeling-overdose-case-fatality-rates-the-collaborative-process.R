
## This script contains the code used for analyses in the following manuscript:
##  Slade E, Mangino AA, Daniels L, Liford M, Quesinberry D. Modeling overdose 
##  case fatality rates over time: The collaborative process. In review.


#####################
### Load Packages ###
#####################

library(tidyverse)
library(brms)
library(broom.mixed)
library(tidybayes)
library(emmeans)
library(tidyquant)


###################
### Import Data ###
###################

# Instructions: Import data as a data frame called 'dat', consisting of columns called 'date', 'deaths', 'ED', and 'EMS'.
#   'date' (class=Date) consists of dates representing the first day of each week of the study period (YYYY-MM-DD).
#   'deaths' represents death certificate counts by week.
#   'ED' represents counts discharged from the emergency department alive by week.
#   'EMS' represents Emergency Medical Service refusals-to-transport by week.

# Create variable representing week number
dat$week <- seq(1, nrow(dat))

# Create variable representing total number of overdose events by week
dat$total <- apply(dat[,c("deaths","ED","EMS")], 1, sum)

# Calculate weekly overdose case fatality rate
dat$ocfr <- dat$deaths/dat$total


#################################
### Exploratory Data Analysis ###
#################################

### Figure 1

# Number of EMS Refusals-to-Transport
ggplot(data = dat, aes(x = date, y = EMS)) +
  geom_line(aes(x = date, y = EMS)) +
  geom_ma(aes(linetype = "12-Week Moving Average"), ma_fun = SMA, n = 12)

# Number of Emergency Department Live Discharges
ggplot(data = dat, aes(x = date, y = ED)) +
  geom_line(aes(x = date, y = ED)) +
  geom_ma(aes(linetype = "12-Week Moving Average"), ma_fun = SMA, n = 12)

# Number of Deaths
ggplot(data = dat, aes(x = date, y = deaths)) +
  geom_line(aes(x = date, y = deaths)) +
  geom_ma(aes(linetype = "12-Week Moving Average"), ma_fun = SMA, n = 12)


### Figure 2

ggplot(data = dat, aes(x = date, y = ocfr)) +
  geom_line(aes(x = date, y = ocfr)) +
  geom_ma(aes(linetype = "12-Week Moving Average"), ma_fun = SMA, n = 12) +
  geom_ma(aes(linetype = "52-Week Moving Average"), ma_fun = SMA, n = 52)


### Figure 3

dat$month <- as.factor(substr(dat$date, 6, 7))
ggplot(dat, aes(x=month, y=ocfr)) +
  geom_boxplot()


### Figure 4

plot(dat$total, dat$ocfr)


##########################################
### Fit Bayesian Beta Regression Model ###
##########################################

# Create sine and cosine predictors
dat$sin_week <- sin((2*pi*dat$week)/52)
dat$cos_week <- cos((2*pi*dat$week)/52)

# Fit Bayesian beta regression model
model_beta_bayes <- brm(bf(ocfr ~ week + sin_week + cos_week, phi ~ 1 + total),
                        data = dat, family = Beta(),
                        chains = 4, iter = 27000, warmup = 2000, cores = 4, backend = "rstan")

# Display model results
beta_bayes_results <- tidy(model_beta_bayes, effects = "fixed")
beta_bayes_results

# Calculate posterior average marginal effects and 95% credible intervals
week_ame <- emtrends(model_beta_bayes, ~ 1, var = "week", transform = "response")
sin_ame <- emtrends(model_beta_bayes, ~ 1, var = "sin_week", transform = "response")
cos_ame  <- emtrends(model_beta_bayes, ~ 1, var = "cos_week", transform = "response")

# Print average yearly increase in overdose case fatality rate
summary(week_ame)$week.trend * 52

# Print amplitude of periodic seasonal trend
sqrt((summary(sin_ame)$sin_week.trend^2) + (summary(cos_ame)$cos_week.trend^2))

# Generate posterior predictive distribution
beta_bayes_pred = model_beta_bayes %>%
  predicted_draws(newdata = dat) %>%
  median_hdi()


### Figure 5

ggplot(beta_bayes_pred) +
  geom_line(aes(x = date, y = ocfr), size = 0.4) +
  geom_line(aes(x = date, y = .prediction), size = 1.6) +
  geom_ribbon(aes(x = date, y = .prediction, ymin = .lower, ymax = .upper), alpha = 0.3)


#########################
### Model Diagnostics ###
#########################

### Figure S1

plot(model_beta_bayes) 


### Figure S2

# Calculate predicted values
pred_dat <- predict(model_beta_bayes, dat, type = "response") %>% as.data.frame()
predicted_ocfr <- pred_dat$Estimate

# Calculate ordinary residuals
ordinary_residuals <- residuals(model_beta_bayes, method="posterior_predict", type="ordinary")[,1]

# Plot of residuals vs. indices of observations
plot(dat$week, ordinary_residuals)

# Plot of residuals vs. linear predictor
logit <- function(x) return(log(x/(1-x)))
plot(logit(predicted_ocfr), ordinary_residuals)

# QQ plot of residuals
qqnorm(ordinary_residuals)
qqline(ordinary_residuals)


### Figure S3

acf(ordinary_residuals)
pacf(ordinary_residuals)



