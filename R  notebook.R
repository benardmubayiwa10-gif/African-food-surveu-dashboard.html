# =========================================================
# AFRICAN TRADITIONAL FOOD SURVEY – FULL ANALYTICS PIPELINE
# =========================================================

# ----------------------------
# 0. INSTALL & LOAD LIBRARIES
# ----------------------------

packages <- c(
  "ggplot2","dplyr","tidyr","corrplot","caret",
  "forecast","tidytext","textdata",
  "survival","rstanarm","igraph"
)

installed <- packages %in% rownames(installed.packages())
if(any(!installed)) install.packages(packages[!installed])

library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(caret)
library(forecast)
library(tidytext)
library(textdata)
library(survival)
library(rstanarm)
library(igraph)

set.seed(123)

# ----------------------------
# 1. LOAD DATA
# ----------------------------

# Create synthetic data if file doesn't exist
if(!file.exists("African_Restaurant_Customer_Survey_USA.csv")) {
  n <- 1000
  survey_data <- data.frame(
    Respondent_ID = 1:n,
    Age_Range = sample(c("18-25","26-35","36-45","46-55","56+"), n, TRUE, prob=c(0.2,0.35,0.25,0.15,0.05)),
    Gender = sample(c("Male","Female","Other/Prefer not to say"), n, TRUE, prob=c(0.45,0.50,0.05)),
    Income_Level = sample(c("Low","Middle","High"), n, TRUE, prob=c(0.3,0.5,0.2)),
    Ate_African_Food_Before = sample(c("Yes","No"), n, TRUE, prob=c(0.6,0.4)),
    Familiarity_Level = sample(1:5, n, TRUE, prob=c(0.1,0.2,0.3,0.25,0.15)),
    Importance_Taste = sample(1:10, n, TRUE, prob=c(0.02,0.03,0.05,0.08,0.12,0.15,0.18,0.15,0.12,0.10)),
    Importance_Price = sample(1:10, n, TRUE, prob=c(0.05,0.08,0.12,0.15,0.18,0.15,0.12,0.08,0.05,0.02)),
    Importance_Cleanliness = sample(1:10, n, TRUE, prob=c(0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.15,0.12)),
    Importance_Cultural_Authenticity = sample(1:10, n, TRUE, prob=c(0.05,0.08,0.10,0.12,0.15,0.18,0.15,0.10,0.05,0.02)),
    Importance_Location = sample(1:10, n, TRUE, prob=c(0.08,0.12,0.15,0.18,0.15,0.12,0.08,0.05,0.03,0.02)),
    Interest_in_Visiting = sample(c("Yes","No","Maybe"), n, TRUE, prob=c(0.6,0.15,0.25)),
    Willingness_to_Pay = sample(c("$10-15","$15-20","$20-25","$25-30","$30+"), n, TRUE, prob=c(0.2,0.3,0.25,0.15,0.1))
  )
  write.csv(survey_data, "African_Restaurant_Customer_Survey_USA.csv", row.names=FALSE)
}

survey_data <- read.csv("African_Restaurant_Customer_Survey_USA.csv")

# ----------------------------
# 2. DESCRIPTIVE STATISTICS
# ----------------------------

summary_stats <- survey_data %>%
  summarise(
    Mean_Taste = mean(Importance_Taste),
    Median_Taste = median(Importance_Taste),
    Mean_Price = mean(Importance_Price),
    Median_Price = median(Importance_Price)
  )
print("Summary Statistics:")
print(summary_stats)

# ----------------------------
# 3. DESCRIPTIVE VISUALIZATIONS
# ----------------------------

# Histogram of Taste Importance
ggplot(survey_data, aes(Importance_Taste)) +
  geom_histogram(binwidth=1, fill="steelblue", color="white", alpha=0.8) +
  theme_minimal() +
  labs(title="Histogram of Taste Importance", x="Importance Rating (1-10)", y="Frequency")

# Pie chart of Interest in Visiting
pie_data <- table(survey_data$Interest_in_Visiting)
pie(pie_data, main="Interest in Visiting African Restaurant", 
    col=c("green3","red2","yellow2"),
    labels=paste0(names(pie_data), "\n", pie_data, " (", round(prop.table(pie_data)*100,1),"%)"))

# Box plot of Taste Importance vs Willingness to Pay
ggplot(survey_data, aes(Willingness_to_Pay, Importance_Taste, fill=Willingness_to_Pay)) +
  geom_boxplot(alpha=0.7) +
  theme_minimal() +
  labs(title="Taste Importance vs Willingness to Pay", x="Willingness to Pay", y="Taste Importance") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  scale_fill_brewer(palette="Pastel1")

# ----------------------------
# 4. CUMULATIVE FREQUENCY
# ----------------------------

survey_data %>%
  count(Age_Range) %>%
  arrange(factor(Age_Range, levels=c("18-25","26-35","36-45","46-55","56+"))) %>%
  mutate(Cumulative=cumsum(n)) %>%
  ggplot(aes(Age_Range, Cumulative, group=1)) +
  geom_line(color="darkblue", size=1.5) + 
  geom_point(color="darkblue", size=3) +
  theme_minimal() +
  labs(title="Cumulative Age Distribution", x="Age Range", y="Cumulative Count") +
  geom_area(fill="lightblue", alpha=0.3)

# ----------------------------
# 5. CHI-SQUARE TEST
# ----------------------------

chi_result <- chisq.test(table(
  survey_data$Ate_African_Food_Before,
  survey_data$Interest_in_Visiting
))
print("Chi-Square Test Results:")
print(chi_result)

# ----------------------------
# 6. ONE-WAY ANOVA
# ----------------------------

anova_result <- summary(aov(Importance_Taste ~ Willingness_to_Pay, data=survey_data))
print("One-Way ANOVA Results:")
print(anova_result)

# ----------------------------
# 7. LINEAR REGRESSION
# ----------------------------

lm_model <- lm(
  Importance_Taste ~ Importance_Price +
    Importance_Cleanliness +
    Importance_Cultural_Authenticity,
  data=survey_data
)
print("Linear Regression Results:")
print(summary(lm_model))

# ----------------------------
# 8. LOGISTIC REGRESSION
# ----------------------------

survey_data$Visit_Binary <- ifelse(
  survey_data$Interest_in_Visiting=="Yes",1,0)

log_model <- glm(
  Visit_Binary ~ Familiarity_Level +
    Ate_African_Food_Before +
    Importance_Taste +
    Importance_Price,
  family=binomial,
  data=survey_data
)
print("Logistic Regression Results:")
print(summary(log_model))

# ----------------------------
# 9. PREDICTIVE MODEL (RANDOM FOREST)
# ----------------------------

survey_data$Interest_in_Visiting <- as.factor(survey_data$Interest_in_Visiting)

train_index <- createDataPartition(
  survey_data$Interest_in_Visiting, p=0.7, list=FALSE)

train <- survey_data[train_index, ]
test <- survey_data[-train_index, ]

rf_model <- train(
  Interest_in_Visiting ~ Importance_Taste +
    Importance_Price +
    Importance_Cleanliness +
    Importance_Location,
  data=train,
  method="rf",
  trControl=trainControl(method="cv", number=5)
)

print("Random Forest Model Performance:")
conf_matrix <- confusionMatrix(predict(rf_model, test), test$Interest_in_Visiting)
print(conf_matrix)

# ----------------------------
# 10. BINOMIAL DISTRIBUTION
# ----------------------------

p <- mean(survey_data$Visit_Binary, na.rm=TRUE)
binom_data <- data.frame(x=0:10, prob=dbinom(0:10,10,p))

ggplot(binom_data, aes(x,prob)) +
  geom_col(fill="steelblue", alpha=0.8) +
  theme_minimal() +
  labs(title="Binomial Distribution of Visit Probability", 
       subtitle=paste("p =", round(p,3), "| n = 10"),
       x="Number of Visits (out of 10 customers)", 
       y="Probability") +
  geom_text(aes(label=round(prob,3)), vjust=-0.3, size=3.5)

# ----------------------------
# 11. CORRELATION HEATMAP
# ----------------------------

cor_matrix <- cor(select(survey_data, starts_with("Importance_")))
corrplot(
  cor_matrix,
  method="color", 
  type="upper",
  addCoef.col="black",
  tl.col="black",
  tl.srt=45,
  title="Correlation Matrix of Importance Factors",
  mar=c(0,0,2,0)
)

# ----------------------------
# 12. ARIMA FORECASTING
# ----------------------------

ts_data <- ts(survey_data$Visit_Binary[1:120], frequency=12)
arima_model <- auto.arima(ts_data)
forecast_plot <- plot(forecast(arima_model, h=12), 
                      main="12-Month Forecast of Visit Probability",
                      xlab="Time", 
                      ylab="Visit Probability")

# ----------------------------
# 13. MONTE CARLO SIMULATION
# ----------------------------

mc <- rnorm(10000, mean(p), sd(survey_data$Visit_Binary))
hist(mc, 
     main="Monte Carlo Demand Risk Simulation", 
     xlab="Projected Demand Probability",
     ylab="Frequency",
     col="lightblue",
     border="white",
     breaks=30)
abline(v=mean(p), col="red", lwd=2, lty=2)
legend("topright", legend=c("Mean Demand"), col=c("red"), lty=2, lwd=2)

# ----------------------------
# 14. SENTIMENT ANALYSIS
# ----------------------------

reviews <- data.frame(text=sample(
  c("Amazing food","Too expensive","Loved culture",
    "Not clean","Authentic taste","Great experience",
    "Will come back","Poor service","Excellent atmosphere"),
  300, replace=TRUE))

sentiment_results <- reviews %>%
  unnest_tokens(word,text) %>%
  inner_join(get_sentiments("bing")) %>%
  count(sentiment) %>%
  mutate(percentage = n/sum(n)*100)

ggplot(sentiment_results, aes(sentiment,n, fill=sentiment)) +
  geom_col(alpha=0.8) +
  geom_text(aes(label=paste0(round(percentage,1),"%")), vjust=-0.5) +
  theme_minimal() +
  labs(title="Customer Review Sentiment Analysis", 
       x="Sentiment", 
       y="Count") +
  scale_fill_manual(values=c("positive"="forestgreen", "negative"="firebrick"))

# ----------------------------
# 15. CUSTOMER CLUSTERING
# ----------------------------

clusters <- kmeans(
  scale(select(survey_data,
               Importance_Taste,
               Importance_Price,
               Importance_Cleanliness)), 3)

survey_data$Cluster <- as.factor(clusters$cluster)

ggplot(survey_data,
       aes(Importance_Taste, Importance_Price, color=Cluster, shape=Cluster)) +
  geom_point(size=3, alpha=0.7) +
  theme_minimal() +
  labs(title="Customer Segmentation",
       x="Importance of Taste",
       y="Importance of Price") +
  scale_color_brewer(palette="Set1")

# ----------------------------
# 16. SURVIVAL ANALYSIS (CLV)
# ----------------------------

survey_data$Time <- sample(1:24,nrow(survey_data),TRUE, prob=rev(seq(0.1,1,length.out=24)))
survey_data$Churn <- sample(c(0,1),nrow(survey_data),TRUE, prob=c(0.7,0.3))

surv_fit <- survfit(Surv(Time,Churn)~1, data=survey_data)
plot(surv_fit,
     main="Customer Retention Curve",
     xlab="Time (Months)",
     ylab="Retention Probability",
     col="darkblue",
     lwd=2)
grid()

# ----------------------------
# 17. BAYESIAN A/B TESTING
# ----------------------------

survey_data$Campaign <- sample(c("A","B"), nrow(survey_data), TRUE, prob=c(0.5,0.5))

bayes_model <- stan_glm(
  Visit_Binary ~ Campaign,
  family=binomial,
  data=survey_data,
  prior=normal(0,2.5),
  prior_intercept=normal(0,2.5),
  chains=2,
  iter=1000,
  seed=123
)

print("Bayesian A/B Test Results:")
print(summary(bayes_model))

# ----------------------------
# 18. WHAT-IF SCENARIO ANALYSIS
# ----------------------------

what_if <- survey_data
what_if$Importance_Price <- what_if$Importance_Price + 1

original_prob <- mean(predict(log_model, survey_data, type="response"))
new_prob <- mean(predict(log_model, what_if, type="response"))

print(paste("Original Visit Probability:", round(original_prob*100,1), "%"))
print(paste("New Visit Probability (after price increase):", round(new_prob*100,1), "%"))
print(paste("Change:", round((new_prob-original_prob)*100,1), "percentage points"))

# ----------------------------
# 19. MENU ITEM ANALYTICS
# ----------------------------

survey_data$Menu_Item <- sample(
  c("Jollof Rice","Injera","Suya","Tagine","Fufu","Piri Piri Chicken"),
  nrow(survey_data), TRUE)

menu_plot <- ggplot(survey_data,
                    aes(Menu_Item, fill=Interest_in_Visiting)) +
  geom_bar(position="fill") +
  theme_minimal() +
  labs(title="Menu Preference vs Interest in Visiting",
       x="Menu Item",
       y="Proportion") +
  scale_fill_brewer(palette="Set2") +
  theme(axis.text.x=element_text(angle=45, hjust=1))

print(menu_plot)

# ----------------------------
# 20. NETWORK ANALYSIS
# ----------------------------

network <- graph_from_data_frame(
  data.frame(
    from=c("Supplier A","Supplier B","Supplier C","Supplier D","Supplier A"),
    to=c("Restaurant","Restaurant","Restaurant","Restaurant","Supplier B")
  ),
  directed=FALSE)

V(network)$color <- c("lightblue","orange","lightgreen","pink","yellow")
V(network)$size <- c(30,40,25,25,20)
E(network)$width <- 2

plot(network,
     main="Supplier Relationship Network",
     vertex.label.color="black",
     vertex.label.cex=1.2,
     vertex.frame.color="gray",
     layout=layout_with_fr)

print("Analysis Pipeline Completed Successfully!")