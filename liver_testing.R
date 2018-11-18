---
  # Title:
  # Objective:
---
 
# Required packages:
# pkgs <- c("keras", "lime", "tidyquant", "rsample", "recipes", "yardstick", "corrr")
# install.packages(pkgs)

library(keras)
library(neuralnet)
library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)

# Get data
data <- read.csv(url('http://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data'),
                 header=F)

# Create outcome variable; V7 is a classifier for training/test data
data$V7 <- ((ifelse(data$V1 > 100, 1,0) +
             ifelse(data$V2 > 115, 1,0) +
             ifelse(data$V3 > 33, 1,0 ) +
             ifelse(data$V4 > 40, 1,0 ) +
             ifelse(data$V5 > 60, 1,0 ) +
             ifelse(data$V6 >= 4, 1,0 )
             )/6)

# Modify data
data <- as.matrix(data)

data[, 1:6] <- normalize(data[, 1:6])
data[, 7] <- as.numeric(data[, 7])

# Set random seed
set.seed(7)

# Split data 70/30 for training/testing
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:6]
test <- data[ind==2, 1:6]
trainingtarget <- data[ind==1,7]
testtarget <- data[ind==2, 7]
 # print(training)
 # print(test)
 # print(trainingtarget)
 # print(testtarget)

#Create matrix of outcome variables
# trainlabel <- to_categorical(trainingtarget)
# testlabel <- to_categorical(testtarget)

# Model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(6)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'softmax')

# Compile
model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# Fit Model
history <- model %>%
  fit(training, 
      trainingtarget,
      epoch = 200,
      batch_size = 32,
      validation_split = 0.2)

# Plot loss, val_loss
plot(history)

# Evaluate model
model %>%
  evaluate(test,
           testtarget)

# Predictions
prob <- model %>%
  predict_proba(test)

pred <- model %>%
  predict_classes(test)

table(Predicted = pred, Actual = testtarget)

cbind(prob, pred, testtarget)

# # TEST - Explain with LIME
# explainer <- lime(data[-training,], model, bin_continuous = TRUE, quantile_bins = FALSE)
# explanation <- explain(data[training, ], explainer, n_labels = 1, n_features = 4)
# # Only showing part of output for better printing
# explanation[, 2:9]

# 1. mcv mean corpuscular volume (80-100)
# 2. alkphos alkaline phosphotase (45 to 115 international unit/L)
# 3. sgpt alanine aminotransferase (29 to 33 international unit/L)
# 4. sgot aspartate aminotransferase (10-40)
# 5. gammagt gamma-glutamyl transpeptidase (8-61)
# 6. drinks number of half-pint equivalents of alcoholic beverages drunk per day (>4 per day)
# 7. selector field created by the BUPA researchers to split the data into train/test sets


# Neural Network Visualization... just an example
n <- neuralnet(V7 ~ V1+V2+V3+V4+V5+V6,
               data = data,
               hidden = c(50,10, 5),
               linear.output = F,
               lifesign = 'full',
               rep = 1)

plot(n,
     col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue'
)

