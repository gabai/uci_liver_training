library(keras)
library(neuralnet)


data <- read.csv(url('http://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data'),
                 header = FALSE)
data <- as.matrix(data)

data[, 1:6] <- normalize(data[, 1:6])
data[, 7] <- as.numeric(data[, 7]) -1
summary(data)

set.seed(7)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:6]
test <- data[ind==2, 1:6]
trainingtarget <- data[ind==1,7]
testtarget <- data[ind==2, 7]

trainlabel <- to_categorical(trainingtarget)
testlabel <- to_categorical(testtarget)
print(trainlabel)

# Model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(6)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, activation = 'softmax')

# Compile
model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# Fit Model
history <- model %>%
  fit(training, 
      trainlabel,
      epoch = 200,
      batch_size = 32,
      validation_split = 0.2)

plot(history)

#Evaluate model
model %>%
  evaluate(test,
           testlabel)

#Predictions
prob <- model %>%
  predict_proba(test)

pred <- model %>%
  predict_classes(test)

table(Predicted = pred, Actual = testtarget)

cbind(prob, pred, testtarget)

# Neural Network Visualization
n <- neuralnet(V7 ~ V1+V2+V3+V4+V5+V6,
               data = data,
               hidden = c(10,5),
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

