library(caTools)
library(caret)
library(Amelia)
 
data.dir <- "/Documents/Data\ Science/Kaggle/Bike\ Sharing\ Demand/"
 
train.file <- paste0(data.dir, "train.csv")
test.file <- paste0(data.dir, "test.csv")
 
bk <- read.csv(train.file)
test <- read.csv(test.file)
 
 
#################################################
#Descriptive statistics and cleaning the dataset
################################################
 
str(bk)
summary(bk)
str(test)
summary(test)
 
 
bk['casual'] <- NULL
bk['registered'] <- NULL
 
table(complete.cases(bk)) #all cases are complete
 
#Imputation
#impute <- amelia(x = bk, m = 1, noms = c('holiday', 'workingday', 'weather'), idvars = c('datetime')) #set datetime as an id variable so removed from model
#plot(impute)
#bk <- as.data.frame(impute$imputations[1])
 
str(bk)
 
#splitting the data set
 
 
 
#copying the dataset
bk.copy <- bk
test.copy <- test
 
#deleting incomplete cases 
#bk <- na.omit(bk)
 
#deleting date rows
bk[,1]<- NULL
test[,1] <- NULL
 
 
#dv is count
 
#splitting into testing and training set
 
bk.split <- sample.split(bk$count, SplitRatio = 0.7)
bk.train <- subset(bk, bk.split == TRUE)
bk.test <- subset(bk, bk.split == FALSE)
 
#Checking the split
str(bk.train)
str(bk.test)
summary(bk.train)
summary(bk.test)
 
##############
#Models
##############
 
#Linear Model -- Baseline
lm1 <- lm(count~., data = bk.train)
summary(lm1)
lm1.test.predict <- predict(lm1, newdata = bk.test)
 
 
 
################
#Predictions for testing set
###############
 
table(complete.cases(test))
#all cases are complete
 
lm1.kaggle.predict <- predict(lm1, newdata = test)
str(lm1.kaggle.predict)
 
final.results <- data.frame(lm1.kaggle.predict, test.copy[,1])
 
final.results.copy <- final.results
 
final.results[,1] <- final.results.copy[,2]
final.results[,2] <- final.results.copy[,1]
 
 
colnames(final.results)[1] <- 'datetime'
colnames(final.results)[2] <- 'count'
 
which(final.results$count < 0)
which(is.na(final.results$count))
final.results[which(final.results$count < 0),]$count <- 0
#eliminating the negative variables
 
write.table(final.results, file = paste0(data.dir,'results.csv'), sep = ",", row.names = FALSE, quote = FALSE)
