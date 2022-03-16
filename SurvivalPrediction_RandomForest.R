
train=read.csv("/Users/komaluntwal/Documents/NJIT 2021/spring 2021/CS636 data analytics with R/train_titanic.csv")
summary(train)
nrow(train)
ncol(train)
colnames(train)
head(train)

#Since Cabin  has more than 70% of the data unknown , Using it will be pointless. Moreover with such less  data , replacing with mean and medium won't beuseful
#Survival of a person doesn't depends on the location the person embarked his/her journey. Hence omitting this too.

train_refined=train[,-c(11,12)]
colnames(train_refined)

#Replaceing NA's in Age with mean or median

boxplot(train_refined$Age)
hist(train_refined$Age)

#Removing Outliers
age_mean=mean(train_refined$Age,na.rm=TRUE)
train_refined$Age[is.na(train_refined$Age)]=age_mean
summary(train_refined)

#Replaceing NA's in fare with mean or median
boxplot(train_refined$Fare)
hist(train_refined$Fare)

#Removing Outliers
#fare_mean=mean(train_refined$Fare,na.rm=TRUE)
#train_refined$Fare[is.na(train_refined$Fare)]=fare_mean
#summary(train_refined)

#Prediction 
#install.packages("randomForest")
library(randomForest)
colnames(train_refined)
train_refined=train_refined[,c(1,2,3,5,6,7,8)]

nrow(train_refined)
train_data=train_refined[1:70000,]
test_data=train_refined[70001:100000,]
target=test_data[,2]
as.factor(target)->target
test_data=test_data[,-c(2)]
test_data


class(train_data$Survived)
as.factor(train_data$Survived)->train_data$Survived

#'PassengerId' 'Pclass'  'Sex' 'Age' 'SibSp' 'Parch' 'Fare'

model_randomForest=randomForest(Survived ~ PassengerId + Pclass + Sex + Age +  SibSp + Parch, data = train_data,na.action=na.exclude)
model_randomForest

pred = predict(model_randomForest,test_data)
print(pred)
data.frame(target,pred)->accuracy_tab

accuracy_tab$target==accuracy_tab$pred->accuracy_vec
#accuracy_vec
count=0
for(i in 1:nrow(accuracy_tab))
{
  if(accuracy_vec[i]== FALSE)
  {
    count=count+1
  }
}

Error = (count/nrow(accuracy_tab))*100
Accuracy = 100 -Error
paste("Accuracy : ",Accuracy)

### "Accuracy :  75.99"

test=read.csv("/Users/komaluntwal/Documents/R dataset/test_titanic.csv")
summary(test)

test=test[,-c(3,8,9,10,11)]
colnames(test)

test_age_mean=mean(test$Age,na.rm=TRUE)
test$Age[is.na(test$Age)]=test_age_mean
summary(test)

Survived = predict(model_randomForest,test)
test$PassengerId->PassengerId
data.frame(PassengerId,Survived) -> Predicted_Results

write.csv(Predicted_Results,file='survival_prediction.csv',row.names=F)
getwd()
setwd("/Users/komaluntwal/Documents/R dataset")
