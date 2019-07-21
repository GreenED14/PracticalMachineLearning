---
title: 'Project Report: Practical Machine Learning'
author: "Chinmay Sharma"

output:
  html_document: 
    fig_caption: yes
    keep_md: yes
---



##Introduction
Personal health devices like *Jawbone Up, Nike FuelBand* and *Fitbit* are a great source of data for personal activity. These devices give their users a detailed understanding of their health as a result of their daily activity and by setting up goals, it also motivates them on certain exercises to boost their health.The report below analyses similar data obtained from here, by correlating parameters such as roll, pitch and yaw of arm,belt and dumbbell to predict how (well) is the activity performed. This is indicated in the dataset through the variable 'classe'.

##Getting and Cleaning Data
The data can be downloaded from the following links

training data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

test data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data can now be read into R

```r
library(caret)
library(dplyr)
trainData<-read.csv("pml-training.csv")
testData<-read.csv("pml-testing.csv")
dim(trainData)
```

```
## [1] 19622   160
```
 
There are a lot of columns(ie., predictors) in the training dataset. This includes variables such as yaw, roll, pitch, acceleration and gyro acceleration in x,y and z directions for the device, arm, forearm and the dumbbell, which was used in all the excercises. Only the columns with finite values are filtered to be used as predictors.


```r
filteredData<-select(trainData,roll_belt,pitch_belt,yaw_belt,gyros_belt_x,gyros_belt_y,gyros_belt_z,accel_belt_x,accel_belt_y,accel_belt_z,magnet_belt_x,magnet_belt_y,magnet_belt_z,roll_arm,pitch_arm,yaw_arm,gyros_arm_x,gyros_arm_y,gyros_arm_z,accel_arm_x,accel_arm_y,accel_arm_z,magnet_arm_x,magnet_arm_y,magnet_arm_z,roll_dumbbell,pitch_dumbbell,yaw_dumbbell,gyros_dumbbell_x,gyros_dumbbell_y,gyros_dumbbell_z,accel_dumbbell_x,accel_dumbbell_y,accel_dumbbell_z,magnet_dumbbell_x,magnet_dumbbell_y,magnet_dumbbell_z,roll_forearm,pitch_forearm,yaw_forearm,gyros_forearm_x,gyros_forearm_y,gyros_forearm_z,accel_forearm_x,accel_forearm_y,accel_forearm_z,magnet_forearm_x,magnet_forearm_y,magnet_forearm_z,classe)
```

Further, the data has been scaled and centered so that predictors with different units are on the same scale.


```r
PpObject<-preProcess(filteredData[,-49],method=c("center","scale"))
NewSub<-predict(PpObject,filteredData[-49])
NewSub<-mutate(NewSub,classe=filteredData$classe)
```
##Visualization
Below, two featureplots are shown:
- one with yaw, pitch and roll data of the belt
- another with roll values of belt,arm,dumbell and forearm


```r
featurePlot(x=NewSub[,c("roll_belt","pitch_belt","yaw_belt","classe")],y=NewSub$classe,plot="pairs")
```

![](projectSubmission_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
featurePlot(x=NewSub[,c("roll_belt","roll_arm","roll_dumbbell","roll_forearm","classe")],y=NewSub$classe,plot="pairs")
```

![](projectSubmission_files/figure-html/unnamed-chunk-4-2.png)<!-- -->

Looking at the graph, we see little correlation between any of the plotted variables,except that of roll_belt with each of roll_arm, roll_dumbbell, roll_forearm. Therefore, all the selected variables can be used as predictors for the predicting outcome.

##Training Dataset: The methods of Decision Trees,linear discriminant analysis, randomforests and boosting

In the next steps, the data is trained to fit the outcome classe to the rest of the variables in NewSub. Four methods are used:Decision Trees(method="rpart"), randomforests(method="rf"),linear discriminant analysis(method="lda") and  boosting(method="gbm").

```r
set.seed(1234)
modelfitrpart<-train(classe~.,method="rpart",data=NewSub)
modelfitrf<-train(classe~.,method="rf",data=NewSub)
modelfitlda<-train(classe~.,method="lda",data=NewSub)
modelfitgbm<-train(classe~.,method="gbm",data=NewSub)
```

The training accuracy of each can be checked. 

```r
modelfitrpart$results
```

```
##           cp  Accuracy      Kappa AccuracySD    KappaSD
## 1 0.03567868 0.5134012 0.36684327 0.02022139 0.03201790
## 2 0.05998671 0.4317881 0.23542336 0.06643151 0.10976699
## 3 0.11515454 0.3323934 0.07338316 0.03946405 0.06119376
```

```r
modelfitrf$results
```

```
##   mtry  Accuracy     Kappa  AccuracySD     KappaSD
## 1    2 0.9934570 0.9917230 0.001176154 0.001487659
## 2   25 0.9928033 0.9908957 0.001347533 0.001707467
## 3   48 0.9839711 0.9797202 0.003690796 0.004672916
```

```r
modelfitlda$results
```

```
##   parameter  Accuracy     Kappa  AccuracySD     KappaSD
## 1      none 0.6928809 0.6110777 0.004534613 0.005715267
```

```r
modelfitgbm$results
```

```
##   shrinkage interaction.depth n.minobsinnode n.trees  Accuracy     Kappa
## 1       0.1                 1             10      50 0.7500045 0.6830697
## 4       0.1                 2             10      50 0.8527491 0.8134778
## 7       0.1                 3             10      50 0.8953313 0.8675156
## 2       0.1                 1             10     100 0.8166478 0.7679838
## 5       0.1                 2             10     100 0.9054152 0.8803229
## 8       0.1                 3             10     100 0.9401673 0.9242929
## 3       0.1                 1             10     150 0.8499748 0.8101937
## 6       0.1                 2             10     150 0.9291676 0.9103718
## 9       0.1                 3             10     150 0.9597902 0.9491267
##    AccuracySD     KappaSD
## 1 0.008985836 0.011333596
## 4 0.005742937 0.007243497
## 7 0.004066199 0.005130731
## 2 0.005154641 0.006463216
## 5 0.003883898 0.004873597
## 8 0.002839346 0.003576176
## 3 0.004552277 0.005650572
## 6 0.003732787 0.004681683
## 9 0.002366654 0.002971324
```
It is clear that boosting method has the highest accuracy after RandomForest, which has an accuracy close to 1, indicating overfitting.Therefore, boosting is the preferred method for prediction.


```r
predictclasse<-predict(modelfitgbm,newdata=testData)
predictclasse
```

```
##  [1] E E E E A E E E B E B E E A E E E B E E
## Levels: A B C D E
```
