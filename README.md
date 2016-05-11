## Write Up for This Study

###Study Goal
It is to predict/classify the manner in which the 6 participants did the exercise by using data from accelerometers on the belt, forearm, arm, and dumbell. 
### Data
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv 
#### The classification processes with the given sampling data
##### 1. Split data into training and test set as 60% vs. 40%
##### 2. Preprocess the data to identify better predictors
A.	remove variables which do not provide information for model: booking information, 
very high NA ratio variable, near zero variance, high correlated variable (>75%)
B.	transform by "center and scale" for not to impact the modeling results
######   The final data set includes 34 predictors (and one dependent variable) reducing from 
original 160 predictors.
 Number of data samples in training set is 11776 and testing set is 7846.

                     sample training testing final training  
                    ------ -------- ------- --------------  
      sample number  19622    11776    7846          11776
      predictors      160      160     160             35  

##### 3 Model select
    With final 31 predictors, 7 models (rpart, linear Dscriminant Analysis, naive Bayes, SVM, random forester, 
gbm bagging, ) were tried to obtain the best fit model with testing data set. The random forest came out 
as the best accurate model.  The combined model for the models having accuarcy >90% was also studied, 
but its result is not as good as random forester model.  The accuacy data are
    
                rpart_acc   lda_acc   nb_acc   svm_acc    rf_acc   gbm_acc   bag_acc 
                ---------  ---------  -------  --------  --------  --------  --------
       Accuracy 0.4969411  0.5850115  0.206857 0.9130767 0.9847056 0.9121846 0.9573031

###### Random Forester Model is selected as final model to use. Its OOB estimate of error rate equals 1.3%. Additional accuacy infomation follows:

Confusion Matrix and Statistics

       Prediction    A    B    C    D    E
                A 2223   18    2    0    0  
                B    6 1488   20    0    0  
                C    0   11 1340   37    1  
                D    2    0    6 1248   14  
                E    1    1    0    1 1427  

  Overall Statistics
                                          
               Accuracy : 0.9847          
                 95% CI : (0.9817, 0.9873)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9807          
   Mcnemar's Test P-Value : NA              

   Statistics by Class:

                           Class: A Class: B Class: C Class: D Class: E  
      Sensitivity            0.9960   0.9802   0.9795   0.9705   0.9896 
      Specificity            0.9964   0.9959   0.9924   0.9966   0.9995 
      Pos Pred Value         0.9911   0.9828   0.9647   0.9827   0.9979  
      Neg Pred Value         0.9984   0.9953   0.9957   0.9942   0.9977 
      Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838  
      Detection Rate         0.2833   0.1897   0.1708   0.1591   0.1819 
      Detection Prevalence   0.2859   0.1930   0.1770   0.1619   0.1823 
      Balanced Accuracy      0.9962   0.9881   0.9860   0.9835   0.9946


The top 20 important predictors are
magnet_dumbbell_z 100.00
pitch_forearm 82.26
magnet_belt_y 81.00
roll_dumbbell 73.96
roll_forearm 72.30
roll_arm 53.73
gyros_belt_z 53.71
total_accel_dumbbell 50.61
yaw_dumbbell 49.87
gyros_dumbbell_y 49.34
total_accel_belt 48.84
accel_forearm_x 45.75
pitch_dumbbell 44.87
magnet_forearm_z 42.65
magnet_arm_x 42.64
magnet_forearm_y 36.46
magnet_belt_x 36.32
yaw_arm 35.21
magnet_forearm_x 34.87
accel_forearm_z 33.96

##### 4. Tuning the model and prediction:  using Caret Train function only mtry parameter is available 
in caret for tuning. The default control setting was used in this study.
    A. default controlcontrol <- trainControl(method="repeatedcv", number=10, repeats=3)
    B. the rf model was applied to predict/classy the final 20 test samples in quiz and it is 100% accurate. Prediction results are:   B A B A A E D B A A B C B A E E A B B B
 
