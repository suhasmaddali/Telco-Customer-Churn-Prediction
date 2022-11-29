# ☎️ Telco Customer Churn Prediction

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Problem Statement 

__Telephone__ company, also known as __telco__, is a telecommunications provider that provides services such as __telephony__ and __data communications access__. They are responsible for providing the phone services to people in different parts of __The United States of America__. What makes things interesting is that there are a lot of customers who use communication services from Telco. There are customers who also opt for other services such as __TV Streaming__ and __movies streaming__. Telco is at present not able to accurately predict whether a given customer who subscribes to the service is willing to churn (leave the service) or not. If they could know with a good accuracy, they would be able to come up with plans and services to those users who are willing to leave the service respectively. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/Telco%20Customer%20Churn%20Background%20Image.jpg" />

## Machine Learning and Data Science 

Since __Telco__ gets a lot of customers who __subscribe__ to their service, it is handy if we are able to predict whether a __customer__ is going to __churn (leave the service)__ within a span of a few days. Furthermore, it would be great if we could consider factors which are __influential__ in the churn of customers such as the __type of billing__, __age__ and whether they have a __partner or not__. After taking a look at these factors and many others which influence customer churn, they might come up with plans that ensure customers do not leave their services.

## Exploratory Data Analysis (EDA)

* Based on the __exploratory data analysis (EDA)__, it was found that the monthly charges for customers is highly correlated with whether the customers opted for fiber optic connection.
* A large proportion of customers opted for __month-to-month__ contracts rather than __year-long__ or __two-year__ long contracts respectively.
* Monthly charges are correlated with whether a person is a __senior__ or not. Therefore, this gives us a good insight that senior citizens are likely going to be enrolling in other services such as __movies streaming services__ and __internet services__ respectively.
* Based on the plots, it was seen that device protection plans led to a significant increase in the monthly charges as well. 

## Metrics Used

Since the output variable is discrete (0 or 1), it is a __binary classification problem__ with the possibilities being whether a customer is going to __churn__ or __not__. Therefore, the metrics that were considered for the __classification__ problem are as follows.

* [__Accuracy__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [__Logistic Loss__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
* [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
* [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
* [__F1 Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* [__ROC AUC Curves__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
* [__Confusion Matrix__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## Machine Learning Models

There are a large number of machine learning models used in the prediction of __customer churn__ in __Telco__. Below are the models that were used for prediction.

* [__Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [__Gaussian Naive Bayes__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
* [__Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
* [__Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 
* [__Gradient Boosting Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

## Future Scope
* Additional features such as the __location__ of the customer could be added which would also help in determining whether a person is going to stay in the __telco service or not__.
* __More training data__ could be collected to ensure that we get better prediction outcomes. 

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git which could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in the "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link to the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(18).png" width = "600" />

5. The link to the repository can be found when you click on "Code" (Green button) and then, there would be an HTML link just below. Therefore, the command to download a particular repository should be "Git clone HTML" where the HTML is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(20).png" width = "600" />

8. Later, open the Jupyter notebook by writing "Jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 

