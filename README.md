# ☎️ Telco Customer Churn Prediction

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Problem Statement 

__Telephone__ company, also known as __telco__, is a telecommunications provider that provides services such as __telephony__ and __data communications access__. They are responsible for providing phone services to people in different parts of __The United States of America__. What makes things interesting is that there are a lot of customers who use communication services from Telco. There are customers who also opt for other services such as __TV Streaming__ and __movies streaming__. The telco is at present not able to accurately predict whether a given customer who subscribes to the service is willing to churn (leave the service) or not. If they could know with good accuracy, they would be able to come up with plans and services for those users who are willing to leave the service respectively. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/Telco%20Customer%20Churn%20Background%20Image.jpg" />

## Machine Learning and Data Science 

Since __Telco__ gets a lot of customers who __subscribe__ to their service, it is handy if we are able to predict whether a __customer__ is going to __churn (leave the service)__ within a span of a few days. Furthermore, it would be great if we could consider factors that are __influential__ in the churn of customers such as the __type of billing__, __age__ and whether they have a __partner or not__. After taking a look at these factors and many others which influence customer churn, they might come up with plans that ensure customers do not leave their services.

## Exploratory Data Analysis (EDA)

* Based on the __exploratory data analysis (EDA)__, it was found that the monthly charges for customers are highly correlated with whether the customers opted for a fiber optic connection.
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

## Visualizations

In this section, we focus on the most important **visualizations** that result from exploratory data analysis, feature engineering, and machine learning model predictions. These visualizations can help us to understand the data better and to make better decisions about how to model it.

The image shows the **input** data, highlighting a list of features and attributes that can be used by ML models to make predictions about customer churn. This information can be used to create more accurate models and to improve the performance of our predictions. Note that the image only shows a few set of features. There are more features in the dataset but this is used only for illustration purposes. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/Input%20data.jpg"/>

We can now understand the total number of missing values from each of the features. **Missingno** plots can be used to show a list of missing values from the features. It clearly indicates that there are less number of missing values in the data. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/Missingno%20plot.jpg"/>

The data used for the list of features is **represented** in the figure below, which shows that there are no missing or null values in the data. The features are categorized as float64, int64, and object types, indicating that the data consists of numerical and non-numerical variables.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/Dataset%20Info.jpg"/>

The plot below displays the total number of data points categorized by **gender**. The number of male and female participants appears to be roughly equal, indicating that there is no significant class imbalance. Therefore, metrics such as accuracy can be used with confidence in evaluating the model's performance.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/Gender%20countplot.jpg"/>

The following figure illustrates the **partner** category and provides information on whether the individuals considered for churning from a service had partners or not. The data shows that there are slightly more individuals who do not have partners than those who do, indicating a **slight** class imbalance. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/Partner%20countplot.jpg"/>

The majority of customers appear to opt for **Fiber optic** connections over **DSLs**, with only a small number choosing neither option. It would be interesting to investigate the extent to which each type of connection impacts customer churn behavior.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/Internet%20Service%20countplot.jpg"/>

A vast majority of participants in the survey did not take internet backup. There are some category of people who did not even take internet service as well.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Onlinebackup%20countplot.jpg"/>

The majority of participants chose electronic checks as their preferred payment method, with a smaller portion selecting mailed checks. Additionally, some participants opted for the convenience of automatic credit card payments.

This revision provides a clearer breakdown of the different payment methods chosen by participants and avoids the use of the word "default," which could imply that electronic checks were the only option or the preferred choice for all participants. The revised statement also emphasizes the convenience factor of automatic credit card payments, which could be helpful information for future payment processing considerations.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/PaymentMethod%20countplot.jpg"/>

The information that a vast majority of participants have month-to-month contracts compared to longer term contracts could potentially be used to predict churn in customers. Here are a few ways this information could be helpful in predicting churn:

* Shorter term contracts may indicate that customers are more likely to be price-sensitive and looking for flexibility. Therefore, price changes or lack of flexibility could be potential drivers of churn.

* Month-to-month contracts may suggest that customers are more willing to switch providers or cancel their service. This could increase the risk of churn if competitors offer more attractive alternatives.

* Customers on shorter term contracts may be less committed to the service, and therefore less likely to recommend it to others or refer new customers.

By analyzing these potential drivers of churn and monitoring customer behavior over time, companies could develop strategies to retain customers and reduce churn. For example, they could offer more flexible pricing options, improve customer service, or provide incentives to encourage customers to commit to longer term contracts.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Contract%20countplot.jpg"/>

After taking a look at the countplot and analyzing the results, it could be found that there are a lot of people (majority) in the dataset that did not churn from the service. There are only fewer number of people who have churned. 

Based on the analysis that the majority of the people in the dataset did not churn from the service, it is recommended to focus on retaining the existing customers rather than acquiring new ones. The existing customers are already familiar with the service and have had a positive experience, which makes it easier to keep them engaged.

* __Improve customer service:__ Providing excellent customer service is one of the best ways to retain customers. Ensure that all customers' queries and concerns are addressed promptly and effectively.

* __Offer loyalty programs:__ Loyalty programs can incentivize customers to continue using the service. This can be in the form of discounts, free upgrades, or exclusive access to new features.

* __Personalize communication:__ Personalizing communication with customers can create a stronger bond and make them feel valued. Use their name in emails and address their specific needs and concerns.

* __Provide regular updates and new features:__ Customers appreciate service providers that continually improve their service. Regular updates and new features can help retain customers and attract new ones.

* __Monitor customer feedback:__ Regularly monitor customer feedback to identify issues or concerns and take necessary steps to address their needs.

Fiber optic connections have a higher mean monthly price than DSL connections, while those who haven't selected either option pay less. This is due to the advanced technology and maintenance required for fiber optic connections. DSL may suffice for those with lower usage needs at a lower cost. To make an informed decision, users should evaluate their individual needs and shop around for the best deal among different providers' pricing packages and promotions.

__Evaluate the benefits and drawbacks of fiber optic and DSL connections:__ Before making a decision, it is essential to understand the advantages and disadvantages of each option. Factors such as speed, reliability, and cost should be considered when choosing between fiber optic and DSL connections.

__Assess the specific needs of each individual:__ Different people have different internet usage habits, and it's important to consider their specific needs before making a recommendation. For example, someone who works from home and requires high-speed internet may benefit from a fiber optic connection, whereas someone who only uses the internet occasionally may find DSL sufficient.

__Encourage people to compare prices and services:__ People should be encouraged to shop around and compare prices and services from different providers. This can help them find the best deal and ensure they are not overpaying for their internet connection.

__Educate people about data usage:__ Internet users should be educated about how much data they need for their usage. Some people may be paying for more data than they need, which can drive up their monthly charges.

__Consider other factors that affect monthly charges:__ Monthly charges for internet services can be affected by factors such as bundling, contract length, and promotions. People should be made aware of these factors so they can make informed decisions about their internet service.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Churn%20countplot.jpg"/>

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/MonthlyCharges%20plot.jpg"/>

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

