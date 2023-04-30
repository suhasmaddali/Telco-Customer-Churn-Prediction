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

Below are some recommendations to be followed by the company to reduce churn rate and increase profits based on the plot below. 

* Offer customized and flexible internet plans that cater to the diverse needs and preferences of customers
* Provide a variety of data speeds and usage limits to cater to different usage patterns
* Provide affordable backup options to securely store and access customer data
* Provide education and support to customers regarding the benefits of internet backup
* Offer promotions or discounts to incentivize customers to sign up for internet backup
* Offer value-added services such as bundled offers with other Telco services or complementary software packages
* Improve overall customer support and service to build stronger relationships with customers
* By providing more targeted and flexible options for internet plans, as well as improving overall customer support and service, Telco companies can reduce churn rate and increase profitability.

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

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Churn%20countplot.jpg"/>

Fiber optic connections have a higher mean monthly price than DSL connections, while those who haven't selected either option pay less. This is due to the advanced technology and maintenance required for fiber optic connections. DSL may suffice for those with lower usage needs at a lower cost. To make an informed decision, users should evaluate their individual needs and shop around for the best deal among different providers' pricing packages and promotions.

* __Evaluate the benefits and drawbacks of fiber optic and DSL connections:__ Before making a decision, it is essential to understand the advantages and disadvantages of each option. Factors such as speed, reliability, and cost should be considered when choosing between fiber optic and DSL connections.

* __Assess the specific needs of each individual:__ Different people have different internet usage habits, and it's important to consider their specific needs before making a recommendation. For example, someone who works from home and requires high-speed internet may benefit from a fiber optic connection, whereas someone who only uses the internet occasionally may find DSL sufficient.

* __Encourage people to compare prices and services:__ People should be encouraged to shop around and compare prices and services from different providers. This can help them find the best deal and ensure they are not overpaying for their internet connection.

* __Educate people about data usage:__ Internet users should be educated about how much data they need for their usage. Some people may be paying for more data than they need, which can drive up their monthly charges.

* __Consider other factors that affect monthly charges:__ Monthly charges for internet services can be affected by factors such as bundling, contract length, and promotions. People should be made aware of these factors so they can make informed decisions about their internet service.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/MonthlyCharges%20plot.jpg"/>

Customers who churned had higher monthly charges on average than non-churned customers, with an average of $80 compared to $60 annually. This shows that pricing is a significant factor in customer retention. Providers should offer competitive pricing packages and incentives to retain customers and reduce churn rates.

* **Analyze pricing strategy:** Providers should analyze their pricing strategy and compare it with their competitors. They should ensure their prices are reasonable and competitive in the market.

* **Offer customized pricing packages:** Providers should offer customized pricing packages that cater to individual customer needs. This can help customers feel valued and reduce the likelihood of switching to competitors.

* **Provide incentives:** Providers should offer incentives such as discounts, loyalty rewards, or other perks to retain their customers. This can encourage customers to stay with the provider and reduce the chances of churn.

* **Improve customer service:** Providers should improve their customer service to ensure that customers are satisfied with their service. Customers who receive quality service are more likely to stay with the provider and less likely to churn.

* **Monitor customer satisfaction:** Providers should regularly monitor customer satisfaction through surveys and other feedback channels. This can help identify areas for improvement and address issues before customers become dissatisfied enough to leave.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Images/monthlycharges%20vs%20churn%20boxplot.jpg"/>

The plot indicates that senior citizens generally pay higher monthly charges compared to non-seniors. This could be attributed to the fact that seniors tend to utilize Telco services more than non-seniors, leading to increased usage and subsequently, higher charges.

Below are some ways to ensure that there is reduction in the churn rate of customers. 

* Offer customized Telco plans for senior citizens that cater to their needs and usage patterns
* Provide discounted rates for services commonly utilized by seniors such as voice calls and text messaging
* Offer flexible data plans that meet the individual needs of senior citizens
* Provide personalized customer service and support to senior citizens
* Assist with technology and device-related issues
* Offer special promotions or rewards for long-term customers to build loyalty
* Addressing the unique needs of senior citizens can help reduce churn rate and build stronger relationships with this demographic.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Senior%20citizens%20total%20charges.jpg"/>

Customers who have subscribed to Device Protection plans are paying higher total charges compared to those who did not opt for an internet connection or those who have subscribed to an internet connection but did not choose the device protection plan.

Below are some recommendations that could be followed in order to reduce the churn rate and increase the profits generated. 

* Offer customized Device Protection plans that cater to the individual needs of customers
* Provide different levels of coverage and pricing options to cater to different budgets and usage patterns
* Offer promotions or discounts for bundling Device Protection plans with other Telco services
* Provide better education and support to customers regarding the benefits of Device Protection plans
* Offer proactive device maintenance and troubleshooting services to minimize the need for claims
* Improve overall customer support and service to build stronger relationships with customers
* By providing more targeted and flexible options for Device Protection plans, as well as improving overall customer support and service, Telco companies can reduce churn rate and increase profitability.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Device%20protection%20total%20charges.jpg"/>

Heatmaps show that monthly charges are __strongly linked__ to fiber optic connection and streaming TV services. Long-time customers often choose paperless billing and device protection/online backup plans. To improve customer satisfaction, the company should consider promoting fiber optic connections, bundling internet and streaming TV services, and encouraging paperless billing for new customers. Highlighting device protection and online backup plans can increase customer loyalty.

* __Fiber optic connection:__ Since monthly charges are highly correlated with whether someone has a fiber optic connection or not, it may be beneficial for the company to promote and offer fiber optic connections to their customers. This could attract more customers and potentially increase revenue.

* __Streaming TV services:__ Since whether someone opted for streaming TV services also determines their monthly charge, it may be worthwhile for the company to offer bundled packages that include both internet and streaming TV services. This could potentially increase customer satisfaction and retention.

* __Paperless billing:__ Since long tenure users tend to opt for paperless billing, the company should promote and encourage paperless billing to their customers. This could not only save costs on printing and mailing bills, but also promote environmental sustainability.

* __Device protection plans and online backup plans:__ Since long tenure users tend to take device protection plans and online backup plans, the company should consider offering such plans to their new customers as well. This could potentially increase customer loyalty and retention.

Overall, by understanding the correlations between various features in the dataset, the company can make data-driven decisions and optimize their services to improve customer satisfaction and retention.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Heatmap%20correlation.jpg"/>

__Principal Component Analysis (PCA)__ can be used to determine the optimal number of components necessary for predicting whether customers are likely to churn. By analyzing the below plot, we can conclude that approximately 15 components are sufficient to account for 90% of the variance in the dataset. This approach of selecting a reduced number of features can help to avoid the problem of the "curse of dimensionality", which can arise when working with high-dimensional data.

In summary, PCA enables us to identify the most important components for predicting customer churn, and selecting a smaller number of features can help to overcome the challenges associated with high-dimensional data.

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/PCA%20Plot.jpg"/>

__ML Model Performance:__

We can evaluate the performance of ML models in this section. A list of models that I have tried are presented along with confusion matrix and AUC curves. We can get a good understanding of the model performance by visually inspecting the plots. After this step, we can select the best model to be hyperparameter tuned and deployed in real-time. 

__K Nearest Neighbors (KNN):__ We will take a look at the performance of ML models on the test data. Confusion matrix gives a good representation of the total number of true positives, true negatives, false positives and false negatives. In addition, this can also help determine the total accuracy, precision, recall and f1-scores using formulas. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/K%20neighbors%20classifier%20confusion%20matrix.jpg"/>

The AUC score for the classifier is about 0.76 respectively. It does a decent job of classifying whether customers are going to churn from the service or not. We can also explore other models that can also improve AUC scores even further. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/k%20neighbors%20classifier%20auc%20scores%20new.jpg"/>

__Support Vector Classifier:__ The support vector classifier does a decent job of classifying whether the customers are going to churn or not. The performance is quite equivalent to both the classes of churn and non-churn respectively. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Support%20vector%20classifier.jpg"/>

The AUC of the classifier was about 0.87 respectively. It did a good job of classifying whether customers are going to churn or not. We can also explore a list of other models that could be deployed in real-time with good accuracy, precision, recall and f1-score. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/AUC%20Support%20Vector%20Classifier.jpg"/>

__Logistic Regression:__ Let us explore another simple model called logistic regression that could be deployed easily in real-time. We tend to see a good performance on the test data from the confusion matrix. We can also see how well it does with the AUC metric. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Logistic%20regression%20confusion%20matrix.jpg">

The AUC of the classifier came out to be about 0.86 respectively. This is a good improvement in the performance as compared to the previous models we have experimented and used in determining the customer churn. We can also test other classifiers that are complex and which can better capture the underlying distribution of data. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/AUC%20logistic%20regression.jpg">

__Decision Tree Classifier:__ Looks like there are quite a few misclassifications given by the model. There are quite a few false negatives which can have an impact on the business. This is because the model is not doing well on the classes who are going to churn. But we can also look for a list of other models to determine the best one to be deployed in real-time. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Confusion%20matrix%20decision%20tree%20classifier.jpg"/>

The AUC of the classifier is also quite lower as compared to ther models. In addition, the false negatives are also higher in this model which could possibly lead to loss in the business opportunity. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/AUC%20decision%20tree%20classifier.jpg"/>

The gaussian naive bayes classifier is quite popular and can be used for customer churn prediction. The model does a decent job in classifying whether customers are going to leave or stay in the service. The false negatives are low, which means that it is capable of identifying accurately the customers who can churn. 

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Naive%20bayes%20confusion%20matrix.jpg"/>

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/AUC%20Naive%20bayes.jpg"/>

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Random%20forest%20confusion%20matrix.jpg"/>

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/AUC%20random%20forest.jpg"/>

<img src = "https://github.com/suhasmaddali/Telco-Customer-Churn-Prediction/blob/main/Plots/Xgboost%20confusion%20matrix.jpg"/>

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

