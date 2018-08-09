# Logistic Regression Example

This project was initially designed for a tech talk on "Machine learning with Apache Spark".

In this project a Logistic Regression model is being trained and evaluated using the famous Iris Dataset.

You can find the dataset here : https://archive.ics.uci.edu/ml/datasets/iris

Project contains a single class describing,

	1. How to initialize a spark session
	
	2. How to load data in to the application using spark session
	
	3. How to transform raw data in to features for machine learning (for logistic regression)
	
	4. How to make predictions
	
	5. How to evaluate the trained model with test data
	
Additionally application will iteratively listen for comma seperated values for features as command line arguments and predict the class using trained model.
	
** Note	: This project is designed to run in Intellij IDEA with java 1.8 & Scala 2.11.6

And spark master is being set to "Local[*]"

If you are willing to run this on hadoop configured spark
			
	1. You have to edit the data path as "file:///Path/To/iris-data.csv" if the file is in local system
				
	2. Or place the irish-data.csv file in given hdfs path
				
Enjoy!