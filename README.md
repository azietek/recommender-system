# Recommender System

## Description of a project

The project represents movie recommender system based on existing user's ratings. 
The recommendation system consists in predicting unknown values based on those we already know. In our case, we want to predict ratings that users would give to specific movies in order to recommend them new titles later.

To predict ratings we are using selected tools such as:
- Singular Value Decomposition (SVD)
- Non-negative Matrix Factorization (NMF)
- Stochastic Gradient Descent (SGD)

The main task of the project is to determine the best algorithm (from the above) along with the best parameters. This is verified by dividing the data into two parts. On the first part of data, we apply algorithms and verify the results by checking the error between the new values and those in the test data. Main program ```recom_system.py``` returns root-mean square error, the smaller the better.
