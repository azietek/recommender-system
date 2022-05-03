# Recommender System

## Description of a project

The project represents movie recommender system based on existing user's ratings. 
The recommendation system consists in predicting unknown values based on those we already know. In our case, we want to predict ratings that users would give to specific movies in order to recommend them new titles later.

To predict ratings we are using selected tools such as:
- Singular Value Decomposition (SVD)
- Non-negative Matrix Factorization (NMF)
- Stochastic Gradient Descent (SGD)

The main task of the project is to determine the best algorithm (from the above) along with the best parameters. This is verified by dividing the data into two parts. On the first part of data, we apply algorithms and verify the results by checking the error between the new values and those in the test data. Main program ```recom_system.py``` returns root-mean square error, the smaller the better.

## Content
The repository contains two programs:
- ```csv_convertion.py``` that splits the data into two ```csv``` files: testing (that contains 10% of the data) and training (with 90% of the data). It was already applied on ```ratings.csv``` and returned ```train_ratings.csv``` and ```test_ratings.csv```.
- ```recom_system.py``` 

## Usage
Program takes following parameters:

```python3 recom_system.py --train train_file --test test_file --alg ALG --result result_file```

where
- ```train_file``` is a file with training data
- ```test_file``` is a file with training data
- ```ALG``` is one of the algorithms: ```SVD1, SVD2, NMF, SGD```
- ```result_file``` is a file where final RMSE will be saved
