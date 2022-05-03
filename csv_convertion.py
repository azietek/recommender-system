import pandas as pd
from sklearn.model_selection import train_test_split

def train_and_test(data, part=0.9):
    """
    :param data: dataframe containing data you are working on
    :param part: which percent of data you want to have in training matrix
    :return: None
    """

    train, test = train_test_split(data, test_size=1 - part, random_state=42)

    train.to_csv('train_ratings.csv', index=False)
    test.to_csv('test_ratings.csv', index=False)

if __name__ == "__main__":
    data = pd.read_csv('ratings.csv')
    train_and_test(data)

