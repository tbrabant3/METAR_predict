# Author: Tyler Brabant & Ashton Allen
# Class:  CSI-270-01
# Certification of Authenticity:
# I certify that this is entirely my own work, except where I have given fully documented
# references to the work of others.  I understand the definition and consequences of
# plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
# assessing this assignment reproduce this assignment and provide a copy to another member
# of academic staff and / or communicate a copy of this assignment to a plagiarism checking
# service(which may then retain a copy of this assignment on its database for the purpose
# of future plagiarism checking).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

# Change this number
SEED = 1


def load_dataset(file):
    dataset = pd.read_csv(file, delimiter=',')
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Transform any fields into their correct data type

    # Drop specific columns that don't matter
    return X, y


def encode(X):
    # Encode the data
    label_encoder = preprocessing.LabelEncoder()
    for col in X.columns:
        X[col] = label_encoder.fit_transform(X[col])


def standard_scale(test, train):
    # Scale / Normalize data
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    return train, test


def main():
    X, y = load_dataset("ENTER DATASET PATH HERE")
    encode(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
    X_train, X_test = standard_scale(X_test, X_train)

    dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)
    # What kind of kind of tree?
    # Recreation


