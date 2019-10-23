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
from sklearn.metrics import accuracy_score


# Change this number
SEED = 1


def load_dataset(file):
    dataset = pd.read_csv(file, delimiter=',')

    df = dataset.drop(columns=['station', 'date', 'time', 'ORIGIN', 'DEST'])

    label_encoder = preprocessing.LabelEncoder()
    df['skyc1'] = label_encoder.fit_transform(df['skyc1'])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def standard_scale(test, train):
    # Scale / Normalize data
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    return train, test


def main():
    X, y = load_dataset("jfk_metars.csv")
    # encode(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
    # X_train, X_test = standard_scale(X_test, X_train)

    dt = DecisionTreeClassifier(max_depth=8, random_state=SEED)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print("Test accuracy: ", acc)


if __name__ == '__main__':
    main()