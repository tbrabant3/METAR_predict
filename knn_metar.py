# Author: Ashton Allen
#  Class:  CSI-480-01
#  Certification of Authenticity:
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
from sklearn.neighbors import KNeighborsClassifier


def load_dataset(csv):
    # load in csv
    dataset = pd.read_csv(csv, delimiter=',')

    # drop not important columns
    df = dataset.drop(columns=['station', 'date', 'time', 'ORIGIN', 'DEST'])

    # encode non numeric column
    label_encoder = preprocessing.LabelEncoder()
    df['skyc1'] = label_encoder.fit_transform(df['skyc1'])

    # set up targets
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def scale_data(testing, training):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)

    test = scaler.fit_transform(testing)
    train = scaler.fit_transform(training)

    return test, train


def main():
    accuracy = []

    X, y = load_dataset('jfk_metars.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
    X_test, X_train  = scale_data(X_test, X_train)

    classifier = KNeighborsClassifier(n_neighbors=2, weights="uniform")
    classifier.fit(X_train, y_train)

    print(X_train)

    accuracy.append(classifier.score(X_test, y_test))
    print("Accuracy: " + str(accuracy))

if __name__ == '__main__':
    main()



