import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


dataset = pd.read_csv("jfk_metars.csv", delimiter=',')

df = dataset.drop(columns=['station','date','time','ORIGIN','DEST'])

label_encoder = preprocessing.LabelEncoder()
df['skyc1'] = label_encoder.fit_transform(df['skyc1'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Scale / Normalize data
scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

test = []

# Classifier to setup amount of neighbors and weighting type
classifier = KNeighborsClassifier(n_neighbors=3, weights="uniform")
classifier.fit(X_train, y_train)

test.append(classifier.score(X_test, y_test))

# Use the training data to train the data then predict
print(test)




