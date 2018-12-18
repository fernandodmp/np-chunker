import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("sintagmas.csv")

mapping = {
    "BNP": 0,
    "BVP": 1,
    "INP": 2,
    "IVP": 3,
    "ENP": 4,
    "EVP": 5,
    "NONE": 6,
}

df = df.drop(df.columns[0], axis=1)
df["Anterior-Tag"] = df["Anterior-Tag"].apply(lambda x: mapping[x])
df["POS"] = pd.Categorical(df["POS"]).codes
df["Tag"] = df["Tag"].apply(lambda x: mapping[x])

# clf = Pipeline([
#     # 0.635
#     ('clf', GaussianNB())
# ])
#
# clf = Pipeline([
#     # 0.739
#     ('clf', DecisionTreeClassifier())
# ])
#
clf = Pipeline([
    # 0.731
    ('clf', KNeighborsClassifier())
])
# clf = Pipeline([
#     # 0.733
#     ('clf', SVC())
# ])
clf = Pipeline([
    # 0.745
    ('clf', RandomForestClassifier())
])

grid = GridSearchCV(clf, {}, n_jobs=-1, cv=100, verbose=True)

y = df["Tag"]
x = df.drop(["Tag"], axis=1)

grid.fit(x, y)
print("Params")
print(grid.best_params_)
print("score")
print(grid.best_score_)

# print(df.groupby("Anterior-Tag").count())

# print(df.to_string())