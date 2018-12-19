import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
import pydotplus

def classify():
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

    labelEncoder = LabelEncoder()
    df = df.drop(df.columns[0], axis=1)
    y = labelEncoder.fit_transform(df["Tag"])
    x =  pd.get_dummies(df.drop(["Tag"], axis=1), drop_first=True)
    #df["Anterior-Tag"] = df["Anterior-Tag"].apply(lambda x: mapping[x])
    #df["POS"] = pd.Categorical(df["POS"]).codes
    #df["Tag"] = df["Tag"].apply(lambda x: mapping[x])

    # clf = Pipeline([
    #     # 0.635
    #     ('clf', GaussianNB())
    # ])
    #
    
    # 0.763671875
    clf = DecisionTreeClassifier()
        
    #
    #clf = Pipeline([
    #     # 0.731
    #   ('clf', KNeighborsClassifier())
    #])
    # clf = Pipeline([
    #     # 0.733
    #     ('clf', SVC())
    # ])
    #clf = Pipeline([
        # 0.745
    #    ('clf', RandomForestClassifier())
    #])

    grid = GridSearchCV(clf, {}, n_jobs=-1, cv=100, verbose=True)
    grid.fit(x, y)
    print("Params")
    print(grid.best_params_)
    print("Score")
    print(grid.best_score_)
    model = grid.best_estimator_
    print(model)
    export_graphviz(model, feature_names=x.columns.values,  out_file='Tree.dot', filled=True, rounded=True, class_names=labelEncoder.classes_)


if __name__ == '__main__':
    classify()

# print(df.groupby("Anterior-Tag").count())

# print(df.to_string())