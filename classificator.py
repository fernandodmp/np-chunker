import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pydotplus

def classify():
    df = pd.read_csv("sintagmas.csv")
    labelEncoder = LabelEncoder()
    df = df.drop(df.columns[0], axis=1)

    y = labelEncoder.fit_transform(df["Tag"])
    x =  pd.get_dummies(df.drop(["Tag"], axis=1), drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    clf = DecisionTreeClassifier() # 0.763671875
    grid = GridSearchCV(clf, {'criterion': ['gini', 'entropy']}, n_jobs=-1, cv=100, verbose=True)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print("Params")
    print(grid.best_params_)
    print("Score")
    print(grid.best_score_)
    model = grid.best_estimator_

    print(classification_report(labelEncoder.inverse_transform(y_test), labelEncoder.inverse_transform(y_pred)))
    export_graphviz(model, feature_names=x.columns.values,  out_file='Tree.dot', filled=True, rounded=True, class_names=labelEncoder.classes_)


if __name__ == '__main__':
    classify()

# print(df.groupby("Anterior-Tag").count())

# print(df.to_string())