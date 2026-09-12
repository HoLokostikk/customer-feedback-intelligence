from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


df = pd.read_csv("../dataset/dataset_clean.csv")
df = df[df["rating_label"] != "neutral"].copy()

X = df.drop(columns = ["year", "month", "day", "Reviewer Name", "Country", "rating_label"])
y = df["rating_label"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)


vectorizer = TfidfVectorizer(max_features = 5000)
X_train_vec = vectorizer.fit_transform(X_train["Review Text"])
X_val_vec = vectorizer.transform(X_val["Review Text"])

le = LabelEncoder()

y_train_vec = le.fit_transform(y_train)
y_val_vec = le.transform(y_val)

models = {
    "Logistic Regression" : LogisticRegression(max_iter = 1000),
    "LinearSVC" : LinearSVC(dual = False),
}

X_test_vec = vectorizer.transform(X_test["Review Text"])
y_test_vec = le.transform(y_test)



params_grid_logreg = {
    "C" : [0.01, 0.1, 1, 10, 10],
    "penalty" : ["l1", "l2"],
    "solver" : ["liblinear"],
}

params_grid_svc = {
    "C" : [0.01, 0.1, 1, 10, 100],
    "loss" : ["squared_hinge"],
}

grid_logreg = GridSearchCV(
    estimator = LogisticRegression(max_iter = 1000),
    param_grid = params_grid_logreg,
    cv = 5,
    scoring = "f1_macro",
    n_jobs = -1,
)

grid_svc = GridSearchCV(
    estimator = LinearSVC(dual = False),
    param_grid = params_grid_svc,
    cv = 5,
    scoring = "f1_macro",
    n_jobs =  -1,
)

grid_logreg.fit(X_train_vec, y_train_vec)
grid_svc.fit(X_train_vec, y_train_vec)

cm = confusion_matrix(grid_logreg.best_estimator_.predict(X_test_vec), y_test_vec)
ConfusionMatrixDisplay(cm).plot()
cm1 = confusion_matrix(grid_svc.best_estimator_.predict(X_test_vec), y_test_vec)
ConfusionMatrixDisplay(cm1).plot()
plt.savefig("conf_mat_linsvc.png")

import joblib
joblib.dump(le, "train_data/label_encoder.pkl")
joblib.dump(vectorizer, "train_data/vectorizer.pkl")
joblib.dump(grid_logreg.best_estimator_, "train_data/logreg.pkl")








