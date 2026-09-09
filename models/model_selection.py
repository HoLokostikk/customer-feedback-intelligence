from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv("../dataset/dataset_clean.csv")

X = df.drop(columns = ["year", "month", "day", "Reviewer Name", "Country"])
y = df["rating_label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size = 0.3, random_state = 42, stratify = y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

vectorizer = TfidfVectorizer(max_features = 5000)
X_train_vec = vectorizer.fit_transform(X_train["Review Text"])
X_val_vec = vectorizer.transform(X_val["Review Text"])

le = LabelEncoder()
y_train_vec = le.fit_transform(y_train)
y_val_vec = le.transform(y_val)

models = {
    "Logistic Regression" : LogisticRegression(max_iter = 1000),
    "LinearSVC" : LinearSVC(dual = False),
    "XGBoost" : XGBClassifier(eval_metric = 'logloss'),
}

for name, model in models.items():
    model.fit(X_train_vec, y_train_vec)
    preds = model.predict(X_val_vec)
    print(name, classification_report(y_val_vec, preds, target_names = le.classes_))





