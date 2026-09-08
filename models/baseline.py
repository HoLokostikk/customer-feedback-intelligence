from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
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

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_val_vec)
print(classification_report(y_val, preds))





