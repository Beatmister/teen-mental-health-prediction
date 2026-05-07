import numpy as np
import pandas as pd
import pre_processing
import evaluation
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



mental_health = pd.read_csv("Teen_Mental_Health_Dataset.csv")

processed_data = pre_processing.transform(mental_health)


X = processed_data.drop("depression_label", axis=1)
y = processed_data["depression_label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


models = {
    "Logistic Regression": Pipeline([
        ("smote", SMOTE(sampling_strategy="minority", random_state=42)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
        ]),
    
    "RandomForestClassifier": Pipeline([
        ("smote", SMOTE(sampling_strategy="minority", random_state=42)),
        ("model", RandomForestClassifier(class_weight="balanced"))
    ]),
}

sf = StratifiedKFold(n_splits=6, random_state=12, shuffle=True)

for name, model in models.items():
    cvs_results = cross_val_score(model, X_train, y_train, cv=sf, scoring="f1")
    print(f"{name}: The Mean F1 is: {cvs_results.mean()}, Standard Deviation: {cvs_results.std()}")


# Using RandomForrestClassifier as a model
pipeline = Pipeline([
    ("smote", SMOTE(sampling_strategy="minority",random_state=42)),
    ("model", RandomForestClassifier(random_state=42)),
])

params = [{
    'model__n_estimators':[50, 100, 200, 300],
    'model__max_depth':[3,5,10, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1,2,4],
}]

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=params,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

scores = cross_val_score(
    grid_search.best_estimator_,
    X,
    y,
    cv=sf,
    scoring="f1"
)

print(scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

y_pred = grid_search.predict(X_test)

evaluation.classifiaction_report(y_test, y_pred)
evaluation.test(X_test, y_test,grid_search.best_estimator_)
evaluation.roc_score(X_test, y_test,grid_search.best_estimator_)

print(processed_data.corr()["depression_label"])