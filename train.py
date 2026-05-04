import numpy as np
import pandas as pd
import pre_processing
import evaluation
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


mental_health = pd.read_csv("Teen_Mental_Health_Dataset.csv")

processed_data = pre_processing.transform(mental_health)


X = processed_data.drop("depression_label", axis=1)
y = processed_data["depression_label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

logreg = LogisticRegression()
rf = RandomForestClassifier()

models = {
    
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", logreg)
        ]),
    
    "RandomForestClassifier": Pipeline([("model", rf)])
}


kf = KFold(n_splits=6, random_state=12, shuffle=True)

for name, model in models.items():
    cvs_results = cross_val_score(model, X_train_sm, y_train_sm, cv=kf, scoring="f1")
    print(f"{name}: The Mean F1 is: {cvs_results.mean()}")


params = [{
    'n_estimators':[50, 100, 200, 300],
    'max_depth':[3,5,10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1,2,4],
}]

grid_search = GridSearchCV(rf, param_grid=params, cv=5, scoring="f1", n_jobs=-1, verbose=1)
grid_search.fit(X_train_sm, y_train_sm)

print(grid_search.best_score_)
print(grid_search.best_params_)

rf_2 = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=2, min_samples_split=2)

rf_2.fit(X_train_sm, y_train_sm)

y_pred = rf_2.predict(X_test)

evaluation.classifiaction_report(y_test, y_pred)
evaluation.test(X_test, y_test,rf_2)

#y_test = np.where(y_test == 1, 1, 0)
#y_proba = rf_2.predict_proba(X_test)[:, 1]

#evaluation.roc_curve(y_test,y_proba)