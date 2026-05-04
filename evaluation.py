import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import  roc_curve
import matplotlib.pyplot as plt


def test(X_test, y_test, model):
    
    depression_indices = np.where(y_test==1)[0]
    sample_index = depression_indices[0]
    sample = X_test.iloc[[sample_index]]
    actual = y_test[sample_index]
    
    prediction = model.predict(sample)
    probability = model.predict_proba(sample)
    
    print(f"Input features:\n{sample.to_string()}")
    print(f"\nActual label:    {'Depression' if actual == 1 else 'No Depression'}")
    print(f"Predicted label: {'Depression' if prediction[0] == 1 else 'No Depression'}")
    print(f"Confidence:      {max(probability[0]) * 100:.1f}%")
    
    
def classifiaction_report(y_test, y_pred):
    return classification_report(y_test, y_pred)


#def roc_curve(y_test, y_proba):
    
    #fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    #plt.plot([0, 1], [0, 1], 'k--')

    # Plot tpr against fpr
    #plt.plot(fpr, tpr)
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('ROC Curve for Satisfaction Prediction')
    #plt.show()
    #plt.savefig()