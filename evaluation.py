from preprocessing import *
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from train import X_train, y_train, X_test, y_test
import seaborn as sns
from create_embedding import train_X, train_y, test_X, test_y

file_name = "model\classify.sav"
loaded_model = pickle.load(open(file_name, "rb"))

yhat = loaded_model.predict(X_test)     #predict index
y_pred = [name_list[i] for i in yhat]   #take class name


cfm = confusion_matrix(y_pred, y_test) 
print(y_pred)
print(y_test)
print(accuracy_score(y_pred, y_test))
print(cfm)
print(classification_report(y_pred, y_test))

plot = sns.heatmap(data = cfm, annot = True, fmt = ".2g",cmap = "plasma")
plt.title("Confusion Matrix for test set", fontsize = 20)
plt.show()

