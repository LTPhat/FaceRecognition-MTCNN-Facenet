from preprocessing import *
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from train import X_train, y_train, X_test, y_test

from create_embedding import train_X, train_y, test_X, test_y
batch_size = 32
file_name = "classify.sav"
loaded_model = pickle.load(open(file_name, "rb"))

random_index = [np.random.randint(0, len(X_test)) for i in range(0, batch_size)]
random_batch_embedding = X_test[random_index]
random_batch_classes = y_test[random_index]

yhat = loaded_model.predict(random_batch_embedding)
predict_class = [name_list[i] for i in yhat]

print("Predict: {}".format(predict_class))
print("Actual: {}".format(random_batch_classes))
correct = np.sum(predict_class==random_batch_classes)
print("Acuracy on batch: {}/{}".format(correct,batch_size))





