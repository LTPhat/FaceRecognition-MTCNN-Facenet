from preprocessing import *
import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from train import X_train, y_train, X_test, y_test
# X_train of train.py are emmbeded vectors
from create_embedding import train_X, train_y, test_X, test_y
# train_X of model are images


#Predict a sample from testset
name_list = ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc','XuanAnh']

in_encoder = Normalizer(norm='l2')

file_name = "classify.sav"
loaded_model = pickle.load(open(file_name, 'rb'))
selection = np.random.choice([i for i in range(X_test.shape[0])])
random_face_img = test_X[selection]
random_face_emb = X_test[selection]
random_face_class = y_test[selection]

samples = np.expand_dims(random_face_emb, axis=0)
yhat_class = loaded_model.predict(samples)
yhat_prob = loaded_model.predict_proba(samples)
class_predict = name_list[yhat_class[0]]
print("True: {}".format(random_face_class))
print(yhat_class)
print(yhat_prob)
print(class_predict)
fig = plt.figure(figsize = (5,5))
if np.round(np.max(yhat_prob[0]),2) < 0.4:
    plt.imshow(random_face_img)
    plt.title("Predict: {} -- Actual: Unknown ()".format(class_predict))
    plt.show()
else:
    plt.imshow(random_face_img)
    plt.title("Predict: {} -- Actual: {} ({})".format(class_predict, np.round(np.max(yhat_prob[0]),2), random_face_class))
    plt.show()
