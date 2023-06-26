from preprocessing import *
import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from train import X_train, y_train, X_test, y_test
# X_train of train.py are emmbeded vectors
from create_embedding import train_X, train_y, test_X, test_y
# train_X of model are images


# Predict a sample from testset
name_list = ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc','XuanAnh']

in_encoder = Normalizer(norm='l2')
file_name = "model\classify.sav"
loaded_model = pickle.load(open(file_name, 'rb'))


def predict_sample():
    """
    Predict a random sample of test set
    """
    print("-------------Random a sample and predict-----------------")
    selection = np.random.choice([i for i in range(X_test.shape[0])])
    random_face_img = test_X[selection]
    random_face_emb = X_test[selection]
    random_face_class = y_test[selection]
    print("Embedded vector of the random sample: ", random_face_emb)
    samples = np.expand_dims(random_face_emb, axis=0)
    # Predict class index
    yhat_class = loaded_model.predict(samples)
    # Class probability vector 
    yhat_prob = loaded_model.predict_proba(samples)
    class_predict = name_list[yhat_class[0]]
    print("Predicted class index:",yhat_class)
    print("Predict probability vector", yhat_prob)
    print("Predict class:", class_predict)
    print("True class: {}".format(random_face_class))
    fig = plt.figure(figsize = (5,5))
    if np.round(np.max(yhat_prob[0]),2) < 0.4:
        plt.imshow(random_face_img)
        plt.title("Predict: {} -- Actual: Unknown ()".format(class_predict))
        plt.axis("off")
        plt.show()
    else:
        plt.imshow(random_face_img)
        plt.title("Predict: {} -- Actual: {} (Confidence: {})".format(class_predict, random_face_class, np.round(np.max(yhat_prob[0]),2)))
        plt.axis("off")
        plt.show()
if __name__ == "__main__":
    predict_sample()