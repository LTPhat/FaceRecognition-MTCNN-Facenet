from preprocessing import *
from facenet_architecture import InceptionResNetV2
import numpy as np
facenet = InceptionResNetV2()
path = "facenet_keras_weights.h5"
facenet.load_weights(path)

data = np.load('faces-dataset.npz',allow_pickle=True)
train_X, train_y, test_X, test_y = data['a'], data['b'], data['c'], data['d']

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = np.float32(face_pixels)
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def convert_embedding(train_X, test_X):
    newTrainX = list()
    for face_pixels in train_X:
        embedding = get_embedding(facenet, face_pixels)
        newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)
    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in test_X:
        embedding = get_embedding(facenet, face_pixels)
        newTestX.append(embedding)
    newTestX = np.asarray(newTestX)
    return newTrainX, newTestX


if __name__ == "__main__":
    embed_trainX, embed_testX = convert_embedding(train_X,test_X)
    np.savez_compressed("face-dataset-embedding.npz", a = embed_trainX, b = train_y, c = embed_testX, d = test_y)
    print("Saved Embedding....")
    print("Embed train X: {}".format(embed_testX.shape))
    print("Train_y shape: {}".format(train_y.shape))
