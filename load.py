from preprocessing import *
import sys
import os
sys.path.append("D:\MTCNN")

root_url = "D:\MTCNN"
root_raw_img = "D:\MTCNN\Raw_images"
detector = MTCNN()
name_list = ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc','XuanAnh']

def load_face(dir):
    faces = list()  #Store img arr
    for filename in os.listdir(dir):
        path = dir + "\\" + filename
        img, resized_img = face_extract(path)
        if img == None and resized_img == None:
            continue
        faces.append(resized_img)
    return faces

def load_dataset(dir):
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + "\\" + subdir  
        faces = load_face(path)
        labels = [subdir for _ in range(len(faces))]
        print('Loaded {} examples for class: {}'.format(len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

if __name__ == "__main__":
    # Change link if we use new dataset
    train_X, train_y = load_dataset('D:\MTCNN\Train')
    print(train_X.shape, train_y.shape)
    # Similarly with new testset
    test_X, test_y = load_dataset('D:\MTCNN\Test')
    print(test_X.shape, test_y.shape)
    # save arrays to one file in compressed format
    np.savez_compressed('faces-dataset.npz', a = train_X, b = train_y, c = test_X, d = test_y)
    print("Train_X shape: {}".format(train_X.shape))
    print("Train_y shape: {}".format(train_y.shape))
    print("Test_X shape: {}".format(test_X.shape))
    print("Test_y shape: {}".format(test_y.shape))

