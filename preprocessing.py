import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.append("D:\MTCNN")

#Define url
root_url = "D:\MTCNN" #Change root_url according to computer directory
root_raw_img = "D:\MTCNN\Raw_images"
detector = MTCNN()
name_list = ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc','XuanAnh']


def display_image(imgfile):
    ax = plt.gca()
    plt.imshow(imgfile)
    img_array = np.array(imgfile)
    plt.axis("off")
    results = detector.detect_faces(img_array)
    for result in results: 
        print(result)
        if result['confidence'] > 0.9:
            x,y,width,height = result['box'] 
            rect = Rectangle((x,y),width,height,fill = False, color = 'green')
            ax.add_patch(rect)

    for _,value in result['keypoints'].items():
        circle = Circle(value, radius = 2, color = 'red')
        ax.add_patch(circle)
    plt.show()

def face_extract(file_name, target_size = (160, 160)):
    img = Image.open(file_name)
    img_arr = np.asarray(img)
    result = detector.detect_faces(img_arr)
    if len(result) == 0:
        return None, None
    else:
        x1, y1, width, height = result[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img_arr[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(target_size)
        resized_arr = np.asarray(image)
    return image, resized_arr

def split_resized(root_url, resized_face, resized_url, name, test_size):
    #Resized_face is a list of img filename(Ex: a.png)
    #Create train/test folder
    if os.path.exists(root_url + "\\Train") == False:
        os.mkdir(root_url + "\\Train")
    if os.path.exists(root_url + "\\Test") == False:
        os.mkdir(root_url + "\\Test")

    train, test = train_test_split(resized_face,test_size = test_size)
    if os.path.exists(root_url + "\\Train" + "\\{}".format(name)) == False:
        os.mkdir(root_url + "\\Train" + "\\{}".format(name))
    #Save train img
    for train_img in train:
        # Get index of img
        img = Image.open(resized_url+ "\\{}".format(name) + "\\{}".format(train_img))
        img.save(root_url + "\\Train" + "\\{}".format(name) + "\\{}".format(train_img), format = "png")

    if os.path.exists(root_url + "\\Test" + "\\{}".format(name)) == False:
        os.mkdir(root_url + "\\Test" + "\\{}".format(name))
    #Save test img
    for test_img in test:
        # Get index of img
        img = Image.open(resized_url+"\\{}".format(name) + "\\{}".format(test_img))
        img.save(root_url + "\\Test" + "\\{}".format(name) + "\\{}".format(test_img), format = "png")
def train_test(resized_face):
    train_set, test_set = train_test_split(resized_face, test_size = 0.3)
    return train_set, test_set

def extract_face_fromdir(name_list):
    face_dict = {}
    file_name_list = {}
    # Create dictionary to store resized img
    for name in name_list:
        face_dict[name] = []
        file_name_list[name] = []
    for name in name_list:
        path = root_raw_img + "\\{}".format(name)
        for _, img in enumerate(os.listdir(path)):
            img_path = path + "\\{}".format(img)
            file_name = img_path.split("\\")[4]
            file_name_list[name].append(file_name)
            file_img, resized_img = face_extract(img_path)
            if (file_img == None and resized_img == None):
                continue
            face_dict[name].append(resized_img)
            if os.path.exists(root_url + "\\Rezised"+ "\\{}".format(name)) == False:
                os.makedirs(root_url + "\\Rezised"+ "\\{}".format(name))
            file_img.save(root_url + "\\Rezised"+ "\\{}".format(name)+ "\\{}".format(img),format = "png")
    return file_name_list, face_dict  

def train_test_seperate():
    path = "D:\MTCNN\Rezised"
    for name in name_list:
        imgs_path = path + "\\{}".format(name) 
        split_resized(root_url,os.listdir(imgs_path),path, name, test_size =0.3)

def normalize(face_picels):
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    return face_picels

if __name__ == "__main__":
    train_test_seperate()
