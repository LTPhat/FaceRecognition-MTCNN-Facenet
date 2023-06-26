import cv2
import numpy as np
import os

def extract_frame():
    name_list = ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc','XuanAnh']
    root_url = "MTCNN\Video" 
    img_folder = "MTCNN\Raw_images\\"
    for name in name_list:
        video = root_url+ "\\" + name + ".mp4"
        vidcap = cv2.VideoCapture(video)
        if os.path.exists(img_folder + name) == False:
            os.mkdir(img_folder + name)
        saved_url = img_folder + name
        count = 50
        leaf = 0
        while vidcap.isOpened:
            success, image = vidcap.read()
            if success and leaf % 3 == 0:
                cv2.imwrite(saved_url +"\\{}{}.png".format(name, count), image)
                count -=1
            leaf += 1
            if count < 0:
                break
if __name__ == "__main__":
    extract_frame()