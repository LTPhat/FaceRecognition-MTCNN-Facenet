import matplotlib.pyplot as plt
from mtcnn import MTCNN
import os
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
# Examples of extracting face from raw images using MTCNN

detector = MTCNN()
raw_img_url = "D:\MTCNN\Raw_images"
fig = plt.figure(figsize=(20,20))
for img in range(1,6):
    random_name = np.random.choice([i for i in os.listdir(raw_img_url)])
    random_img_url = os.path.join(raw_img_url, random_name)
    random_img_file = np.random.choice([i for i in os.listdir(random_img_url)])
    random_img = Image.open(os.path.join(random_img_url, random_img_file))
    random_img = np.array(random_img)
    ag = plt.subplot(15*10+img)
    results = detector.detect_faces(random_img)
    for result in results: 
        print(result)
        if result['confidence'] > 0.9:
            x,y,width,height = result['box'] 
            rect = Rectangle((x,y),width,height,fill = False, color = 'green')
            ag.add_patch(rect)

    for _,value in result['keypoints'].items():
        circle = Circle(value, radius = 2, color = 'red')
        ag.add_patch(circle)
    ag.imshow(random_img)
    ag.set_title("{}".format(random_name))
    ag.axis("off")
plt.show()
plt.savefig('MTCNN detect face example.png')

