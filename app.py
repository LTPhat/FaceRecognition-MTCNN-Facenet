from unicodedata import name
from tensorflow import keras
from keras.preprocessing import image
from PIL import Image
import streamlit as st
import numpy as np
import base64
import warnings
warnings.filterwarnings('ignore')
import pickle
import cv2
from keras.utils import img_to_array, load_img
from preprocessing import *
from create_embedding import get_embedding
from facenet_architecture import InceptionResNetV2

facenet = InceptionResNetV2()
path = "facenet_keras_weights.h5"
facenet.load_weights(path)

name_list = ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc','XuanAnh']
file_name = "classify.sav"
loaded_model = pickle.load(open(file_name, "rb"))


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# Set background for local web
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('Bg1.jpg')

def process_input(filename, target_size = (160,160)):
    """
    Return resized 160x160 image and image file
    """
    img = load_img(filename)
    img_arr = np.array(img)
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

def embed_input(model, resized_arr):
    """
    Convert resized_arr to embedded vector though face_net
    """
    embed_vector = get_embedding(facenet, resized_arr)
    return embed_vector

def predict(model, embed_vector):
    sample = np.expand_dims(embed_vector, axis = 0)
    yhat_index  = model.predict(sample)
    yhat_prob = np.max(model.predict_proba(sample)[0])
    class_predict = name_list[yhat_index[0]]
    return yhat_prob, class_predict
# def catch_face(imgfile):
#     fig = plt.figure()
#     img_load = Image.open(imgfile)
#     img_arr = np.array(img_load, dtype=float)
#     plt.imshow(img_arr)
#     plt.axis("off")
#     results = detector.detect_faces(img_arr)
#     for result in results: 
#         if result['confidence'] > 0.9:
#             x,y,width,height = result['box'] 
#             rect = Rectangle((x,y),width,height,fill = False, color = 'green')
#             fig.add_patch(rect)

#     for _,value in result['keypoints'].items():
#         circle = Circle(value, radius = 2, color = 'red')
#         fig.add_patch(circle)
#     # plt.show()
#     fig.savefig('catchface'+'\{}'.format(imgfile))
#     return imgfile
def main():
    st.markdown("<h2 style='text-align:center; color: yellow;'>Face Recogniton With MTCNN And Facenet</h2>",
                unsafe_allow_html=True)
    html_class_term = """
    <div style="background-color: white ;padding:5px; margin: 20px">
    <h5 style="color:black;text-align:center; font-size: 10 px"> There are 14 classes in the dataset: ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc', 'XuanAnh']</h5>
    """
    st.markdown(html_class_term, unsafe_allow_html=True)
    html_temp = """
       <div style="background-color: brown ;padding:5px">
       <h3 style="color:black;text-align:center; font-size: 15 px"> Click the below button to upload image.</h3>
       </div>
       """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("")
    uploaded_file = st.file_uploader("Choose image file", accept_multiple_files=False)
    if uploaded_file is not None:
        st.write("File uploaded:", uploaded_file.name)
        show_img = load_img(uploaded_file,target_size=(300,300))
        st.image(show_img, caption= "Original image uploaded")
        save_dir = "image_from_user"
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    if st.button("Predict"):
        saveimg_dir = "image_from_user" + "\{}".format(uploaded_file.name)
        image, resized_arr = process_input(saveimg_dir)

        if (image == None and resized_arr == None):
            print("Can't detect face! Please try another image")
        else:
            embed_vector = embed_input(loaded_model, resized_arr)
            prob, pred_class = predict(loaded_model, embed_vector)
            st.success('Predict {} with confidence: {}'.format(pred_class, np.round(prob,4)))
            # catch_face_img = catch_face(saveimg_dir)
            # catchface_dir = "catchface"
            # res_img = Image.open(os.path.join(catchface_dir, catch_face_img))
            # st.image(res_img, caption= "Result")
if __name__=='__main__':
    main()