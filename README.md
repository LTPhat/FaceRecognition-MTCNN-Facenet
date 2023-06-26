# Face Recognition With MTCNN And FaceNet

`Finished time:` 03/12/2022.

## About the dataset

Videos containing faces of many people at different angles. 

This project dataset includes videos from 14 people so there are 14 target classes: ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc', 'XuanAnh']
            
Run ``extract_img_from_video.py`` to save images capturing from videos to folders which belong to each class using ``cv2.captureVideo()`` of ``OpenCV`` library. Each folder includes 50 images. All results are saved in ``raw_images`` folder for preprocessing steps before feeding to neural networks.

```
python extract_img_from_video.py
```

## MTCNN (Multi-task Cascaded Convolutional Networks)

The MTCNN model consists of 3 separate networks: ``P-Net``, ``R-Net``, ``O-Net``.

### P-net (Proposal Network)

P-Net is used to obtain potential windows and their bounding box regression vectors (coordinates). After that, we employ
non-maximum suppression (NMS) to merge highly overlapped candidates.

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/P-net.png)

### R-net (Refine Network)

The R-net (Refine Network) performs the same steps as the P-net. However, the network still uses a method called padding, assuming the insertion of zero pixels into the missing parts of the bounding box if the bounding box is limit exceeded too compile of image. All bounding boxes will now be resized to 24x24 size, treated as 1 kernel and fed into the R-net. which further rejects a large number of false candidates, performs calibration with bounding box regression, and NMS candidate merge.

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/R-net.png)

### O-net (Output Network)

This stage is similar to the second stage, but in this stage we aim to describe the face in more details. In particular,
the network will output five facial landmarks’ positions which include 4 coordinates of bounding box (out[0]), coordinates of 5 landmark points on the face, including 2 eyes, 1 nose, 2 sides of lips (out[1]) and confidence point of each box (out[2] ). All will be saved into a dictionary with the 3 keys mentioned above. For example:
```sh
{'box': [564, 118, 209, 212], 'confidence': 0.9637771248817444, 'keypoints': {'left_eye': (620, 184), 'right_eye': (718, 180), 'nose': (670, 191), 'mouth_left': (632, 267), 'mouth_right': (714, 266)}}
```

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/O-net.png)

### Example results of 3 stages of MTCNN (image in original paper):

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/example.png)

### Face extracted images of some raw images in the dataset:

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/detect_face.png)

## Facenet

``Embedding Vector:`` A fixed-size vector (often smaller in dimension than normal Feature Vectors), learned during training, and represents a set of features responsible for classifying features in space time. was changed.

``Base Network:`` A convolutional neural network which inputs are images and outputs are embedding vectors containing the features of face of input images. In this repo, we use ``Inceptionresnetv2`` as base network which outputs are 128-dimension embedding vectors. Base Network implementation is showed in ``facenet_artchitecture.py``.

``Triplet Loss:`` Triplet Loss introduces a new formula that includes 3 input values including:
- ``Anchor:`` $x_i^{a}$: Output image of neuron network.
- ``Positive`` $x_i^{p}$: The image of the same person as the anchor.
- ``Negative`` $x_i^{n}$: The image is not the same person as the anchor.

The triplot loss function always takes 3 images as input and in all cases we expect:

$$ || f\left(x_i^{a}\right)-f\left(x_i^{p}\right) ||^{2} + \alpha < || f\left(x_i^{a}\right)-f\left(x_i^{n}\right) ||^{2} $$
            
$||$ is Euclidean, $f\left(x_i^{a}\right)$ is the embedding vector of $x_i^{a}$.

Loss function proposed in original paper:

$$L = \sum \limits^{N}_{i} \left[ || f\left(x_i^{a}\right)-f\left(x_i^{p}\right) ||^{2}- || f\left(x_i^{a}\right)-f\left(x_i^{n}\right) ||^{2}+\alpha\right]$$

## Overall Pipeline 

- Extract frames from original videos and save results in raw_images folder by running ``extract_img_from_video.py``.

- Display some random images from raw_images folder after feeding to MTCNN by running   ``display_images.py``.

- Resize images to ``160x160`` shape to match with input shape of Base Network in this repo, train_test split and save train, test images to folders by running ``preprocessing.py``.

- Load dataset by running ``load.py``, dataset is saved in ``faces-dataset.npz``.

- Feed to facenet and extract embedding vectors by running ``create_embedding.py``. Embedding vectors are saved in ``face-dataset-embedding.npz``.

- Train embedding vectors with ``SVM`` classifier by running ``train.py``.

- Predict one random sample in test set by running ``predict_samples.py``.

- Predict batch of images in test set by running ``predict_batch.py``.

# Some results

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/res1.png)

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/res2.png)

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/res3.png)
## Note
To test with new images, create folder containings all new images, changes link to this new folder and run ``load.py``. After that, run  ``predict_samples.py`` to get prediction for new images.

## Deploy on Streamlit App
Streamlit is an open-source app framework for Machine Learning and Data Science teams.

```sh
streamlit run app.py
```

Some web images and results:

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/web_surface.png)

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/right_predict0.png)

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/right_predict1.png)

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/right_predict.png)

![alt text](https://github.com/LTPhat/FaceRecognition_MTCNN_Facenet/blob/master/Images/wrong_predict.png)

## Comments

### Pros
- Good performence with very high accuracy.

- Acceptable runtime.

- Easy to train.

- Simple source code.

### Cons

- Simple dataset, extracting frames from videos somehow makes train and test images quite close --> Result in high accuracy and sometime unrealistic.

- Accuracy on SVM classifier may reduce with more classes. Applying classifier to embeded vector, we have to retrain the whole classifier when there are new classes in the dataset.

- Face problems when predicting person that is not in dataset, which is sometimes unacceptable in real-life problems, it can be partly solved by setting threshold for predict probability.

### Development orientation

- Get more data with various aspects.

- Apply cosine similarity between embeded vector of test images and those of training images to determine which class that test image belongs to. (Replace SVM classifier).

- Try on more general model (ArcFace, DeepFace ...). 

## References
[1] Joint Face Detection and Alignment using
Multi-task Cascaded Convolutional Networks, https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf.

[2] FaceNet: A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/pdf/1503.03832.pdf.

[3] Mô hình Facenet trong face recognition, https://phamdinhkhanh.github.io/2020/03/12/faceNetAlgorithm.html.

[4] Nhận diện khuôn mặt với mạng MTCNN và FaceNet (Phần 2), https://viblo.asia/p/nhan-dien-khuon-mat-voi-mang-mtcnn-va-facenet-phan-2-bJzKmrVXZ9N.

