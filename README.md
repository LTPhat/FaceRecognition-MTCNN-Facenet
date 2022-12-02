# Face Recognition With MTCNN And Facenet
Personal project for ``CourseWork 2`` in AI4E course.

## About the dataset

Videos containing faces of many people at different angles. 

This project dataset includes videos from 14 people so there are 14 target classes: ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
            ,'Truong', 'Tuan', 'Van', 'VietDuc', 'XuanAnh']
            
Run ``extract_img_from_video.py`` to save images capturing from videos to folders which belong to each class using ``cv2.captureVideo()`` of ``OpenCV`` library. Each folder includes 50 images. All results are saved in ``raw_images`` folder for preprocessing steps before feeding to neural networks.

```
python extract_img_from_video.py
```

## MTCNN

The MTCNN model consists of 3 separate networks: the P-Net, the R-Net, and the O-Net.

### P-net (Proposal Network)

P-Net is used to obtain potential windows and their bounding box regression vectors (coordinates). After that, we employ
non-maximum suppression (NMS) to merge highly overlapped candidates.


To test with new images, create folder containings all new images, changes link to this new folder and run ``load.py``. After that, run  ``predict_samples.py``.
to get prediction for new images.
