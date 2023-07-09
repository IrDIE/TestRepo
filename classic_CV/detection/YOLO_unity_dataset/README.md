
Use pre-trained YOLO-5 model and fine-tune it, to detect and classify interior items from custom dataset.

Example of pictures from Unity:

![alt text](https://github.com/IrDIE/TestRepo/blob/main/classic_CV/detection/YOLO_unity_dataset/step225.camera.png)

![alt text](https://github.com/IrDIE/TestRepo/blob/main/classic_CV/detection/YOLO_unity_dataset/step273.camera.png)



Custom dataset was created in Unity and contain 493 pictures, labels:

* lamp
* light
* picture
* pillow
* plant
* plug
* shair (mistake in labeling =( )
* shelf
* sofa
* table
* tap
* window

Unity provide useful way to converd dataset to COCO format - https://github.com/Unity-Technologies/com.unity.perception

From COCO dataset eas converted to darknet format ( for YOLO )
