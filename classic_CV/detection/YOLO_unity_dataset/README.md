
Use pre-trained YOLO-5 model and fine-tune it, to detect and classify interior items from custom dataset.

Example of pictures from Unity:

| Unity exaple 1                                                 | Unity exaple 2             |
| -------------------------------------------------------- | ---------------------- |
|![alt text](https://github.com/IrDIE/TestRepo/blob/main/classic_CV/detection/YOLO_unity_dataset/images/step225.camera.png) |![alt text](https://github.com/IrDIE/TestRepo/blob/main/classic_CV/detection/YOLO_unity_dataset/images/step273.camera.png) |




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

From COCO dataset was converted to darknet format ( for YOLO )

One of validation results:



| Unity bounding boxes                                  | YOLO predicted  boxes         |
| -------------------------------------------------------- | ---------------------- |
|![alt text](https://github.com/IrDIE/TestRepo/blob/main/classic_CV/detection/YOLO_unity_dataset/images/val_batch0_labels.jpg) |![alt text](https://github.com/IrDIE/TestRepo/blob/main/classic_CV/detection/YOLO_unity_dataset/images/val_batch0_pred.jpg) |


---------------------------------

One of test results:


![alt text](https://github.com/IrDIE/TestRepo/blob/main/classic_CV/detection/YOLO_unity_dataset/images/inference.jpg) 


