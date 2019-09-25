# Safety_Detection
Open the Helmet_Vest_Detection.ipynb in Google Colab for Yolo object detection training. use the below url for colab.
http://colab.research.google.com

Upload the yolo-voc.2.0.cfg, obj.data, obj.names files in colab under darknet/cfg directory.

Upload Makefile, train.txt and test.txt in colab under darknet directory.

After running all the cells in colab. The last cell actually starts training the yolo object detection model. Every 100 epochs of training the model weights are stored under darknet/backup directory. 

https://www.youtube.com/watch?v=pqLSeACd97w&feature=youtu.be

Download the trained model weights and run test code using yolo_video.py on real time video streams or recorded video.





