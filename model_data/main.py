from Detector import *

# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz "
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz"
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"
image_path = "images/Test1.JPG"
# video_path = "videos/test7.mp4"
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.download(modelURL)
detector.loadModel()
detector.predictImage(image_path, threshold)
# detector.predictvideo(video_path, threshold)
