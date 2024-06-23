import cv2
import tensorflow as tf
import os
import numpy as np
import time
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)


class Detector:
    def __init__(self):
        self.model = None
        self.cache_dir = None
        self.model_name = None
        self.classes_list = None
        self.color_list = None

    def readClasses(self, classes_filepath):
        with open(classes_filepath, 'r') as f:
            self.classes_list = f.read().splitlines()

        # Colors list
        self.color_list = np.random.uniform(low=0, high=255, size=(len(self.classes_list), 3))

    def download(self, modelURL):

        file_name = os.path.basename(modelURL)
        self.model_name = file_name[:file_name.index('.')]

        self.cache_dir = "./pretrained_models"

        os.makedirs(self.cache_dir, exist_ok=True)

        get_file(fname=file_name, origin=modelURL, cache_dir=self.cache_dir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading Model " + self.model_name)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cache_dir, "checkpoints", self.model_name, "saved_model"))

        print("Model " + self.model_name + " loaded successfully...")

    def createBoundingBox(self, image, threshold = 0.5):
        input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor)

        bboxes = detections['detection_boxes'][0].numpy()
        class_indexes = detections['detection_classes'][0].numpy().astype(np.int32)
        class_scores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bbox_idx = tf.image.non_max_suppression(bboxes, class_scores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

        print(bbox_idx)

        if len(bbox_idx) != 0:
            for i in range(0, len(bbox_idx)):
                bbox = tuple(bboxes[i].tolist())
                class_confidence = round(100 * class_scores[i])
                class_index = class_indexes[i]

                class_label_text = self.classes_list[class_index].upper()
                class_color = self.color_list[class_index]

                display_text = '{}: {}%'.format(class_label_text, class_confidence)

                y_min, x_min, y_max, x_max = bbox

                x_min, x_max, y_min, y_max = (x_min*imW, x_max*imW, y_min*imH, y_max*imH)
                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=class_color, thickness=1)
                cv2.putText(image, display_text, (x_min, y_min-10), cv2.FONT_HERSHEY_PLAIN, 1, class_color, 2)

                line_width = min(int((x_max - x_min) * 0.2), int((y_max - y_min) * 0.2))

                cv2.line(image, (x_min, y_min), (x_min + line_width, y_min), class_color, thickness=5)
                cv2.line(image, (x_min, y_min), (x_min, y_min+line_width), class_color, thickness=5)

                cv2.line(image, (x_max, y_max), (x_max - line_width, y_max), class_color, thickness=5)
                cv2.line(image, (x_max, y_max), (x_max, y_max - line_width), class_color, thickness=5)

        return image

    def predictImage(self, image_path, threshold = 0.5):
        image = cv2.imread(image_path)

        bbox_image = self.createBoundingBox(image, threshold)

        cv2.imwrite(self.model_name + ".jpg", bbox_image)
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictvideo(self, video_path, threshold = 0.5):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening file...")
            return
        start_time = 0
        (success, image) = cap.read()
        while success:
            current_time = time.time()
            fps = 1 / (current_time - start_time)
            start_time = current_time

            bbox_image = self.createBoundingBox(image, threshold)

            cv2.putText(bbox_image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Result", bbox_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()

        cv2.destroyAllWindows()

