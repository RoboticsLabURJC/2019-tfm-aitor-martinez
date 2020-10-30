import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2
from PIL import Image
from datetime import datetime


from object_detection.utils import label_map_util
from.utils import wrap_frozen_graph

LABELS_DICT = {'voc': 'networks/labels/pascal_label_map.pbtxt',
               'coco': 'networks/labels/mscoco_label_map.pbtxt',
               'kitti': 'networks/labels/kitti_label_map.txt',
               'oid': 'networks/labels/oid_bboc_trainable_label_map.pbtxt',
               'pet': 'networks/labels/pet_label_map.pbtxt'}

class DetectionNetwork():
    def __init__(self, model, label_map_file, model_type, threshold=0.5, print_graph=False):

        label_map = label_map_util.load_labelmap(label_map_file) # loads the labels map.

        self.classes = label_map_util.get_label_map_dict(label_map, use_display_name=True)

        self._classes_idx = list(self.classes.values())
        self._classes_names = list(self.classes.keys())

        self.net = None
        self.type = model_type
        # Load the TRT frozen graph from disk
        if (model_type == 'graph'):
            with open(model, 'rb') as fid:
                graph_def = tf.compat.v1.GraphDef()
                read = fid.read()
                print(read)
                graph_def.ParseFromString(read)
            print ("Loaded...")
            self.net = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=["x:0"],
                                        outputs=["Identity:0"],
                                        print_graph=print_graph)

        elif (model_type == 'saved_model'):
            self.net = tf.saved_model.load(model)

        self.confidence_threshold = threshold

        print("Network ready!")

    def predict(self, img, detect_class=None):
        input_image = np.array(img)
        height = img.shape[0]
        width = img.shape[1]
        predictions = boxes = scores = None
        if (self.type == 'saved_model'):
            tf_img = tf.expand_dims(tf.convert_to_tensor(input_image, dtype=tf.uint8), axis=0)
            result = self.net(tf.constant(tf_img))
            boxes = result['detection_boxes']
            scores = result['detection_scores']
            predictions = result['detection_classes']
        elif (self.type == 'graph'): 
            (predictions, boxes, scores, _) = self.net(x=tf.constant(input_image))
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        predictions = np.squeeze(predictions)


        # We only keep the most confident predictions.
        mask1 = scores > self.confidence_threshold # bool array

        # We map the predictions into a mask (human or not)
        mask2 = []
        if (detect_class):
            for idx in predictions:
                mask2.append(self._classes_names[self._classes_idx.index(int(idx))] == detect_class)
        else:
            mask2 = [True]*len(predictions) 

        # Total mask: CONFIDENT PERSONS
        mask = np.logical_and(mask1, mask2)
        # Boxes containing only confident humans
        boxes = boxes[mask]
        # aux variable for avoiding race condition while int casting
        # Box format and reshaping...
        tmp_boxes = np.zeros([len(boxes), 4])
        tmp_boxes[:,[0,2]] = boxes[:,[1,3]] * width
        tmp_boxes[:,[3,1]] = boxes[:,[2,0]] * height
        return_boxes = tmp_boxes.astype(int)

        return_scores = scores[mask]
        return_predictions = []
        for idx in predictions[mask]:
            return_predictions.append(self._classes_names[self._classes_idx.index(int(idx))])

        return (return_predictions, return_boxes, return_scores)

    def getPerson(self, img):
        return self.predict(img, 'person')

