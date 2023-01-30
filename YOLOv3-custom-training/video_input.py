import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time
import datetime

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess
import string

# import tensorflow.compat.v1.keras.backend as K 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def get_precision(label,y_pred):
    """
    label is int (encoded label)
    y_pred is a list type
    """
    alphabet = list(string.ascii_uppercase)

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(alphabet)
    encoded_labels = encoder.transform(y_pred)

    total_frames = len(y_pred)
    y_true = total_frames * [label]
            
    from sklearn.metrics import precision_score
    ps = precision_score(y_true,encoded_labels,average='micro')
    print(f"label is {y_true}")
    print(f"encoded labels are {encoded_labels}")

    return ps

class YOLO(object):
    _defaults = {
        #"model_path": 'logs/trained_weights_final.h5',
        "model_path": 'D:\Python\\yolo_prom02\\yolo_keras\YOLOv3-custom-training\logs\\001\\trained_weights_final_2.h5',
        "anchors_path": 'D:\Python\yolo_prom02\\yolo_keras\YOLOv3-custom-training\model_data\yolo_anchors.txt',
        "classes_path": 'D:\Python\yolo_prom02\\yolo_keras\YOLOv3-custom-training\classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "text_size" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        if len(out_scores)==0:
            return image,ObjectsList,50

        predicted_classes = []
        boxes = []
        scores = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_classes.append(self.class_names[c])
            boxes.append(out_boxes[i])
            scores.append(out_scores[i])
        order = np.argmax(scores)
        predicted_class = predicted_classes[order]
        box = boxes[order]
        score = scores[order]
        label = '{} {:.2f}'.format(predicted_class, score)
        #label = '{}'.format(predicted_class)
        scores = '{:.2f}'.format(score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

        mid_h = (bottom-top)/2+top
        mid_v = (right-left)/2+left

        # put object rectangle
        cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

        # get text size
        (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

        # put text rectangle
        cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

        # add everything to list
        ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList, predicted_class

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        #image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        r_image, ObjectsList, predicted_class = self.detect_image(original_image_color)
        return r_image, ObjectsList, predicted_class
def get_precision_over_classes(file_path,class_path,y_true_class):

    # set start time to current time
    start_time = time.time()
    # displays the frame rate every 2 second
    display_time = 2
    # Set primarry FPS to 0
    fps = 0

    # we create the video capture object cap
    cap = cv2.VideoCapture(os.path.join(file_path,class_path+".mp4"))
    i=0 #frame counter
    frameTime = 17 # time of each frame in ms, you can add logic to change this value. for 60 fps each frame is 17ms in duration
    y_pred = []    
    if not cap.isOpened():
        raise IOError("We cannot open webcam")

    while(cap.isOpened()):
        ret = cap.grab() #grab frame
        i=i+1 #increment counter
        if i % 10 == 0 and ret != False:
            ret, frame = cap.retrieve()
            frame = cv2.flip(frame,1)
            # resize our captured frame if we need
            frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

            frame=cv2.flip(frame, 1)

            # detect object on our frame
            r_image, ObjectsList,predicted_class = yolo.detect_img(frame)
            if predicted_class != 50:
                y_pred.append(predicted_class)

            # show us frame with detection
            cv2.imshow("Web cam input", r_image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


            # calculate FPS
            fps += 1
            TIME = time.time() - start_time
            if TIME > display_time:
                print("FPS:", fps / TIME)
                fps = 0 
                start_time = time.time()
        if ret == False:
            # count the number of frames
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # calculate duration of the video
            seconds = round(frames / fps)
            video_time = datetime.timedelta(seconds=seconds)
            print(f"duration in seconds: {seconds}")
            print(f"video time: {video_time}")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    ps_per_class = (get_precision(y_true_class,y_pred))
    print(y_pred)
    print(ps_per_class)
    return ps_per_class

    
if __name__=="__main__":

    import json

    ps_per_letter = {}
    alphabet = list(string.ascii_uppercase)
    yolo = YOLO()
    file_path = r"D:\\obs-studio\\vidoes\\mp4" # directory of recorded videos
    for idx,letter in enumerate(alphabet):
        ps_per_letter[letter] = get_precision_over_classes(file_path,letter,idx)

    yolo.close_session()
    with open('ps_yolov3_per_class.json','w') as write_file:
        json.dump(ps_per_letter,write_file)
