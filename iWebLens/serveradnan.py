import numpy as np
import time
import cv2
import os
from flask import Flask, request, Response
import io
import json
from PIL import Image

yoloPath = './'

def getWeights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yoloPath, weights_path])
    return weightsPath

def loadModel(configpath, weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    return cv2.dnn.readNetFromDarknet(configpath, weightspath)


def get_predection(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    class_ids = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # Max value inside array
            class_id = np.argmax(scores)
            #Max value in class_id goes to confidence
            confidence = scores[class_id]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.3:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                object = []
                #
                for i in class_ids:
                    object.append(LABELS[i])

                accuracy = []
                for j in confidences:
                    accuracy.append(j * 100)

                jsonList = []
                for i in range(0, len(object)):
                    jsonList.append({"label": object[i], "accuracy": format(accuracy[i],'.2f')})
    return json.dumps(jsonList, indent=1)


labelsPath="coco.names"
cfgpath="yolov3-tiny.cfg"
wpath="yolov3-tiny.weights"



#Here we load all of our class LABELS

lpath = os.path.sep.join([yoloPath, labelsPath])
Lables = open(lpath).read().strip().split("\n")


CFG=os.path.sep.join([yoloPath, cfgpath])
Weights=getWeights(wpath)
# nets=loadModel(CFG, Weights)

#Random COLORS  are then assigned to each label
np.random.seed(42)
Colors=np.random.randint(0, 255, size=(len(Lables), 3), dtype="uint8")

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/iweblens', methods=['POST'])
def main():
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    nets=loadModel(CFG, Weights)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res=get_predection(image,nets,Lables,Colors)
    return Response(res)

    # start flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)