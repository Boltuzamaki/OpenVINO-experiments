import argparse
import sys
import os 
os.environ['Path'] +="C:\\openvino\\openvino_2021\\deployment_tools\\ngraph\\lib;" \
    "C:\\openvino\\openvino_2021\\deployment_tools\\inference_engine\\external\\tbb\\binl;" \
    "C:\\openvino\\openvino_2021\\deployment_tools\\inference_engine\\bin\\intel64\\Release;" \
    "C:\\openvino\\openvino_2021\\deployment_tools\\inference_engine\\bin\\intel64\\Debug;" \
    "C:\\openvino\\openvino_2021\\deployment_tools\\inference_engine\\external\hddl\\bin;" \
    "C:\\openvino\\openvino_2021\\deployment_tools\\model_optimizer;"

import cv2
import numpy as np
from openvino.inference_engine import IECore
import time 


model_path =r"C:\\openvino\\openvino_2021\\deployment_tools\\model_optimizer\\experiments\\new\\saved_model.xml"
device = 'CPU'
image_path = "r1.jpg"
list_files ="No"
label_path = "label.txt"
number_top = 10

def main():
    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    ie = IECore()

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    net = ie.read_network(model_path)

    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'U8'
    net.outputs[out_blob].precision = 'FP32'

    # Get a number of classes recognized by a model
    num_of_classes = max(net.outputs[out_blob].shape)

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    exec_net = ie.load_network(network=net, device_name=device)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    if list_files == "No":
        start_time = time.time()
        original_image = cv2.imread(image_path)
        image = original_image.copy()
        _, _, h, w = net.input_info[input_blob].input_data.shape

        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))

        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        # Add N dimension to transform to NCHW
        image = np.expand_dims(image, axis=0)

        # ---------------------------Step 7. Do inference----------------------------------------------------------------------
        res = exec_net.infer(inputs={input_blob: image})

        # ---------------------------Step 8. Process output--------------------------------------------------------------------
        # Generate a label list
        if label_path:
            with open(label_path, 'r') as f:
                labels = [line.split(',')[0].strip() for line in f]

        res = res[out_blob]
        # Change a shape of a numpy.ndarray with results to get another one with one dimension
        probs = res.reshape(num_of_classes)
        print(probs)
        # Get an array of args.number_top class IDs in descending order of probability
        top_n_idexes = np.argsort(probs)[-number_top :][::-1]

        header = 'classid probability'
        header = header + ' label' if label_path else header

        for class_id in top_n_idexes:
            probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
            label_indent = ' ' * (len('probability') - 8) if label_path else ''
            label = labels[class_id] if label_path else ''

        print(probability_indent, label_indent, label)
        end_time = time.time()

        print(str(round((end_time - start_time), 2))+" sec")
    else:
        path_image = os.listdir(image_path)
        full_path = [os.path.join(image_path, file) for file in path_image]
        for imgs_path in full_path:
            start_time = time.time()
            original_image = cv2.imread(imgs_path)
            image = original_image.copy()
            _, _, h, w = net.input_info[input_blob].input_data.shape

            if image.shape[:-1] != (h, w):
                image = cv2.resize(image, (w, h))

            # Change data layout from HWC to CHW
            image = image.transpose((2, 0, 1))
            # Add N dimension to transform to NCHW
            image = np.expand_dims(image, axis=0)

            # ---------------------------Step 7. Do inference----------------------------------------------------------------------
            res = exec_net.infer(inputs={input_blob: image})

            # ---------------------------Step 8. Process output--------------------------------------------------------------------
            # Generate a label list
            if label_path:
                with open(label_path, 'r') as f:
                    labels = [line.split(',')[0].strip() for line in f]

            res = res[out_blob]
            # Change a shape of a numpy.ndarray with results to get another one with one dimension
            probs = res.reshape(num_of_classes)
            print(probs)
            # Get an array of args.number_top class IDs in descending order of probability
            top_n_idexes = np.argsort(probs)[-number_top :][::-1]

            header = 'classid probability'
            header = header + ' label' if label_path else header

            for class_id in top_n_idexes:
                probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
                label_indent = ' ' * (len('probability') - 8) if label_path else ''
                label = labels[class_id] if label_path else ''

            print(probability_indent, label_indent, label)
            end_time = time.time()

            print(str(round((end_time - start_time), 2))+" sec")        

if __name__ == "__main__":
    main()