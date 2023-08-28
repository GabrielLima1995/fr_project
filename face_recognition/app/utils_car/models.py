import cv2
import numpy as np
import torch
from .general import percentage2pixel, pad_resize_image, non_max_suppression, scale_coords
from typing import Tuple
import easyocr

def model_init(ie_core, model_path: str) -> Tuple:

    model = ie_core.read_model(model=model_path)
    compiled_model = ie_core.compile_model(model=model, device_name="CPU")
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model


def detect_cars(raw_image, output_keys, model, input_size):
    """
        raw_image: image to infer in bgr
        model: model compiled in openvino
        input_keys: input keys of the model
        output_keys: output keys of the model
    """
    vehicles_names = {3 : "Carro", 4 : "Moto", 6 : "Onibus", 8 : "Caminhao"}
    raw_image_resized = cv2.resize(raw_image, input_size)
    raw_image_transposed = np.expand_dims(raw_image_resized.transpose(0, 1, 2), 0)
    detections = model([raw_image_transposed])[output_keys]
    detections = np.squeeze(detections, (0, 1))
    detections = detections[~np.all(detections == 0, axis=1)]
    boxes = detections[:, 2:]
    

    (real_y, real_x), (resized_y, resized_x) = raw_image.shape[:2], raw_image_resized.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    converted_boxes = []
    for box in boxes:
        new_box = percentage2pixel(box, ratio_x, ratio_y, resized_y, resized_x)
        converted_boxes.append(new_box)

    predictions = []
    
    for detec in range(len(detections[:-1])):
        
        if int(detections[detec][1]) not in [3,4,6,8]: 
                continue
        predictions.append({"class_id":int(detections[detec][1]),
                            "class_name": vehicles_names[int(detections[detec][1])],
                            "xyxy_coordinates":converted_boxes[detec],
                            "confidence": round(detections[detec][6],2)},)

    return {"resized_image":raw_image_resized, "predictions" : predictions}

def detect_plates(raw_image, output_keys, model, input_size):
    """
        raw_image: image to infer in bgr
        model: model compiled in openvino
        input_keys: input keys of the model
        output_keys: output keys of the model
    """
    
    raw_image_resized = pad_resize_image(raw_image, input_size)
    img_in = np.transpose(raw_image_resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in /= 255.0
    raw_image_transposed = np.expand_dims(img_in, axis=0)
    
    detections = model([raw_image_transposed])[output_keys]
    detections = torch.from_numpy(detections)
    detections = non_max_suppression(detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)[0]
    labels = detections[..., -1].tolist()
    boxs = detections[..., :4].numpy()
    confs = detections[..., 4].tolist()
    boxs[:, :] = scale_coords(input_size, boxs[:, :], raw_image.shape).round()
    predictions = []
    for detec in range(len(labels)):
        xyxy_coordinates = [int(x) for x in boxs[detec].tolist()]
        coords_keys = ["x_min", "y_min", "x_max", "y_max"]
        xyxy_coordinates = dict(zip(coords_keys, xyxy_coordinates))
        predictions.append({"class_id":int(labels[detec]), "class_name": "Placa", "xyxy_coordinates": xyxy_coordinates, "confidence" : round(confs[detec],2)})


    return {"resized_image":raw_image_resized, "predictions" : predictions}

def attributes_recognition(compiled_model_re, input_size, raw_image):

    colors = ['Branco', 'Cinza', 'Amarelo', 'Vermelho', 'Verde', 'Azul', 'Preto']
    
    resized_image_re = cv2.resize(raw_image, input_size)
    input_image_re = np.expand_dims(resized_image_re.transpose(2, 0, 1), 0)
    
    predict_colors = compiled_model_re([input_image_re])[compiled_model_re.output(0)]

    attr_color = (colors[np.argmax(predict_colors)])
    return attr_color



reader= easyocr.Reader(["en"], gpu=False, model_storage_directory="/home/gabriel/fr_project/face_recognition/app/modelos/ocr/", user_network_directory="/home/gabriel/fr_project/face_recognition/app/modelos/ocr/", recog_network='custom_example')
def plate_reader_easyocr(plate_img):
    print(plate_img.shape)
    detections = reader.readtext(image = plate_img, allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", text_threshold=0.3,
                                batch_size=5, contrast_ths = 0.3, adjust_contrast =0.8)
    for detection in detections:
        bbox, text, score = detection
        print(bbox)
        text = text.upper().replace(" ", "")
        return text
