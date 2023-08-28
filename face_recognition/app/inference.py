
import cv2
from openvino.runtime import Core
import time
from utils_car.models import model_init, detect_cars, detect_plates, attributes_recognition
from utils_car.visual import draw_plate, draw_vehicle
from utils_car.postprocessing import validate_plate
from utils_car.preprocessing import image_preprocessing
from utils_car.models import plate_reader_easyocr
import numpy as np

ie_core = Core()

modelo_carros_de_path = "/home/gabriel/fr_project/face_recognition/app/modelos/detec_veiculos/ssd_mobilenet_v1_coco.xml"
input_key_carros_de, output_keys_carros_de, compiled_modelo_carros_de = model_init(ie_core, modelo_carros_de_path)
height_carros_de, width_carros_de = list(input_key_carros_de.shape)[1:3]
carros_input_size = (height_carros_de, width_carros_de)


modelo_placas_de_path = "/home/gabriel/fr_project/face_recognition/app/modelos/detec_placas/yolov5n_plates.xml"
input_key_placas_de, output_keys_placas_de, compiled_modelo_placas_de = model_init(ie_core, modelo_placas_de_path)
height_placas_de, width_placas_de = list(input_key_placas_de.shape)[2:]
placas_input_size = (height_placas_de, width_placas_de)


modelo_atributos_re_path = "/home/gabriel/fr_project/face_recognition/app/modelos/detec_cores/vehicle-attributes-recognition-barrier-0042.xml"
input_key_atributos_de, output_keys_atributos_de, compiled_modelo_atributos_re = model_init(ie_core, modelo_atributos_re_path)
height_atributos_de, width_atributos_de = list(input_key_atributos_de.shape)[2:]
atributos_input_size = (height_atributos_de, width_atributos_de)





def filter_minimum_size(coords, width, height):
    print(coords["x_max"])
    if (coords["x_max"] - coords["x_min"]) > width and (coords["y_max"] - coords["y_min"]) > height:
        return True
    else:
        return False


def car_analytics(img: np.ndarray,
                  detect_vehicles: bool=True,
                  recognize_color: bool=True,
                  detect_plate: bool=True,
                  plate_ocr: bool=True):
    
    if plate_ocr and not detect_plate:
        raise ValueError("detect_plate must be true if plate_ocr is true")
    
    if recognize_color and not detect_vehicles:
        raise ValueError("detect_vehicles must be true if recognize_color is true")
    
    detections = {}
    detections["plates"] = {}
    detections["vehicles"] = {}

    if detect_vehicles:
        
        detected_vehicles = detect_cars(img, output_keys_carros_de, compiled_modelo_carros_de, carros_input_size)

        for id, vehicle in enumerate(detected_vehicles["predictions"]):
            if all( i < 0 for i in vehicle["xyxy_coordinates"].values()):
                continue
            detections["vehicles"]["detection " + str(id)] = {"class_id" : vehicle["class_id"],
                                                              "class_name" : vehicle["class_name"],
                                                              "xyxy_coordinates" : vehicle["xyxy_coordinates"],
                                                              "confidence" : vehicle["confidence"]}

        if recognize_color:
            for vehicle in detections["vehicles"].keys():
                
                coords = detections["vehicles"][vehicle]["xyxy_coordinates"]
                cropped_vehicle = img[coords["y_min"] : coords["y_max"],
                                  coords["x_min"] : coords["x_max"]]
                color = attributes_recognition(compiled_modelo_atributos_re, atributos_input_size, cropped_vehicle)
                detections["vehicles"][vehicle]["color"] = color
        
        for vehicle in detections["vehicles"].keys():
            img = draw_vehicle(img, detections["vehicles"][vehicle], recognize_color)
        
    
    if detect_plate:

        detected_plates = detect_plates(img, output_keys_placas_de, compiled_modelo_placas_de, placas_input_size)
        
        for id, plate in enumerate(detected_plates["predictions"]):

            if not filter_minimum_size(plate["xyxy_coordinates"], 10, 10):
                continue

            if all(i < 0 for i in plate["xyxy_coordinates"].values()):
                continue

            detections["plates"]["detection " + str(id)] = {"class_id" : plate["class_id"],
                                                            "class_name" : plate["class_name"],
                                                            "xyxy_coordinates" : plate["xyxy_coordinates"],
                                                            "confidence" : plate["confidence"]}

 
        t = time.time()
        if plate_ocr:
            for plate in detections["plates"]:

                coords = detections["plates"][plate]["xyxy_coordinates"]
                cropped_plate = img[coords["y_min"] : coords["y_max"],
                                  coords["x_min"] : coords["x_max"]]
                
                preprocessed_image = image_preprocessing(cropped_plate,resize_factor=1.0)
                ocr_result = plate_reader_easyocr(preprocessed_image)

                if not isinstance(ocr_result, str):
                    ocr_result = ""
                has_plate_reading, result = validate_plate(ocr_result)
                detections["plates"][plate]["ocr"] = result
                img = draw_plate(img, detections["plates"][plate], plate_ocr)

    if len(detections["vehicles"]) > 0 or len(detections["plates"]) > 0:
        has_detections = True
    else:
        has_detections = False

    #cv2.imwrite("imagens_de_teste/teste.jpg", img)
    

    return has_detections, img, detections






