import cv2
import matplotlib.pyplot as plt

""" def convert_result_to_image(rgb_image, car_predictions, plate_predictions, time_spent):

    
    fps_count = str(round(1/time_spent,2))
    colors = {"red": (0, 0, 255), "green": (0, 255, 0) , "blue": (255, 0, 0)}
    vehicles = {3 : "Carro", 4 : "Moto", 6 : "Onibus", 8 : "Caminhao"}
    vehicles_color = {3 : (0,0,255), 4 : (2, 105, 173), 6 : (14, 107, 3), 8 : (125, 5, 77)}
    vehicle_count = {3 : 0, 4 : 0, 6 : 0, 8 : 0}
    index = 0
    for class_id, x_min, y_min, x_max, y_max, conf, cor in car_predictions:
        vehicle_count[class_id] += 1
        area = (x_max-x_min) * (y_max - y_min)
        car_predictions[index].append(area)
        index+=1
    
    car_predictions.sort(key = lambda x: x[-1], reverse=False)

    text = f" Contagem de veiculos: {len(car_predictions)} | FPS : {fps_count}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    text_w, text_h = text_size
    cv2.rectangle(rgb_image, (10, 30), (30 + text_w, 30 + text_h), (0,0,0), -1)
    cv2.rectangle(rgb_image, (10, 10), (30 + text_w, 30 + text_h*2+8), (0,0,0), -1)
    cv2.rectangle(rgb_image, (10, 10), (30 + text_w, 30 + text_h*3+16), (0,0,0), -1)
    cv2.rectangle(rgb_image, (10, 10), (30 + text_w, 30 + text_h*4+24), (0,0,0), -1)
    cv2.rectangle(rgb_image, (10, 10), (30 + text_w, 30 + text_h*5+32), (0,0,0), -1)
    rgb_image = cv2.putText(
        rgb_image, 
        text,
        (15, 15 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        1,
        cv2.LINE_AA
    )
    
    text = f" Contagem de carros________{vehicle_count[3]} "
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    text_w, text_h = text_size
    rgb_image = cv2.putText(
        rgb_image, 
        text,
        (15, 15 + text_h*2+8),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        1,
        cv2.LINE_AA
    )

    text = f" Contagem de motos________{vehicle_count[4]} "
    rgb_image = cv2.putText(
        rgb_image, 
        text,
        (15, 15 + text_h*3+16),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        1,
        cv2.LINE_AA
    )

    text = f" Contagem de caminhoes____{vehicle_count[6]} "
    rgb_image = cv2.putText(
        rgb_image, 
        text,
        (15, 15 + text_h*4+24),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        1,
        cv2.LINE_AA
    )

    text = f" Contagem de onibus________{vehicle_count[8]} "
    rgb_image = cv2.putText(
        rgb_image, 
        text,
        (15, 15 + text_h*5+32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        1,
        cv2.LINE_AA
    )
    
    
    for class_id, x_min, y_min, x_max, y_max, conf, ocr_text in plate_predictions:
        
        plt.close()

        rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["blue"], 2)
        text = f"{ocr_text}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_w, text_h = text_size
        cv2.rectangle(rgb_image, (x_min, y_min), (x_min + text_w + 15, y_min - text_h-4), colors["blue"], -1)
        rgb_image = cv2.putText(
            rgb_image, 
            text,
            (x_min, y_min-1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            2,
            cv2.LINE_AA
        )
    


    for class_id, x_min, y_min, x_max, y_max, conf, cor, area in car_predictions:
        plt.close()
        vehicle_color = vehicles_color[class_id]
        rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), vehicle_color, 2)
        

        # Print the attributes of a vehicle. 
        # Parameters in the `putText` function are: img, text, org, fontFace, fontScale, color, thickness, lineType.
        #x, y = pos
        text = f" {vehicles[class_id]} {str(int(conf*100))}%"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_w, text_h = text_size
        cv2.rectangle(rgb_image, (x_min, y_min), (x_min + text_w + 15, y_min - text_h - 15), vehicle_color, -1)
        cv2.rectangle(rgb_image, (x_min, y_min), (x_min + text_w + 15, y_min - text_h*2 - 15), vehicle_color, -1)
        rgb_image = cv2.putText(
            rgb_image, 
            text,
            (x_min, y_min - text_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2,
            cv2.LINE_AA
        )
        text = f" {cor}"
        
        rgb_image = cv2.putText(
            rgb_image, 
            text,
            (x_min, y_min - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2,
            cv2.LINE_AA
        ) 

        


    return rgb_image
 """

vehicles_color = {3 : (0,0,255), 4 : (2, 105, 173), 6 : (14, 107, 3), 8 : (125, 5, 77)}

def draw_vehicle(img, vehicle, recognize_color):

    vehicle_color = vehicles_color[vehicle["class_id"]]
    coords = vehicle["xyxy_coordinates"]
    img = cv2.rectangle(img, (coords["x_min"], coords["y_min"]), (coords["x_max"], coords["y_max"]), vehicle_color, 2)
        

    text = f" {vehicle['class_name']} {str(int(vehicle['confidence']*100))}%"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = text_size
    cv2.rectangle(img, (coords["x_min"], coords["y_min"]), (coords["x_min"] + text_w + 15, coords["y_min"] - text_h - 15), vehicle_color, -1)
    cv2.rectangle(img, (coords["x_min"], coords["y_min"]), (coords["x_min"] + text_w + 15, coords["y_min"] - text_h*2 - 15), vehicle_color, -1)
    img = cv2.putText(
        img, 
        text,
        (coords["x_min"], coords["y_min"] - text_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2,
        cv2.LINE_AA
    )

    if recognize_color:
        text = f" {vehicle['color']}"
            
        img = cv2.putText(
            img, 
            text,
            (coords["x_min"], coords["y_min"] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

    return img

def draw_plate(img, plate, plate_ocr):

    coords = plate["xyxy_coordinates"]
    img = cv2.rectangle(img, (coords["x_min"], coords["y_min"]), (coords["x_max"], coords["y_max"]), (255, 0, 0), 2)
        

    text = f" {plate['class_name']}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = text_size
    cv2.rectangle(img, (coords["x_min"], coords["y_min"]), (coords["x_min"] + text_w + 15, coords["y_min"] - text_h - 15), (255, 0, 0), -1)
    cv2.rectangle(img, (coords["x_min"], coords["y_min"]), (coords["x_min"] + text_w + 15, coords["y_min"] - text_h*2 - 15), (255, 0, 0), -1)
    img = cv2.putText(
        img, 
        text,
        (coords["x_min"], coords["y_min"] - text_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2,
        cv2.LINE_AA
    )

    if plate_ocr:
        print(plate)
        text = f"{plate['ocr']}"

        img = cv2.putText(
            img, 
            text,
            (coords["x_min"], coords["y_min"] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

    return img