import cv2
import numpy as np

def blob_filter(img, height, width):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_c = height * width
    filtered_img = img.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        if w > width* 0.4 :
            cv2.fillPoly(filtered_img, [contour], 0)
        elif w > width* 0.1 and h < height * 0.2:
            cv2.fillPoly(filtered_img, [contour], 0)
        elif h > height * 0.95 and w < width* 0.1 :
            cv2.fillPoly(filtered_img, [contour], 0)
        elif area < area_c * 0.01:
            cv2.fillPoly(filtered_img, [contour], 0)
    if cv2.countNonZero(filtered_img) < area_c * 0.03:
        return img
    else:
        return filtered_img

def correct_color(img) :
    one_count = cv2.countNonZero(img)
    img2 = cv2.bitwise_not(img)
    zero_count = cv2.countNonZero(img2)
    if one_count>zero_count:
        return img2
    else:
        return img2
    
def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def image_preprocessing(img,resize_factor):

    
    #img = cv2.imread(image_path)
    height, width, _ = img.shape
    img = cv2.resize(img, (int(width*resize_factor), int(height*resize_factor)), interpolation = cv2.INTER_CUBIC)
    
    img = cv2.GaussianBlur(img, (5, 5), 1)
    #img = cv2.medianBlur(img,5)
    #img = cv2.bilateralFilter(img,4,20,30)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
    #topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    #blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img + topHat - blackHat
    
    alpha = 0.9
    beta = 8
    #img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    #img[img > 160 ] = 160
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,	iterations=2)
    #img = cv2.dilate(img, kernel, iterations=1)
    #img = noise_removal(img)
    #img = correct_color(img)
    #img = cv2.dilate(img, (1,1), iterations=3)
    #img = blob_filter(img, height*2, width*2)
    #kernel = np.ones((3,3),np.uint8)
    #img = cv2.erode(img, kernel, iterations=1)
    #img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    #img = cv2.GaussianBlur(img, (5, 5), 1)
    #img = cv2.bitwise_not(img)
    #_, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        
    return img
