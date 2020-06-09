import cv2
import numpy as np
from matplotlib import pylab as plt

def roi(image, vertices):
    mask = np.zeros_like(image)
    #channel_count = image.shape[2]
    #match_mask_color = (255,) * channel_count
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1,y1), (x2,y2), (255,0,0), thickness=4)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0)
    return img

def process(img):
    #Step 1: masking the image by defining the region of interest (ROI)
    #print(img.shape)
    height, width = img.shape[:2]

    roi_vertices = [
        (0, height),
        (0, 0.66*height),
        (width/2, height/2),
        (width, 0.66*height),
        (width, height)
    ]

    #Step 2: applying the canny edge detection and cropping the image
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = roi(canny_image, np.array([roi_vertices], np.int32))

    #Step 3: applying the Hough Line Transform and draw the lines
    lines = cv2.HoughLinesP(cropped_image, rho = 2, theta = np.pi/60,
                            threshold = 160,
                            lines = np.array([]),
                            minLineLength = 40,
                            maxLineGap = 25)
    image_with_lines = draw_the_lines(img, lines)
    return image_with_lines

cap = cv2.VideoCapture('Driving.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    frame = process(frame)
    cv2.imshow('driving', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()