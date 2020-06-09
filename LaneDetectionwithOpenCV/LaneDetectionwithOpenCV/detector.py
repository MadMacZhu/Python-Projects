import cv2
import numpy as np
from matplotlib import pylab as plt

img = cv2.imread('road.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Step 1: masking the image by defining the region of interest (ROI)
print(img.shape)
height, width = img.shape[:2]

roi_vertices = [
    (0, height),
    (0, 1100),
    (width/2, height/2),
    (width, 1100),
    (width, height)
]

def roi(image, vertices):
    mask = np.zeros_like(image)
    #channel_count = image.shape[2]
    #match_mask_color = (255,) * channel_count
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#Step 2: applying the canny edge detection and cropping the image
gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)
cropped_image = roi(canny_image, np.array([roi_vertices], np.int32))

#Step 3: applying the Hough Line Transform and draw the lines
def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (255,0,0), thickness=4)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0)
    return img

lines = cv2.HoughLinesP(cropped_image, rho = 6, theta = np.pi/60,
                        threshold = 160,
                        lines = np.array([]),
                        minLineLength = 40,
                        maxLineGap = 25)
image_with_lines = draw_the_lines(img, lines)


plt.imshow(image_with_lines)
plt.show()
