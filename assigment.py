#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#reading in an image
image = mpimg.imread('challenge4.jpeg')
#image = mpimg.imread('test_images/test1.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
#plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def line_x(y, b, k):
    x = (y-b)/k
    if math.isnan(x): print (y, b, k)
    return int(x)

def draw_lines_src(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), color,
                     thickness)

def hough_lines_src(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines_src(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    if lines is None: return
    #define medium lef and line slopes (additional filter)
    sd = {-1: 0, 1:0} # -1 - left, 1 - right
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope == 0: continue
            sign = np.sign(slope)
            if sd[sign] ==0 :  sd[sign] = slope
            sd[sign] = (sd[sign] + slope)/2

    top = img.shape[0]
    bottom = 0
    bd = {-1: 0, 1: 0}
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope == 0: continue
            sign = np.sign(slope)
            correlation = (sd[sign] - slope)/slope
            if np.abs(correlation) > 0.2: continue # BONUS: filter by slope
            top = min(top, y1, y2)
            bottom = max(bottom, y1, y2)

            b=y1-slope*x1
            if bd[sign] == 0 :  bd[sign] = b
            bd[sign] = (bd[sign] + b)/2

    for sign in (-1,1):
        if not math.isnan(bd[sign]) and not math.isinf(bd[sign]) and sd[sign] != 0:
            cv2.line(img, (line_x(top, bd[sign], sd[sign]), top), (line_x(bottom, bd[sign], sd[sign]), bottom), color, 15)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

import os
os.listdir("test_images/")

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def process_image(image, vertices, hsv = False):
    if hsv:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([70, 0, 0])
        upper_yellow = np.array([100, 255, 255])
        mono_image = cv2.inRange(hsv, lower_yellow, upper_yellow)
    else:
        color_select = np.copy(image)
        rgb_threshold = [200, 200, 0]
        thresholds = (image[:, :, 0] < rgb_threshold[0]) \
                     | (image[:, :, 1] < rgb_threshold[1]) \
                     | (image[:, :, 2] < rgb_threshold[2])
        color_select[thresholds] = [0, 0, 0]
        mono_image = grayscale(color_select)

    kernel_size = 5
    blur_gray = gaussian_blur(mono_image, kernel_size)

    high_thresh, thresh_im = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5 * high_thresh
    cany_image = canny(blur_gray, lowThresh, high_thresh)

    masked_edges = region_of_interest(cany_image, vertices)

    rho = 5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 80  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    hough_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    result = weighted_img(hough_image, image)

    return result

def process_image1(image):
    imshape = image.shape
    vertices = np.array([[(100, imshape[0]), (450, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    return process_image(image, vertices, False)

def process_image2(image):
    vertices = np.array([[(220, 680), (600, 440), (720, 440), (1120, 670)]], dtype=np.int32)
    return process_image(image, vertices, True)


"""
res = process_image2(image)
plt.imshow(res)
plt.show()


"""
challenge_output = 'white.mp4'
clip2 = VideoFileClip('solidWhiteRight.mp4')

#clip2.save_frame("challenge3.jpeg", t=4)

challenge_clip = clip2.fl_image(process_image1)
challenge_clip.write_videofile(challenge_output, audio=False)

