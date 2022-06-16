import cv2
import numpy as np


def find_seed(image):
    height = image.shape[0]
    width = image.shape[1]

    x = int(width/2)
    y = 0

    positive = 0

    # Iterate through rows until current row contains desk pixels
    while(positive < 2):
        if image[y, x] != 0:
            positive += 1
            y += int(height/10)
        else:
            y += (height/5)

    return x, y


def find_edge(image, direction: str):
    height = image.shape[0]
    width = image.shape[1]
    dir_factor = 1

    seed_x, seed_y = find_seed(image)

    # Set boundary start points for searching edges
    if direction == 'top':
        start = (0, width/2)
    elif direction == 'down':
        start = (height-1, width/2)
        dir_factor = -1
    elif direction == 'left':
        start = (seed_y, 0)
    else:
        start = (seed_y, width-1)
        dir_factor = -1

    y = int(start[0])
    x = int(start[1])

    # Iterate until finding desk pixels
    if direction == 'top' or direction == 'down':
        while(True):
            if image[y, x] != 0:
                return y + 20 * dir_factor
            y += 1 * dir_factor
    else:
        while(True):
            if image[y, x] != 0:
                return x + 20 * dir_factor
            x += 1 * dir_factor


def edges(image):
    top = find_edge(image, 'top')
    down = find_edge(image, 'down')
    left = find_edge(image, 'left')
    right = find_edge(image, 'right')

    return top, down, left, right


def crop_wood(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresholded_image = cv2.threshold(
        gray_image, 100, 255, cv2.THRESH_TOZERO)

    top, down, left, right = edges(thresholded_image)
    wood_image = image[top:down, left:right]

    return wood_image


def bound_defects(image):
    new_width = 460
    new_height = 200

    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    new_image = cv2.convertScaleAbs(resized_image, alpha=1.55, beta=0)
    gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    edges_image = cv2.Canny(blurred_image, 50, 250)

    kernel = np.ones((10, 10), np.uint8)
    closed_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_image, mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    boundings = []
    offset = 4

    # Creating bounding rectangles of defects contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        x = (x - offset) / new_width
        y = (y - offset) / new_height
        w = (w + 2*offset) / new_width
        h = (h + 2*offset) / new_height

        x = max(0, x)
        y = max(0, y)

        boundings.append((x, y, w, h))

    return boundings
