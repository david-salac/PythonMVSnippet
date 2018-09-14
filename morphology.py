import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.ndimage import label


def threshold(im, value):
    new_image = np.zeros(im.shape)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
                if im[i,j] > value:
                    new_image[i,j] = 0
                else:
                    new_image[i,j] = 1

    return new_image


def histogram(im):
    hist, be = np.histogram(im, 256, (0, 256))
    return hist


def add_neighbor(collection, tup):
    if (tup[0] > 0) and (tup[1] > 0):
        if tup[0] > tup[1]:
            collection.add((tup[1], tup[0]))
        else:
            collection.add(tup)


def identify_neighbors(im_binary):

    colored = copy.deepcopy(im_binary)
    ctr = 1
    neighbors = set()
    for i in range(1, colored.shape[0]):
        for j in range(1, colored.shape[1]-1):
            if colored[i,j] == 1:
                mask = [colored[i-1, j-1],
                        colored[i - 1, j],
                        colored[i - 1, j + 1],
                        colored[i, j - 1]]
                indices = np.nonzero(mask)[0]

                if len(indices) == 0: ## vsude v okoli jsou hodnoty pozadi
                    ctr += 1
                    colored[i,j] = ctr
                    continue
                else: ## v okoli jsou i jine hodnoty
                    no_zeros = []
                    no_zeros[:] = (value for value in mask if value != 0)

                    if np.all(no_zeros == mask[indices[0]]): # vsechny hodnoty v okoli jsou stejne
                        colored[i,j] = mask[indices[0]]
                    else:
                        colored[i,j] = np.min(no_zeros)
                        add_neighbor(neighbors, (colored[i-1, j-1], colored[i-1, j]))
                        add_neighbor(neighbors, (colored[i-1, j-1], colored[i - 1, j + 1]))
                        add_neighbor(neighbors, (colored[i - 1, j + 1], colored[i, j - 1]))

    return colored, neighbors


def join_neighbors(im_colored, neighbors_set):
    im_joined = im_colored
    for i in range(im_joined.shape[0]):
        for j in range(im_joined.shape[1]):
            if im_joined[i,j] > 0:
                matches = []
                matches[:] = (value for value in neighbors_set if value[1] == im_joined[i,j])

                if len(matches) > 0: # nalezli se sousede
                    im_joined[i,j] = min(matches, key = lambda t: t[0])[0]
    return im_joined


## returns format (dictionary) {area_number: [y_t, x_t, sum]}
def area_information(im_colored):
    # {area_number: [y_sum, x_sum, sum]}
    info = dict()
    for i in range(im_colored.shape[0]):
        for j in range(im_colored.shape[1]):
                area_n = im_colored[i,j]
                if area_n > 0:
                    if info.get(area_n, 0) == 0:
                        info[area_n] = [i, j, 1]
                    else:
                        info[area_n] = [sum(x) for x in zip(info[area_n], [i,j,1])]

    for key in info:
        #devide by sum
        info[key][0] = int(np.floor(info[key][0] / info[key][2]))
        info[key][1] = int(np.floor(info[key][1] / info[key][2]))
    return info


def draw_centers(original_image, segments_info, color):

    for key in segments_info:

        x = segments_info[key][1]
        y = segments_info[key][0]

        for m in range(y-2, y+3):
            for n in range (x-2, x+3):
                original_image[m,n] = color


def invert(binary_im):
    for i in range(binary_im.shape[0]):
        for j in range(binary_im.shape[1]):
            if binary_im[i,j] == 0:
                binary_im[i,j] = 1
            else:
                binary_im[i,j] = 0


image = cv2.imread(r'images/cv08_im1.bmp')
im_copy = copy.deepcopy(image)
thresholded = threshold(image[:,:,0],100)
B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
bw1 = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, B)

colored, neighbors = identify_neighbors(bw1)

## spojeni sousedu (pruchod 2)
joined = join_neighbors(colored, neighbors)

## ziskani informaci (teziste, pocet pixelu, apod)
segments_info = area_information(joined)

## vykresleni stredu do objektu (cervena)
draw_centers(im_copy, segments_info, [255,0,0])

plt.figure()

plt.subplot(2,2,1)
plt.title("segmentation im1")
plt.imshow(bw1, 'gray')

plt.subplot(2,2,2)
plt.title("centers im1")
plt.imshow(im_copy)


image2 = cv2.imread(r'images/cv08_im2.bmp')
im2_copy = copy.deepcopy(image2)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2YCR_CB)
im_cr = image2[:,:,1]
im_thresholded = threshold(im_cr, 120)
B2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
bw2 = cv2.morphologyEx(im_thresholded, cv2.MORPH_CLOSE, B2)

invert(bw2)

colored, neighbors = identify_neighbors(bw2)

## spojeni sousedu (pruchod 2)
joined = join_neighbors(colored, neighbors)

## ziskani informaci (teziste, pocet pixelu, apod)
segments_info = area_information(joined)

## vykresleni stredu do objektu (cervena)
draw_centers(im2_copy, segments_info, [255,0,0])

plt.subplot(2,2,3)
plt.title("segmentation im2")
plt.imshow(bw2, 'gray')

plt.subplot(2,2,4)
plt.title("centers im2")
plt.imshow(im2_copy)

plt.show()


