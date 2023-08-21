from numpy.random import randint
import matplotlib.pyplot as plt
import cv2
from skimage.draw import random_shapes
import numpy as np

def show_img(img):
    while True:
        cv2.imshow('a', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



image3, _ = random_shapes((256*3, 256*3), \
                          min_shapes = 4, max_shapes=5,shape = 'triangle',min_size = 50, max_size = 150,
                          intensity_range=((50, 255),), num_trials = 5)
#show_img(image3)



rng = np.random.default_rng(42)
print(rng.integers(max(1, 256*3 - 50)))