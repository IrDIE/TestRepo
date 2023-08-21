from Generator import DataGenerator
import cv2
import sys
from numpy.random import randint, random

def show_img(img):
    while True:
        cv2.imshow('a', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main_task_1_generation():
    generator = DataGenerator()
    for i in range(1, 101):
        n_points = randint(1, 6)
        img = generator.generate_image_ann(is_hexagon_required = False, \
                                           n_points = n_points, path = './task1_generated_dataset' + f'/{i}')

    #show_img(img)

def main():
    generator = DataGenerator()
    n_points = randint(1, 6)
    i = 0
    img = generator.generate_image_ann(is_hexagon_required=False, n_points=n_points, path='.' + f'/{i}')

if __name__ == '__main__':
    main_task_1_generation()