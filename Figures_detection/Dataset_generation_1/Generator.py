import cv2
import os
import sys
from numpy.random import randint, random
import numpy as np
import random
from Figures_detection.Dataset_generation_1.utils import convert
import json


class DataGenerator():

    def __init__(self, img_size : int = 256 ):
        self.img_size = img_size


    def generate_batch(self, batch_size : int , path_save: str):
        self.path_save = path_save
        pass

    def create_base_img(self):
        base_img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        base_img[:, :, 0] = randint(0, 256)
        base_img[:, :, 1] = randint(0, 256)
        base_img[:, :, 2] = randint(0, 256)
        return base_img


    def generate_image_ann(self, is_hexagon_required = False, n_points : int = 2, \
                           hexagon_no_need : bool = False,\
                            path_img : str = './task1_generated_dataset', path_annot : str =  './task1_generated_dataset'):
        # save to self.path_save
        img = self.create_base_img()
        img, points_list = self.generate_points(n_points=n_points, img = img)
        annotated_json = []
        for i in range(len(points_list)):
            if is_hexagon_required:
                # draw it
                img , meta_info = self.generate_hexagon(img, points_list[i])
                [x_max, y_max, x_min, y_min] = meta_info
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 1)
                is_hexagon_required = False
                annotated_json.append(convert(x_max, y_max, x_min, y_min, id = i, shape_name = -1))
                continue

            generators = {
                1 : self.generate_hexagon,
                2 : self.generate_triangle,
                3 : self.generate_circle,
                4 : self.generate_rhombus,


            }
            shape = randint(1,5) if not hexagon_no_need else randint(2,5)
            img, meta_info = generators[shape](img,  points_list[i])
            [x_max, y_max, x_min, y_min] = meta_info
            annotated_json.append(convert(x_max, y_max, x_min, y_min, id=i, shape_name=shape))
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 1)

        with open(path_annot + '.json', 'w') as outfile:
            json.dump(annotated_json, outfile, indent=4)


        cv2.imwrite(path_img + '.png', img)
        return img



    def generate_triangle(self, img, point):
        x, y = point
        x2, y2 = x + randint(28, 35), y + randint(2, 5) * randint(5, 8)
        x3, y3 = x + randint(5, 10)* randint(1, 2), y + randint(25, 30)* (-1 if random.random() > 0.5 else 1 )

        p1 = [x, y ]
        p2 = [x2, y2 ]
        p3 = [x3, y3 ]

        vertices = np.array([p1, p2, p3], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 20), thickness=1)
        img = cv2.fillPoly(img, [pts], color=(0, 0, 255))
        return img, [max(x, x2, x3), max(y, y2, y3), min(x, x2, x3), min(y, y2, y3)]


    def generate_circle(self, img, point, rad = randint(15, 22)):
        img = cv2.circle(img, point, rad, (0, 0, 0), -1)

        x_max, y_max, x_min, y_min = point[0] + rad, point[1]  + rad, point[0] - rad, point[1] - rad
        return img, [x_max, y_max, x_min, y_min]

    def generate_rhombus(self, img, point):
        len_romb = randint(10, 15)
        x, y = point[0], point[1]
        points_rotated = [[x - len_romb , y - len_romb], [x - len_romb , y + len_romb], \
                          [x + len_romb , y + len_romb] , [x + len_romb , y - len_romb]]
        points_rotated = self.rotate(points_rotated, randint(0, 30))
        img = cv2.drawContours(img, [points_rotated], 0, (155, 155, 155), -1, cv2.LINE_AA)
        #img = cv2.rectangle(img,points_rotated[0],points_rotated[1], (0, 11, 111), -1)

        return img, [max(points_rotated[:, 0]),max(points_rotated[:, 1]),min(points_rotated[:, 0]),min(points_rotated[:, 1])]

    def generate_hexagon(self, img, point):
        rad = randint(15, 20)
        x, y = point[0], point[1]
        h = rad
        points = np.array([[x + rad//2, y - h], [x + rad, y], [x + rad//2, y + h],\
                                [x - rad // 2, y + h], [x - rad, y], [x - rad // 2, y - h]])
        points = self.rotate(points, randint(0, 30))


        img = cv2.fillPoly(img, pts=[points], color=(0, 255, 0))

        return img, [points[1][0] , points[2][1] , points[4][0] , points[5][1] ]


    def generate_points(self, n_points : int, img ):
        #print(f"gonna generate {n_points} shapes")
        points_list = []
        switch_list = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
        random.shuffle(switch_list)
        for i in range(n_points):
            if i == 4:
                points_list.append(
                    [128 - randint(1, 20) * switch_list[-1][0], 128 - randint(1, 20) * switch_list[-1][1]])
                break
            else:
                x = 128 - randint(36, 70) * switch_list[i][0]
                y = 128 - randint(36, 90) * switch_list[i][1]
                points_list.append([x, y])


        for point in points_list:
            img = cv2.circle(img, (point[0], point[1]), radius=0, color=(0, 0, 255), thickness=0)
        return img, points_list

    def rotate(self, points, angle):
        ANGLE = np.deg2rad(angle)
        c_x, c_y = np.mean(points, axis=0)
        return np.array(
            [
                [
                    c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                    c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)
                ]
                for px, py in points
            ]
        ).astype(int)




