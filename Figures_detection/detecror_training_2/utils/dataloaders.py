import random
from torch.utils.data import DataLoader
import torch

from PIL import Image
from Figures_detection.Dataset_generation_1.Generator  import DataGenerator
from Figures_detection.detecror_training_2.utils.utils import parse_json_to_yolo

LABEL_DIR = './dataset_batches_generated'
IMG_DIR = './dataset_batches_generated/imgs'


class ShapeDatasetFromGenerator(torch.utils.data.Dataset):
    def __init__(self,total_len_annoattions, img_dir = IMG_DIR, label_dir = LABEL_DIR , \
                 patch_size = 4, boxes = 2, classes = 4, transform = None, is_hexagon_required = False, hexagon_no_need = False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.boxes = boxes
        self.classes = classes
        self.transform = transform
        self.len_annotations = total_len_annoattions
        self.generator = DataGenerator()
        self.is_hexagon_required = is_hexagon_required
        self.hexagon_no_need = hexagon_no_need


    def __len__(self):
        return self.len_annotations

    def __getitem__(self, index):
        n_points = random.randint(1, 6)
        self.generator.generate_image_ann(n_points=n_points, path_img=self.img_dir + f'/{index}',\
         path_annot=self.label_dir + '/annotations'+ f'/{index}', is_hexagon_required = self.is_hexagon_required,  hexagon_no_need = self.hexagon_no_need)
        parse_json_to_yolo(path=self.label_dir + '/annotations', path_save_txt= self.label_dir + '/annotations_txt')

        label_path = self.label_dir + '/annotations_txt' + f'/{index}.txt'
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) for x in label.replace("\n", "").split()
                ]
                if x > 0.99:
                  x = 0.99
                if y > 0.99:
                  y = 0.99

                boxes.append([class_label, x, y, width, height])

        img_path = self.img_dir + f'/{index}.png'
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.patch_size, self.patch_size, self.classes + 5*self.boxes))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.patch_size * y), int(self.patch_size * x)
            if (i == 4 or j == 4):
              print("index = ", index)
              print("i, j = ", i, j)
              print("class_label, x, y, width, height = ", class_label, x, y, width, height)
            
            x_cell, y_cell = self.patch_size * x - j, self.patch_size * y - i

            width_cell, height_cell = (
                width * self.patch_size,
                height * self.patch_size,
            )

            if label_matrix[i, j, 4] == 0:
                # Set that there exists an object
                label_matrix[i, j, 4] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 5:9] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


def get_loaders(transform, batch_size, TRAIN_SIZE_DATA, TEST_SIZE_DATA):
    train_dataset = ShapeDatasetFromGenerator(transform = transform, total_len_annoattions = TRAIN_SIZE_DATA)
    test_dataset = ShapeDatasetFromGenerator(transform = transform, total_len_annoattions = TEST_SIZE_DATA)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False)

    return train_loader, test_loader

def try_loader():
    dataloader = ShapeDatasetFromGenerator()
    image, label_matrix = dataloader.__getitem__(1)
    print(image)
    print("###############")
    print(label_matrix)






#try_loader()





