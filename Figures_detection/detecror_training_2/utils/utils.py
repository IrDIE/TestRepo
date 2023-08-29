import os
import json
import torch


def convert(size, box):

    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h) # return relative midpoint (x, y) and rel w and h (relative to img size )

def parse_json_to_yolo(path:str, path_save_txt : str):

    shapes_to_id = {
        "triangle" : 0,
        "hexagon": 1,
        "circle": 2,
        "rhombus": 3,

    }
    annotations = os.listdir(path)
    for ann in annotations:

        with open(path + '/' + ann) as f:
            json_data = json.load(f)

        with open(path_save_txt + '/' + ann[:-4] + 'txt', "w") as f:
            for box_img in json_data:
                name = box_img['name']
                class_id = shapes_to_id[name]
                x, y = int(box_img['region']['origin']['x']), int(box_img['region']['origin']['y'])
                width, height = int(box_img['region']['size']['width']), int(box_img['region']['size']['height'])
                x,y,w,h = convert((256, 256), (x, x+width, y, y+height))
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")





def save_checkpoint(epoch, model, optimizer,LOSS, path, filename="/my_checkpoint"):
    print("=> Saving checkpoint. LOSS = ", LOSS)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': LOSS,
    }, path + filename)


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("=> Loaded checkpoint")
    return model, optimizer

def update_dataset_info(path_annotations : str,dict_to_update:dict,  n_classes = 4):

    annts = os.listdir(path_annotations)
    n_imgs = len(annts)

    n_imgs_per_class = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,

    }



    for ann in annts:
        with open(path_annotations + '/' +ann, 'r') as fp:
            # opened image annotation file
            classes_in_img = []
            for line in fp: # read which classes image has
                cls = int(line[0])
                classes_in_img.append(cls)
            for key_cls in n_imgs_per_class.keys():
                if key_cls in classes_in_img:
                    n_imgs_per_class[key_cls] += 1

    for key in dict_to_update.keys():
        dict_to_update[key] += n_imgs_per_class[key]

    return dict_to_update