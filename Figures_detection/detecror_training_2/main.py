import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as F
from tqdm import tqdm
from Figures_detection.detecror_training_2.utils.loss import lossYolov1
from yolov_1_model import YOLO_model
from Figures_detection.detecror_training_2.utils.metrics import *
from Figures_detection.detecror_training_2.utils.utils import save_checkpoint, load_checkpoint, update_dataset_info
from Figures_detection.detecror_training_2.utils.dataloaders import get_loaders
import pandas as pd
import pickle

EPOCHS = 3
BATCH_SIZE = 4
LEARN_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY = 0
CHECK_PATH = './checkpoints'
RESIZE_SHAPE = 256
MODEL_VERSION = "v1" + ".pth.tar"
TEST_RUNS_RES = './runs/test'
TRAIN_RUNS_RES = './runs/train'
LABEL_DIR = './dataset_batches_generated'
TEST_SIZE_DATA = 12
TRAIN_SIZE_DATA = 6



class Compose(object):
    def __init__(self, tfs):
        self.tfs = tfs

    def __call__(self, img, bboxes):
        for t in self.tfs:
            img, bbox = t(img), bboxes
        return img, bbox



def get_iou(true_boxes, pred_boxes, n_classes = 4):
    iou_list = []
    for c in range(n_classes):
        detections = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        detections.sort(key=lambda x: x[2], reverse=True)
        total_true_bboxes = len(ground_truths)

        # print("total_true_bboxes = ", total_true_bboxes)
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format="midpoint",
                )
                iou_list.append(iou.item())

    return iou_list

def inference(model, loader, epoch, save_image_detected_res = None, save_dataframe_path = None):
    iou_per_batch = []
    all_pred_boxes, all_true_boxes = get_bboxes( loader = loader, model = model,iou_threshold = 0.5,\
                                                 threshold = 0.001, save_image_detected_res=save_image_detected_res)

    true_boxes = torch.Tensor(all_true_boxes)
    pred_boxes =  torch.Tensor(all_pred_boxes)
    iou_list = get_iou(true_boxes, pred_boxes)
    mean_iou = np.mean(iou_list)
    max_iou, min_iou = max(iou_list), min(iou_list)

    mAPrecision, precisions_list, recall_list = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, get_prec_rec = True)

    df = pd.DataFrame(data={
        "MODEL_VERSION" : MODEL_VERSION,
        "epoch": epoch,
        "mean_iou" : mean_iou ,
        "max_iou": max_iou,
        "min_iou": min_iou,
        "precision": np.mean(precisions_list),
        "recall": np.mean(recall_list),
        "mAP.0.5" : mAPrecision.item()
    },  index=[0])

    with pd.option_context('display.max_rows', 100,
                           'display.max_columns', 20,
                           'display.precision', 5
                           ):
        print(df)
    if save_dataframe_path is not None:
        df.to_csv(save_dataframe_path + f'/metrics_{MODEL_VERSION}__{epoch}.csv')



def train_function(yolo_model, train_loader, optimizer, loss_fn , model_version, load_check):
    loss_per_train = []
    try:
        yolo_model, optimizer = load_checkpoint(checkpoint_path= load_check, model=yolo_model, optimizer=optimizer)
    except:
        print("\n====================\n  looks like we gonna train from scratch  \n======================")

    n_images = 0
    n_img_per_class_epoch = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,

    }

    for i, (x, y) in enumerate(train_loader):
        n_images += (len(x))
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = yolo_model(x)
        loss = loss_fn(preds, y)
        print("\n## TRAIN LOSS ## = ", loss.item())
        optimizer.zero_grad()
        loss_per_train.append(loss.item())
        loss.backward()
        optimizer.step()
        n_img_per_class_epoch = update_dataset_info(path_annotations = LABEL_DIR + '/annotations_txt', dict_to_update=n_img_per_class_epoch)


    return sum(loss_per_train) / len(loss_per_train), n_img_per_class_epoch , n_images

def save_metainfo(n_img_per_class_epoch , n_images, epoch, MODEL_VERSION):
    info = {
        "n_images" : n_images,
        "n_img_per_class_epoch" : n_img_per_class_epoch
    }
    with open(TRAIN_RUNS_RES + f'/train_meta_{epoch}_{MODEL_VERSION}.pickle', 'wb') as handle:
        pickle.dump(info, handle)


def run(save_dataframe_path=TEST_RUNS_RES, load_check = ''):
    tfs = Compose([transforms.Resize((RESIZE_SHAPE, RESIZE_SHAPE)), transforms.ToTensor()])

    model =  YOLO_model(patch_size=4, n_boxes_per_img = 2, input_img_width = 256, n_classes = 4)
    train_loader, test_loader = get_loaders(tfs, BATCH_SIZE,  TRAIN_SIZE_DATA, TEST_SIZE_DATA)
    loss_fn = lossYolov1()
    opt = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)
    loss_per_epoch = []

    for epoch in range(EPOCHS):
        # #################
        # CALCULATE METRICS
        # #################
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print("mean_avg_prec = ", mean_avg_prec)

        avg_train_loss , n_img_per_class_epoch , n_images = train_function(yolo_model=model, train_loader = train_loader, \
                           optimizer = opt, loss_fn = loss_fn, model_version = MODEL_VERSION, load_check=load_check)

        loss_per_epoch.append(avg_train_loss)

        save_checkpoint(epoch = epoch, model = model, optimizer = opt, LOSS = avg_train_loss,\
                        path = CHECK_PATH, filename=f"/yolo_{epoch}_" + MODEL_VERSION )

        save_losses(loss_per_epoch, MODEL_VERSION)
        save_metainfo(n_img_per_class_epoch , n_images, epoch, MODEL_VERSION)

        if ( epoch % 2 == 0 ) or ( epoch == EPOCHS-1 ):
            inference(model, train_loader, epoch, save_dataframe_path=None)
            inference(model, test_loader, epoch, save_dataframe_path=save_dataframe_path, save_image_detected_res=save_dataframe_path)
            # for x, y in train_loader:  # see results during training
            #     x = x.to(DEVICE)
            #     for idx in range(1):
            #         bboxes = cellboxes_to_boxes(model(x))
            #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.5, box_format="midpoint")
            #         plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes, idx=idx)


if __name__ == '__main__':
    run()


