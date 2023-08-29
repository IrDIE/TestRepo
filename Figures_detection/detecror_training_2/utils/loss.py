import torch
import torch.nn as nn
from Figures_detection.detecror_training_2.utils.metrics import intersection_over_union

class lossYolov1(nn.Module):
    def __init__(self, path_size = 4, boxes = 2, classes = 4):
        super(lossYolov1, self).__init__()

        self.mse = nn.MSELoss(reduction="sum")
        self.path_sise = path_size
        self.classes = classes
        self.boxes = boxes

        self.lambda_noobj = 0.7 # 0.5 in paper
        self.lambda_coord = 8 # 5 in paper

    def forward(self, predictions, target):

        predictions = predictions.reshape(-1, self.path_sise, self.path_sise, \
                                          self.classes + self.boxes*5)

        iou_b1 = intersection_over_union(predictions[..., 5:9], target[..., 5:9])
        iou_b2 = intersection_over_union(predictions[..., 10:14], target[..., 5:9])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        iou_maxes, bestboxes = torch.max(ious, dim = 0)

        exists_box = target[..., 4].unsqueeze(3) # is there an object in sell i


        # for box coordinates
        box_predictions = exists_box * (
            (bestboxes * predictions[..., 10:14] + (1-bestboxes) * predictions[..., 5:9])
        )

        box_targets = exists_box * target[..., 5:9]

        box_predictions[..., 2:4] = (torch.sign(box_predictions[..., 2:4])) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] =  torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim= -2)
        )


        # for object loss
        pred_box = ( # where responsible box
            bestboxes * predictions[..., 9:10] + (1-bestboxes) * predictions[..., 4:5]
        )
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 4:5])
        )

        # for no object
        no_obj_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 4:5], start_dim = 1),
            torch.flatten((1-exists_box) * target[..., 4:5], start_dim=1)
        )

        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 9:10], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 4:5], start_dim=1)
        )

        # for class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :4], end_dim=-2),
            torch.flatten(exists_box * target[..., :4], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_obj_loss + class_loss * 0.8
        )

        return loss



