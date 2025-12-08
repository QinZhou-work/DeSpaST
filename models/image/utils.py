import os
import re
import skimage
import skimage.io
import cv2
import random
import pickle as pkl
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import torchvision.models.detection as tmdet
from torch import nn, Tensor

from torch.jit.annotations import Tuple, List, Dict, Optional
from torchvision.models.detection.image_list import ImageList

import sys
sys.path.append("/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/projects/mrcnn_pkg")
from utils_image import rgba2rgb, Crop#, split_masks, binary_mask_to_polygon
from torch_layers import deep_update, BoxPredictor, MaskPredictor, KeypointPredictor
import utils_image_dannie as uimg_d
import mrcnn as mrcnn
import networks
import copy
import evaluator
import shidan_evaluator as eva



DEFAULT_DATASET_CONFIG = {
    "output_size": (256,256),
    "normalizer": uimg_d.RGB_Normalizer(mean=[168.05047310350506, 168.05047310350506, 162.54708725145446],
                                 std=[51.83780744440758, 51.83780744440758, 43.12277861396769],
                                 inverse=False),
    "shape_augmenter": uimg_d.shape_augmenter,
    "color_augmenter": uimg_d.Color_Augmenter(set=2),#Color_Augmenter(set=2),#
    "resolution": None
    }
normalizer = uimg_d.RGB_Normalizer(inverse=True) # normalizer used to visualization


## Default model config
MODEL_CONFIG = {
    'transform': {
        'min_size': 256, #(640, 800, 960, 1024,),
        'max_size': 256, #800, #1024,
        'image_mean': [0.0, 0.0, 0.0],
        'image_std': [1.0, 1.0, 1.0],
        # 'size_divisible': 32,
        # 'fixed_size': None,
    },
}

## Training Config
DEFAULT_TRAIN_CONFIG = {
    'pretrained_mrcnn': True,
    'num_epochs': 1000,
    'batch_size': 4,
    'workers': 2,
    'lr': 0.001,
    'class_weights': [1., 1., 1., 1., 1., 1., 1., 1.],  # BG + 7 classes
    'ignore_index': 100,
    # 'optimizer': ('SGD', {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4}),  # use 0.001 for sgd
    # 'optimizer': ('SGD', {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 1e-4}),  # use 0.0001 for retina
    # 'optimizer': ('Adam', {'lr': 0.001, 'betas': (0.5, 0.999), 'weight_decay': 0.01, 'amsgrad': False}),
    # 'optimizer': ('AdamW', {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01, 'amsgrad': False}),
    # 'clipnorm': None,
    # 'lr_scheduler': ('MultiStepLR', {'milestones': [200, 400, 600, 800], 'gamma': 0.5}),
    # 'lr_scheduler': ('CosineAnnealingLR', {'T_max': 1000}),
    # 'lr_scheduler': ('CosineAnnealingWarmRestarts', {'T_0': 100, 'T_mult': 2, 'eta_min': 0.}),
    # 'reduce_lr_on_plateau': ('overall', {'mode': 'min', 'factor': 0.5, 'patience': 100,
    #                                      'verbose': False, 'threshold': 1e-4}),
    'epoch0': 0,
    'save_freq': 1,
    'disp_freq': 100,
    'eval_batch_size': 4,
    'eval_score_thresh': 0.3,
    'eval_iou_thresh': 0.5,
#     'eval_class_weights': [1., 1., 1., 0., 1., 0.],  # we ignore blood nuclei and dead nuclei in eval
}


COLORS = {'air space': [180, 180, 180],
          'normal nuclei': [144, 0, 255],
          'normal region': [198, 123, 255],
          'tumor nuclei': [0, 255, 0],
          'tumor region': [0, 199, 47],
          'stroma nuclei': [255, 0, 0],
          'stroma region': [255, 108, 42],
          'lymphocyte nuclei': [0, 0, 255],
          'lymphocyte region': [0, 43, 134],
          'macrophage nuclei': [255, 255, 0],
          'macrophage region': [221, 221, 121],
          'red blood cell': [255,   0, 255],
          'blood vessel': [],
          'bronchus': [],
          'cartilage': [],
          'gland': [],
          'hemorrhage': [],
          'keratinization': [],
          'dead nuclei': [0, 148, 225],
          'necrosis': [0, 255, 255],
          'apoptotic body': [50, 148, 255],
          'fibroblast': [255, 148, 0],
          'mitotic figure': [100, 255, 0],
          'plasma cell': [0, 0, 200],
          'ductal epithelium': [100, 0, 255],
          'eosinophil': [0, 100, 100],
          'myoepithelium': [255, 100, 0],
          'neutrophil': [0, 50, 150],
          'vascular endothelium': [255, 50, 255],
          'unlabeled': [148, 148, 148]
          }

# modeled classes
CLASSES  = ['tumor nuclei', 'stroma nuclei', 'lymphocyte nuclei', 'red blood cell', 'macrophage nuclei',
           'dead nuclei', 'ductal epithelium']

# all classes
CLASSES_ = CLASSES+['unlabeled']


cell_type_convert = {
    "apoptotic body": "dead nuclei",
    "fibroblast": "stroma nuclei",
    "mitotic figure": "tumor nuclei",
    "plasma cell": "lymphocyte nuclei",
    "eosinophil": "lymphocyte nuclei",
    "myoepithelium": "stroma nuclei",
    "neutrophil": "lymphocyte nuclei",
    "vascular endothelium": "stroma nuclei"
}


def generate_ids(labeled_df, img_folder, mask_folder, masks_folder, image_dict, ratio=1, discard_files=[]):  #
    if ratio < 1:
        num = round(len(labeled_df) * ratio)
        f_ids = random.sample(range(len(labeled_df)), num)
    else:
        f_ids = list(range(len(labeled_df)))

    indexes = []
    for index in f_ids:
        if labeled_df.loc[index, 0] not in discard_files:
            image_name = labeled_df.loc[index, 0] + ".png"
            image_note = str(labeled_df.loc[index, 1])

            if not re.match(r'\*|#', image_name):
                _dict = dict()
                _dict['image_path'] = os.path.join(img_folder, image_name)
                img    = skimage.io.imread(os.path.join(img_folder, image_name))
                mask   = skimage.io.imread(os.path.join(mask_folder, image_name))
                loaded = pkl.load(open(os.path.join(masks_folder, image_name.replace(".png", ".pkl")), "rb"))
                masks     = loaded['masks']
                class_nms = loaded['class_nms']
                lb_typs   = loaded['lb_typs']

                _dict['image']     = img
                _dict['mask_rgba'] = mask
                _dict['masks']     = masks
                _dict['class_nms'] = class_nms
                _dict['lb_typs']   = lb_typs

                if len(lb_typs) != masks.shape[2]:
                    import pdb; pdb.set_trace()
                indexes.append(image_name)
                if '20' in image_note:
                    _dict['magnitude'] = 20
                else:
                    _dict['magnitude'] = 40
                image_dict[image_name] = _dict

    return indexes, image_dict


def objects_to_tensor_targets(masks, labels, image_id):
    """
    Args:
        masks
        labels
        image_id
    """

    num_objs = len(labels)
    # masks = [skimage.segmentation.clear_border(_ > 0) for _ in masks]
    h, w = masks[0].shape
    masks = [_ > 0 for _ in masks]

    bboxes = []
    to_remove = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        if np.max(pos[1]) > np.min(pos[1]) and np.max(pos[0]) > np.min(pos[0]):
            bboxes.append([np.min(pos[1]), np.min(pos[0]),
                           np.max(pos[1]), np.max(pos[0])])
        else:
            to_remove.append(i)
    bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
    labels = torch.as_tensor([_ for i, _ in enumerate(labels) if i not in to_remove], dtype=torch.int64)
    masks = torch.as_tensor([_ for i, _ in enumerate(masks) if i not in to_remove], dtype=torch.uint8)

    if len(bboxes):
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
    else:
        area = torch.tensor([], dtype=torch.float32)
        bboxes = torch.empty(0, 4, dtype=torch.float32)
        labels = torch.tensor([], dtype=torch.int64)
        masks = torch.empty(0, h, w, dtype=torch.uint8)
    iscrowd = torch.zeros((num_objs - len(to_remove),), dtype=torch.int64)
    image_id = torch.tensor([image_id])

    return {"boxes": bboxes, "labels": labels, "masks": masks,
            "area": area, "image_id": image_id, "iscrowd": iscrowd}, to_remove


def convert_cell_type(masks, from_type, to_type):
    """a generalization of previous function 'normal_cell_to_stroma'. """
    in_val = COLORS[from_type]
    out_val = COLORS[to_type]
    masks[np.all(masks == in_val, axis=-1), :] = out_val
    return masks


class Dataset(torch.utils.data.Dataset):
    __initialized = False

    def __init__(self, indices, image_dict, classes, class_ids, val_to_label,
                 normalize=True, augment=False, cell_type_convert=None, **kwargs):
        """
        Args:
        indices
        image_dict
        classes: a list modeled class in correct order, i.e. the 1st item, classes[0] will have class label 1

        """
        self.indices = indices
        self.image_dict = image_dict
        self.classes = classes
        self.class_ids = class_ids
        self.normalize = normalize
        self.augment = augment
        self.val_to_label = val_to_label
        self.cell_type_convert = cell_type_convert

        self.kwargs = kwargs
        self.res = self.kwargs["resolution"] if "resolution" in self.kwargs.keys() else None
        self.crop_size = self.kwargs["output_size"]
        self.normalizer = self.kwargs["normalizer"] if "normalizer" in self.kwargs.keys() else None
        self.shape_augmenter = self.kwargs["shape_augmenter"] if "shape_augmenter" in self.kwargs.keys() else None
        self.color_augmenter = self.kwargs["color_augmenter"] if "color_augmenter" in self.kwargs.keys() else None

    def __len__(self):
        """Denotes the number of samples"""
        return len(self.indices)

    def __getitem__(self, index):
        """Generate one batch of data.

        Returns:
            idx: indexes of samples (long)
        """

        # Generate indexes of the batch
        data_index = self.indices[index]

        #         print(self.image_dict[data_index].keys())

        data = dict()

        data['img'], data['target'], data['rgba_mask'] = self.__data_generation(data_index, verbose=True)
        data['paths'] = self.image_dict[data_index]['image_path']
        data['magnitude'] = self.image_dict[data_index]['magnitude']
        data['pad_width'] = self.image_dict[data_index]['pad_width']

        return data

    def __data_generation(self, index, idx=0, verbose=False):
        """Generates image containing batch_size samples.

        Returns:
            image: [b, ch, h, w]
        """

        if self.res is not None and self.image_dict[index]['magnitude'] != self.res:  # (for inferencing: always load images of the same magnitude)
            image, target = [], []
        else:
            #             print(index)
            image = rgba2rgb(self.image_dict[index]['image'])
            image_path = self.image_dict[index]['image_path']
            masks = self.image_dict[index]['masks']
            class_nms = self.image_dict[index]['class_nms'].astype(object)
            rgb_mask = self.image_dict[index]['mask_rgba']
            lb_typs = self.image_dict[index]['lb_typs']

            if masks is not None:
                rgb_mask = rgba2rgb(rgb_mask)
                if self.cell_type_convert is not None:
                    for from_type in self.cell_type_convert.keys():
                        to_type = self.cell_type_convert[from_type]
                        rgb_mask = convert_cell_type(rgb_mask, from_type, to_type)
                        for x in np.where(class_nms == from_type)[0]:
                            class_nms[x] = to_type

                labels = [self.class_ids[self.classes.index(class_nm)] for class_nm in class_nms]

            # Normalization
            if self.normalize:
                image = self.normalizer(image)

            # Augmentation
            if self.augment:
                # shape augmentation
                image, masks = self.shape_augmenter(image, masks)
                rgb_mask = self.shape_augmenter(rgb_mask)[0]

                # color augmentation
                image = self.color_augmenter(image)

            # Resize
            if self.image_dict[index]['magnitude'] == 20:
                image = skimage.transform.rescale(image, 2, preserve_range=True, multichannel=True, order=1)
                if masks is not None:
                    masks = skimage.transform.rescale(masks, 2, preserve_range=True, multichannel=True, order=0)
                    rgb_mask = skimage.transform.rescale(rgb_mask, 2, preserve_range=True, multichannel=True, order=1)
                # print("resize")

            # Random crop
            image, masks, rgb_mask = Crop(size=(self.crop_size[0], self.crop_size[1]), pos='random')([image, masks, rgb_mask])

            # Padding
            padder = uimg_d.Pad(size=(self.crop_size[0], self.crop_size[1]), mode='constant')
            image = padder(image)
            masks = padder(masks)
            rgb_mask = padder(rgb_mask)

            self.image_dict[index]['pad_width'] = padder.pad_width

            # Filter small objects & create target for MaskRCNN
            if masks is not None:
                non0s = [(masks[..., i], labels[i], lb_typs[i]) for i in range(len(labels)) if
                         np.sum(masks[..., i]) > 10]
                if len(non0s):
                    masks, labels, lb_typs = zip(*non0s)

                else:
                    masks = [np.zeros((self.crop_size[0], self.crop_size[1]))]
                    labels = []
                    lb_typs = []
                target, to_remove = objects_to_tensor_targets(masks, labels, idx)
                if len(lb_typs) > 0:
                    lb_typs = [_ for i, _ in enumerate(lb_typs) if i not in to_remove]
                target['lb_typs'] = np.array(lb_typs, dtype=object)  # add a new key-item to target
                #                 print(f"{len(target['lb_typs'])}, {target['masks'].shape[0]}")
                assert (len(target['lb_typs']) == target['masks'].shape[0])

            image = torch.tensor(np.transpose(image, (2, 0, 1)).astype(float))

        return image, target, rgb_mask


# re-define fastrcnn loss and maskrcnn loss
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, class_weights=None, ignore_index=-100):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.
    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    sizes = [x.shape[0] for x in regression_targets]
    regression_targets = torch.cat(regression_targets, dim=0)
    pred_labels = torch.argmax(class_logits, dim=1)
    pred_labels_ = list(torch.split(pred_labels, sizes))

    classification_loss = F.cross_entropy(class_logits, labels, weight=class_weights, ignore_index=ignore_index)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    pred_labels_pos = pred_labels[sampled_pos_inds_subset]

    # use predicted label
    labels_pos_ = copy.deepcopy(labels_pos)
    ids = torch.where(labels_pos == ignore_index)[0]
    labels_pos_[ids] = pred_labels_pos[ids]

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos_],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss, pred_labels_


def maskrcnn_loss(mask_logits,
                  proposals,
                  gt_masks,
                  gt_labels,
                  mask_matched_idxs,
                  pred_lbls,
                  device,
                  sample_weights=None):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])
    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    sample_ws = [sample_weight[idxs] for sample_weight, idxs in zip(sample_weights, mask_matched_idxs)]
    mask_targets = [
        tmdet.roi_heads.project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    pred_lbls = torch.cat(pred_lbls, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    sample_weights = torch.unsqueeze(torch.cat(sample_ws, dim=0), 2).to(device)
    #     import pdb; pdb.set_trace()
    #     if sum([len(torch.where(x==0)[0]) for x in sample_ws]) > 0:
    #         print('0 exists in sample_weights to calculate mask loss')

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    max_idx = mask_logits.shape[1] - 1
    labels_ = copy.deepcopy(labels)
    ids = torch.where(labels_ > max_idx)[0]
    labels_[ids] = pred_lbls[ids]

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels_], mask_targets, weight=sample_weights
    )
    return mask_loss


class RoIHeads(tmdet.roi_heads.RoIHeads):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 class_weights=None,
                 ignore_index=-100):
        super(RoIHeads, self).__init__(box_roi_pool,
                                       box_head,
                                       box_predictor,
                                       # Faster R-CNN training
                                       fg_iou_thresh, bg_iou_thresh,
                                       batch_size_per_image, positive_fraction,
                                       bbox_reg_weights,
                                       # Faster R-CNN inference
                                       score_thresh,
                                       nms_thresh,
                                       detections_per_img)
        #         super(RoIHeads, self).__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self,
                features,  # type: Dict[str, Tensor]
                proposals,  # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                device,
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        #         print(f"ignore index in ROIHeads: {self.ignore_index}")
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            #             if sum([len(torch.where(x['labels']==ignore_idx)[0]) for x in targets]) > 0:
            #                 print(f"{ignore_idx} exists when entering in roi heads")
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            #             if sum([len(torch.where(x['labels']==ignore_idx)[0]) for x in targets]) > 0:
            #                 print(f"{ignore_idx} exists before calculating fastrcnn_loss")

            loss_classifier, loss_box_reg, pred_lbls_ = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets, self.class_weights, self.ignore_index)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                pred_lbls = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
                    pred_lbls.append(pred_lbls_[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                lb_typs = [t["lb_typs"] for t in targets]
                sample_weights = []
                for lb_typ in lb_typs:
                    weights = torch.ones((len(lb_typ), 1))
                    weights[np.where(lb_typ == "rectangle")] = 0
                    sample_weights.append(weights)
                #                 try:
                #                     print(f"min index for calculating mask loss: {min([min(x) for x in gt_labels if len(x)>0])}; max index for calculating mask loss: {max([max(x) for x in gt_labels if len(x)>0])}")
                #                 except:
                #                     import pdb; pdb.set_trace()
                #                 import pdb; pdb.set_trace()
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs,
                    pred_lbls, device, sample_weights)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = tmdet.roi_heads.maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = tmdet.roi_heads.keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses


class MaskRCNN_NuCLS(nn.Module):
    def __init__(self, backbone, num_classes, class_weights, masks=False, keypoints=None, config={},
                 pretrained=False, device=torch.device("cuda:0"), ignore_index=None):
        super(MaskRCNN_NuCLS, self).__init__()
        self.config = self.default_config(num_classes, masks, keypoints)

        mrcnn.deep_update(self.config, config)
        self.config['featmap_names'] = self.config['featmap_names'] or backbone.featmap_names
        self.config['in_channels'] = self.config['in_channels'] or backbone.out_channels
        self.class_weights = class_weights
        print(f"class weights: {class_weights}, \nignore_index: {ignore_index}")
        self.ignore_index = ignore_index

        self.rpn = self.get_rpn()
        self.roi_heads = self.get_roi_heads()
        self.transform = self.get_transform()
        self.backbone = backbone

        # define different conv1 layers
        self.conv1_40x = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1_40x.load_state_dict(self.backbone.body.conv1.state_dict())

        self.conv1_20x = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.conv1_20x.load_state_dict(self.backbone.body.conv1.state_dict())

        # used only on torchscript mode
        self._has_warned = False
        self.device = device
        self.to(self.device)

        if pretrained:
            self.load_pretrain(pretrained)

    @torch.jit.unused
    def eager_outputs(self, losses, detections, transfered_images):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections, transfered_images

    def forward(self, images, targets=None, is_20x=False):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor of shape [N, 4], "
                                         "got {:}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                # print(boxes)
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenrate box
                    bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invaid box {} for target at index {}.".format(degen_bb, target_idx))

        if is_20x:
            # Resize image
            _images = ImageList(nn.functional.interpolate(images.tensors, scale_factor=0.5), images.image_sizes)

            # Shift resnet 1st conv2d to stride = 1
            self.backbone.body.conv1 = self.conv1_20x
        else:
            _images = images

            # Shift resnet 1st conv2d to stride = 2
            self.backbone.body.conv1 = self.conv1_40x
        self.backbone.to(self.device)

        features = self.backbone(_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

#         if sum([len(torch.where(x['labels']==ignore_idx)[0]) for x in targets]):
#             print(f"{ignore_idx} exists before entering roi_heads")
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, self.device, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return (losses, detections)

    def get_transform(self):
        return tmdet.transform.GeneralizedRCNNTransform(**self.config['transform'])

    def get_rpn(self):
        in_channels = self.config['in_channels']
        rpn_params = self.config['rpn_params']

        rpn_anchor = tmdet.rpn.AnchorGenerator(**rpn_params['anchor'])
        rpn_header = tmdet.rpn.RPNHead(in_channels, rpn_anchor.num_anchors_per_location()[0])
        rpn = tmdet.rpn.RegionProposalNetwork(rpn_anchor, rpn_header, **rpn_params['rpn'])

        return rpn

    def get_roi_heads(self):
        featmap_names = self.config['featmap_names']
        in_channels = self.config['in_channels']
        roi_params = self.config['roi_params']

        ## box header
        box_header = BoxPredictor(in_channels, featmap_names, **roi_params['box'])

        ## roi heads
        roi_heads = RoIHeads(
            box_roi_pool=box_header.box_roi_pool,
            box_head=box_header.box_head,
            box_predictor=box_header.box_predictor,
            class_weights=self.class_weights,
            ignore_index=self.ignore_index,
            **roi_params['roi']
        )

        ## add mask header
        if 'mask' in roi_params:
            mask_header = MaskPredictor(in_channels, featmap_names, **roi_params['mask'])
            roi_heads.mask_roi_pool = mask_header.mask_roi_pool
            roi_heads.mask_head = mask_header.mask_head
            roi_heads.mask_predictor = mask_header.mask_predictor

        ## add keypoint header
        if 'keypoint' in roi_params:
            keypoint_header = KeypointPredictor(in_channels, featmap_names, **roi_params['keypoint'])
            roi_heads.keypoint_roi_pool = keypoint_header.keypoint_roi_pool
            roi_heads.keypoint_head = keypoint_header.keypoint_head
            roi_heads.keypoint_predictor = keypoint_header.keypoint_predictor

        return roi_heads

    def load_pretrain(self, pretrained):
        if isinstance(pretrained, str):
            weights = torch.load(pretrained, map_location=self.device)

            # TODO: use a better condition
            if self.roi_heads.box_predictor.cls_score.weight.shape[0] != \
                    weights["roi_heads.box_predictor.cls_score.weight"].shape[0]:
                with torch.no_grad():
                    self.roi_heads.box_predictor.cls_score.weight[0:7, :] = \
                        weights["roi_heads.box_predictor.cls_score.weight"]

                    self.roi_heads.box_predictor.cls_score.bias[0:7] = \
                        weights["roi_heads.box_predictor.cls_score.bias"]

                    self.roi_heads.box_predictor.bbox_pred.weight[np.r_[0:7, 8:15, 16:23, 24:31], :] = \
                        weights["roi_heads.box_predictor.bbox_pred.weight"]

                    self.roi_heads.box_predictor.bbox_pred.bias[np.r_[0:7, 8:15, 16:23, 24:31]] = \
                        weights["roi_heads.box_predictor.bbox_pred.bias"]

                    self.roi_heads.mask_predictor.mask_fcn_logits.weight[0:7, ...] = \
                        weights["roi_heads.mask_predictor.mask_fcn_logits.weight"]

                    self.roi_heads.mask_predictor.mask_fcn_logits.bias[0:7] = \
                        weights["roi_heads.mask_predictor.mask_fcn_logits.bias"]


        else:
            if self.roi_heads.has_mask():
                m = tmdet.maskrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            elif self.roi_heads.has_keypoint():
                m = tmdet.keypointrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            else:
                m = tmdet.fasterrcnn_resnet50_fpn(
                    pretrained=True, progress=False, pretrained_backbone=False)
            weights = m.state_dict()

        try:
            # remove backbone from state_dict
            w = {k: v for k, v in weights.items()
                 if not k.startswith('backbone.body')}
            self.load_state_dict(w, strict=False)
        except RuntimeError as e:
            print(e)

    def default_config(self, num_classes, masks, keypoints):
        config = {
            ## backbone
            'featmap_names': None,
            'in_channels': None,
            ## fpn
            'fpn_params': {
                'out_channels': 256,
            },
            ## rpn
            'rpn_params': {
                'anchor': {
                    'sizes': [[32], [64], [128], [256], [512]],
                    'aspect_ratios': [[0.5, 1.0, 2.0]] * 5,
                },
                'rpn': {
                    'fg_iou_thresh': 0.7,
                    'bg_iou_thresh': 0.3,
                    'batch_size_per_image': 256,
                    'positive_fraction': 0.5,

                    'pre_nms_top_n': {'training': 2000, 'testing': 1000},
                    'post_nms_top_n': {'training': 2000, 'testing': 1000},
                    'nms_thresh': 0.7,
                },
            },
            ## roi
            'roi_params': {
                ## roi predictor
                'roi': {
                    # Faster R-CNN training
                    'fg_iou_thresh': 0.5,
                    'bg_iou_thresh': 0.5,
                    'batch_size_per_image': 512,
                    'positive_fraction': 0.25,
                    'bbox_reg_weights': None,
                    # Faster R-CNN inference
                    'score_thresh': 0.05,
                    'nms_thresh': 0.5,
                    'detections_per_img': 1000,
                },
                ## box predictor
                'box': {
                    'num_classes': num_classes,
                    'roi_output_size': 7,
                    'roi_sampling_ratio': 2,
                    'layers': [1024, 1024],
                },
            },
            ## transform
            'transform': {
                'min_size': 800, 'max_size': 1333,
                'image_mean': [0.485, 0.456, 0.406],
                'image_std': [0.229, 0.224, 0.225],
            }
        }
        if masks:
            ## mask predictor
            config['roi_params']['mask'] = {
                'num_classes': num_classes,
                'roi_output_size': 14,
                'roi_sampling_ratio': 2,
                'layers': [256, 256, 256, 256],
                'dilation': 1,
                'dim_reduced': 256,
            }
        if keypoints:
            ## keypoint predictor
            config['roi_params']['keypoint'] = {
                'num_keypoints': keypoints,
                'roi_output_size': 14,
                'roi_sampling_ratio': 2,
                'layers': [512] * 8,
            }

        return config


class MaskRCNNModel_NuCLS(mrcnn.MaskRCNNModel):
    def __init__(self, config_mrcnn, classes, is_train=True, device=torch.device("cuda:0"), pretrained_mrcnn=True,
                 lr=0.001, beta1=0.5, backbone_name='resnet101', save_dir="./models/", n_epochs=500,
                 n_epochs_decay=100, class_weights=None, ignore_index=None):
        self.is_train = is_train
        self.device = device
        self.save_dir = save_dir  # which folder to save models
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.model_names = ['G']
        num_class = len(classes) + 1
        if class_weights is None:
            class_weights = [1. for x in range(num_class)]
        self.class_weights = torch.tensor(class_weights).to(self.device)
        assert len(self.class_weights) == num_class

        # define networks
        resnet_fpn = tmdet.backbone_utils.resnet_fpn_backbone(backbone_name, pretrained=True)
        resnet_fpn.featmap_names = ['0', '1', '2', '3']
        resnet_fpn = resnet_fpn.to(self.device)

        self.netG = MaskRCNN_NuCLS(resnet_fpn, num_classes=num_class,
                                   class_weights=self.class_weights, masks=True,
                                   config=config_mrcnn, pretrained=pretrained_mrcnn,
                                   device=self.device, ignore_index=ignore_index)
        self.netG = self.netG.to(self.device)

        if self.is_train:
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.schedulers = [networks.get_scheduler(
                optimizer, lr_policy="linear", n_epochs=n_epochs, n_epochs_decay=n_epochs_decay)
                for optimizer in self.optimizers]

    def set_input(self, data, is_valid=False):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.img = torch.stack([_['img'] for _ in data]).float().to(self.device)
        self.image_paths = [_['paths'] for _ in data]
        # sample_weights = []
        for _ in data:
            # ws = []
            if _['target'] is not None:
                for key in list(_['target'].keys()):
                    if key == 'lb_typs':
                        pass
                    #     # weight = torch.ones((len(_['target'][key]), 1))
                    #     # weight[np.where(_['target'][key] == "rectangle")] = 0
                    #     #
                    #     # ws += weight
                    else:
                        _['target'][key] = _['target'][key].to(self.device)
            # ws = torch.tensor(ws).to(self.device)
            # sample_weights.append(ws)

        #         for _ in data:
        #             if len(torch.where(_['target']['labels']==ignore_idx)[0]):
        #                 print(f"{ignore_idx} exists when set input")
        self.target = [_['target'] for _ in data]
        # self.sample_weights = sample_weights

        # assumption: if not in training mode, batch size is always 1? No. But normally a batch always have the same magnitude
        if is_valid:
            self.is_20x = True if data[0]['magnitude'] == 20 else False
            # self.sample_weights = None  # remove sample weights if not in training mode
        else:
            if np.random.rand() < 0.5:
                self.is_20x = True
            else:
                self.is_20x = False

    def forward(self, val=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.netG.train()

        self.mask_loss, self.mask_pred = self.netG(self.img, self.target, is_20x=self.is_20x)

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                optimizer = getattr(self, 'optimizer_' + name)

                if isinstance(net, torch.nn.DataParallel):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if self.is_train:
                    optimizer = getattr(self, 'optimizer_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict['model_state_dict'])
                setattr(self, 'net' + name, net)
                if self.is_train:
                    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                    setattr(self, 'optimizer_' + name, optimizer)


def eval_output(outputs, targets_gt, class_ids_model, ignore_idx=None, thres_mask=0.5, thres_iou=0.5, visualize=False,
                class_nms=None, verbose=False):
    # evaluate accuracy
    n_cells_gt = []
    n_matched = []
    n_cells_excluding_blood_necrosis = []
    n_cells_matched_excluding_blood_necrosis = []
    iou_scores_matched_all = []
    classes_matched_all = []
    classes_matched_gt_all = []
    # classes_matched_gt_total_all = []
    for (output, target) in zip(outputs, targets_gt):
        # gt
        masks_gt = target['masks'].permute(1, 2, 0).numpy()
        lb_typs = target['lb_typs']
        class_ids_gt = target['labels'].detach().cpu().numpy()

        # ot
        masks = output['masks']
        class_ids = output['labels']
        boxes = output['boxes']
        scores = output['scores']

        masks = masks.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        masks = ((masks >= thres_mask) * 1).astype(np.int)
        class_ids = class_ids.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

        ## post process mrcnn result
        # 1) filter output based on overlap
        occlusion = np.ones(np.shape(masks)[0:2])
        to_keep = []
        for j in range(masks.shape[2]):
            # Remove the mask with tooooo much overlapping with previously detected region
            original_area = np.sum(masks[:, :, j])
            masks[:, :, j] = masks[:, :, j] * occlusion
            new_area = np.sum(masks[:, :, j])
            if new_area > 0 and new_area / original_area > 0.6:
                to_keep.append(True)
                occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, j]))
            else:
                to_keep.append(False)

        if sum(to_keep) > 1:
            masks = masks[:, :, to_keep]
            class_ids = class_ids[to_keep]
            boxes = boxes[to_keep]
            scores = scores[to_keep]

        # make masks_box from boxes
        boxes = boxes.astype(np.int64)
        r, c, _ = masks.shape
        masks_box = []

        for box in boxes:
            x1, y1, x2, y2 = box
            vertices = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]

            mask_cell = np.zeros((r, c), dtype=np.uint8)
            cv2.drawContours(mask_cell, [np.array(vertices)], 0, 1, -1)
            masks_box.append(mask_cell.astype(np.int64))

        masks_box = np.transpose(np.array(masks_box), (1, 2, 0))

        ############## Evaluation ###############

        # iou scores

        idx_rec = np.where(lb_typs == 'rectangle')[0]
        # idx_pol = np.where(lb_typs == 'polyline')[0]

        iou_scores, _, _, n_cell_gt, n_cell, _ = evaluator.compute_overlaps_weights(masks_gt, masks)
        iou_bbox, _, _, _, _, _ = evaluator.compute_overlaps_weights(masks_gt, masks_box)

        iou_scores[idx_rec, :] = iou_bbox[idx_rec, :]

        max_iou_score = np.max(iou_scores, axis=1)
        max_iou_id = np.argmax(iou_scores, axis=1)
        matched = np.where(max_iou_score > thres_iou)[0]

        # Coverage

        covg = len(matched) / n_cell_gt
        if covg == 0:
            print("zero coverage")
            continue
        else:
            if verbose:
                print("Coverage: {}".format(covg))

            classes_matched = class_ids[max_iou_id[matched]]
            try:
                classes_matched_gt = class_ids_gt[matched]
            except:
                import pdb;
                pdb.set_trace()

            # Plot confusion matrix
            try:
                cm = confusion_matrix(classes_matched_gt, classes_matched, labels=class_ids_model)
            except:
                import pdb;
                pdb.set_trace()

            if visualize:
                plt.figure(figsize=(5, 3))
                eva.plot_confusion_matrix(
                    cm, classes=class_nms)
            if verbose:
                print("Accuracy: {}".format(np.sum(np.diag(cm)) / np.sum(cm)))
        # Set values
        n_cells_gt.append(n_cell_gt)
        n_matched.append(len(matched))
        iou_scores_matched_all.extend(max_iou_score[matched])
        classes_matched_all.extend(classes_matched)
        classes_matched_gt_all.extend(classes_matched_gt)

    if len(classes_matched_all) > 0:
        # Total confusion matrix
        cm_all = confusion_matrix(classes_matched_gt_all, classes_matched_all, labels=class_ids_model)
        covg_all = sum(n_matched) / sum(n_cells_gt)
        #         import pdb; pdb.set_trace()
        if ignore_idx is not None:
            idx = class_ids_model.index(ignore_idx)
            cm_all = np.delete(cm_all, idx, 0)
            cm_all = np.delete(cm_all, idx, 1)

        accu_all = np.sum(np.diag(cm_all)) / np.sum(cm_all)

    else:
        cm_all = None
        covg_all = np.nan
        accu_all = np.nan

    return covg_all, accu_all, cm_all