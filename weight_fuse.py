from ensemble_boxes import *
import os
from IPython import embed


def get_one(root, name):

    abs_file = os.path.join(root, name)
    labels = []
    confidences = []
    boxes = []
    if os.path.exists(abs_file):
        with open(abs_file) as f:
            predictions = f.readlines()
            predictions = [line.strip().split() for line in predictions]
            for prediction in predictions:
                labels.append(int(prediction[0]))
                tmp = [float(prediction[2])/1920, float(prediction[3])/1080, float(prediction[4])/1920, float(prediction[5])/1080]
                boxes.append(tmp)
                confidences.append(float(prediction[1]))
            return labels, confidences, boxes
    else:
        return [], [], []



def write_one(labels, confidences, boxes, root, name):

    with open(os.path.join(root, name), 'a') as f:
        for label, confidence, box  in zip(labels, confidences, boxes):
            # embed()
            f.write(('%g ' * 6 + '\n') % (label, confidence, float(box[0])*1920, float(box[1])*1080, float(box[2])*1920, float(box[3])*1080))  # label format


where_is_set1 = '/home/oneco/manyusers/yyc/onecolabBOAT/tobefused/set1'
where_is_set2 = '/home/oneco/manyusers/yyc/onecolabBOAT/tobefused/set2'
final_set = '/home/oneco/manyusers/yyc/onecolabBOAT/tobefused/fused'


set1 = os.listdir(where_is_set1)
set2 = os.listdir(where_is_set2)
set_all = set1 + set2
set_all = list(set(set_all))

for one in set_all:
    set1_l, set1_c, set1_b = get_one(where_is_set1, one)
    set2_l, set2_c, set2_b = get_one(where_is_set2, one)

    iou_thr = 0.2
    skip_box_thr = 0.0001
    sigma = 0.1
    boxes_list = []
    scores_list = []
    labels_list =[]

    weights = [2, 1]
    boxes_list.append(set1_b)
    boxes_list.append(set2_b)
    scores_list.append(set1_c)
    scores_list.append(set2_c)
    labels_list.append(set1_l)
    labels_list.append(set2_l)

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # embed()
    write_one(labels, scores, boxes, final_set, one)


