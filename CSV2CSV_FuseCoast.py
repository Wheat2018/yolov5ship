import pandas as pd
import numpy as np
from collections import defaultdict


def bounding_rect(rects: np.ndarray):
    """
    :param rects: ndarray[[conf, x1, y1, x2, y2], [conf, x1, y1, x2, y2]...]
    :return: ndarray[conf, x1, y1, x2, y2]
    """
    conf = rects[:, 0]
    x = rects[:, [1, 3]]
    y = rects[:, [2, 4]]
    return np.array([conf.max(), x.min(), y.min(), x.max(), y.max()])


source = './final_fantasy8.csv'
target = './final_fantasy8_fuse.csv'
if __name__ == '__main__':
    # read csv as DataFrame
    data = pd.read_csv(source)
    label = data['label']
    img_id = data['img_id']

    # index the coast ranges for each video
    coast_ranges = defaultdict(list)
    video = None
    for i in range(len(label)):
        if label[i] == 'coast':
            coast_ranges[img_id[i]].append(i)

    # fuse the coast rows
    rect_column = ['confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    rect = np.array(data[rect_column])

    coast_rect = {}
    for img, rg in coast_ranges.items():
        data.loc[rg[0], rect_column] = bounding_rect(rect[rg, :])
        data.drop(rg[1:], axis=0, inplace=True)

    # save
    data.to_csv(target, index=None)
