import pandas as pd
from IPython import embed
import os
import random

std = pd.read_csv('A_sample_submission.csv')  # 读取模板
# 获取表头 在表头的基础上继续添加行
nothing = std.iloc[0:0].copy()
# embed()

def get_address_and_id(txt):

    root_dir = '/home/oneco/manyusers/yyc/onecolabBOAT/yolov5/inference/bfuse'
    address = os.path.join(root_dir, txt)
    pieces = txt.split('_')
    empty = pieces[0]
    for piece in pieces[1:-1]:
        empty = empty + '_' + piece
    
    return address, empty



def create_line(alist, gt=False):
    # list has seven sins, each one is str
    if len(alist) == 6:
        alist.insert(2, random.random())
    funcs = ['str', 'str', 'float', 'float', 'float', 'float', 'float']
    empty = {'label':'nothing',
            'img_id':'nothing',
            'confidence':0.1,
            'xmin':1.0,
            'ymin':1.0,
            'xmax':1.0,
            'ymax':1.0}
    # embed()
    for i, (key, value) in enumerate(empty.items()):
        empty[key] = eval(funcs[i])(alist[i])
    # if gt:
    #     cx, cy, w, h = empty['xmin'], empty['ymin'], empty['xmax'], empty['ymax']
    #     empty['xmin'] = 1920*(cx - w/2)
    #     empty['ymin'] = 1080*(cy - h/2)
    #     empty['xmax'] = 1920*(cx + w/2)
    #     empty['ymax'] = 1080*(cy + h/2)
    return empty

def get_gts_from_txt(address, name):
    cats = ['barrier', 'coast']
    with open(address, 'r') as f:
        gts = f.readlines()
    # print(gts)
    gts = [gt.strip().split(' ') for gt in gts]
    gts_replace = []
    for gt in gts:
        # embed()
        gt.insert(1, name)
        gt[0] =  cats[int(gt[0])]
        gts_replace.append(gt)

    return gts_replace

imgs_and_txts = os.listdir('/home/oneco/manyusers/yyc/onecolabBOAT/yolov5/inference/bfuse')
txts = [i for i in imgs_and_txts if 'jpg' not in i]
print(len(txts))
bs = []
# txts.reverse()
for index, txt in enumerate(txts):
    print(index)
    address, image_id = get_address_and_id(txt)
    bs.append(image_id)
    gts = get_gts_from_txt(address, image_id)
    for gt in gts:
        if create_line(gt)['confidence'] < 0.05:
            continue
        nothing = nothing.append(create_line(gt), ignore_index=True)

nothing.to_csv('./final_fantasy34.csv', index=None)
# embed()     




# embed()

# local = get_gts_from_txt('./inference/output/20200829163250129_1.mp4_300_Frame99.txt', 'hhhh')
# embed()
# a = create_line(['5','5', 5, 5, 5, 5, 5])
# embed()
# print(csv_data.shape)  # (189, 9)
# N = 5
# csv_batch_data = csv_data.tail(N)  # 取后5条数据
# print(csv_batch_data.shape)  # (5, 9)
# train_batch_data = csv_batch_data[list(range(3, 6))]  # 取这20条数据的3到5列值(索引从0开始)
# print(train_batch_data)

    # data = pd.DataFrame()
    # a = {"x":1,"y":2}
    # data = data.append(a,ignore_index=True)
    # print(data)