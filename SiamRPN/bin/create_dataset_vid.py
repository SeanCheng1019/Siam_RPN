'''
Crop and pad VID_dataset
'''
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool
import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from lib.util import get_instance_img
from net.config import Config
import functools
import pickle


def process_single_document_data(sequence_dir, output_dir):
    image_names = glob(os.path.join(sequence_dir, '*.JPEG'))
    image_names = sorted(image_names, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    sequence_name = sequence_dir.split('/')[-1]
    save_folder = os.path.join(output_dir, sequence_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    trajs = {}

    for image_name in image_names:

        img = cv2.imread(image_name)
        img_weight, img_height, _ = img.shape
        # R，G，B 通道的平均像素值
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        # 对应GT的路径
        anno_name = image_name.replace('Data', 'Annotations')
        anno_name = anno_name.replace('JPEG', 'xml')
        tree = ET.parse(anno_name)
        root = tree.getroot()
        bboxes = []
        # filename of each frame in one sequence
        filename = root.find('filename').text
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            trkid = int(obj.find('trackid').text)
            # eg: {0:[00000,00001,00002,00003],1:[00001,00004,00005]}
            if trkid in trajs:
                trajs[trkid].append(filename)
            else:
                trajs[trkid] = [filename]
            bbox = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]
            )
            # 此处得到的w，h是resize后，目标的尺寸，已经不是原始图片上目标的尺寸
            instance_img, w, h, _ = get_instance_img(img, bbox, Config.exemplar_size, Config.instance_crop_size,
                                                     Config.context_margin_amount, img_mean)
            instance_img_save_path = os.path.join(save_folder, filename +
                                                  ".{:02d}.patch_w_{:.2f}_patch_h{:.2f}_img_w_{:.0f}_img_h_{:.0f}.jpg".format(
                                                      trkid, w, h, img_weight, img_height
                                                  ))
            cv2.imwrite(instance_img_save_path, instance_img)
    return sequence_name, trajs


def process_all_sequence(vid_dir, output_dir, num_threads=mp.cpu_count()):
    all_sequence_documents_dirs = glob(os.path.join(vid_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                                  glob(os.path.join(vid_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                                  glob(os.path.join(vid_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                                  glob(os.path.join(vid_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                                  glob(os.path.join(vid_dir, 'val/*'))
    len = 0
    total = all_sequence_documents_dirs.__len__()
    meta_data = []
    with Pool(processes=num_threads) as pool:
        print("staring process, please waiting...")
        for sequence_dir in all_sequence_documents_dirs:
            meta_data.append(process_single_document_data(sequence_dir, output_dir))
            len += 1
            print("finish process ", sequence_dir, "\n")
            print("left ", total - len, " waiting to precess \n")

    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))


if __name__ == '__main__':
    vid_dir = '/media/csy/e1d382bb-e140-4c82-bbc6-8ee2f62a4240/dataset/ILSVRC2015_VID/ILSVRC2015/Data/VID'
    output_dir = '/media/csy/e1d382bb-e140-4c82-bbc6-8ee2f62a4240/dataset/ILSVRC2015_VID_curation3'
    process_all_sequence(vid_dir, output_dir, num_threads=mp.cpu_count())
