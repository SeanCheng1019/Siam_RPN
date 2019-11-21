import json
from data.tracking_dataset import Dataset
from data.video import Video
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


class VOTDataset(Dataset):
    def __init__(self, name, dataset_path, load_img=False):
        super(VOTDataset, self).__init__(name, dataset_path)
        # 加载vot json文件
        with open(os.path.join(dataset_path, name + '.json'), 'r') as f:
            meta_data = json.load(f)
        # load videos
        process_bar = tqdm(meta_data.keys(), desc='loding' + name, ncols=100)
        self.videos = {}
        for video in process_bar:
            process_bar.set_postfix_str(video)
            self.videos[video] = VOTVideo(video, dataset_path,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['camera_motion'],
                                          meta_data[video]['illum_change'],
                                          meta_data[video]['motion_change'],
                                          meta_data[video]['size_change'],
                                          meta_data[video]['occlusion'],
                                          load_img=load_img)
        self.tags = ['all', 'camera_motion', 'illum_change', 'motion_change', 'size_change',
                     'occlusion', 'empty']



class VOTVideo(Video):
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect, camera_motion,
                 illum_change, motion_change, size_change, occlusion, load_img=False):
        super(VOTVideo, self).__init__(name, root, video_dir, init_rect, img_names, gt_rect, None,
                                       load_img)
        self.tags = {'all': [1] * len(gt_rect)}
        self.tags['camera_motion'] = camera_motion
        self.tags['illum_change'] = illum_change
        self.tags['motion_change'] = motion_change
        self.tags['size_change'] = size_change
        self.tags['occlusion'] = occlusion
        all_tag = [v for k, v in self.tags.items() if len(v) > 0]
        self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
        self.tag_names = list(self.tags.keys())
        if not load_img:
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]

    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]

    def load_tracker(self, path, tracker_name=None, store=True):
        pass
