import json
from data.tracking_dataset import Dataset
from data.video import Video
import os
from tqdm import tqdm

class VOTDataset(Dataset):
    def __init__(self, name, dataset_path, load_img=False):
        super(VOTDataset, self).__init__(name,dataset_path)
        # 加载vot json文件
        with open(os.path.join(dataset_path, name+'.json'), 'r') as f:
            meta_data = json.load(f)
        # load videos
        process_bar = tqdm(meta_data.keys(), desc='loding'+name, ncols=100)
        self.videos = {}
        for video in process_bar:
            process_bar.set_postfix_str(video)
            self.videos[video] = VOTVideo


class VOTVideo(Video):
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect, camera_motion,
                 illum_change,motion_change, size_change, occlusion, load_img=False):
        super(VOTVideo, self).__init__(name, root, video_dir, init_rect, img_names, gt_rect, None,
                                       load_img)
        self.tags = {'all': [1] * len(gt_rect)}
        self.tags['camera_motion'] = camera_motion
        self.tags['illum_change'] = illum_change
        self.tags['motion_change'] = motion_change
        self.tags['size_change'] = size_change
        self.tags['occlusion'] = occlusion
