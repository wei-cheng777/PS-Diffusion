import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base1 import BaseDataset

class Mydataset(BaseDataset):
    def __init__(self, image_dir):
        self.image_root = image_dir

        # video_dirs = []
        # with open(self.meta_file) as f:
        #     records = json.load(f)
        #     records = records["videos"]
        #     for video_id in records:
        #         video_dirs.append(video_id)

        # self.records = records
        
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 1
        subfolders = []
        for item in os.listdir(self.image_root):
            item_path = os.path.join(self.image_root, item)
            if os.path.isdir(item_path):
                subfolders.append(item_path)
        self.data = subfolders
    def __len__(self):
        return len(self.data)

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag

    def get_sample(self, idx):
        file_name = self.data[idx]
        # objects_id = np.random.choice( list(self.records[video_id]["objects"].keys()) )
        # frames = self.records[video_id]["objects"][objects_id]["frames"]

        # # Sampling frames
        # min_interval = len(frames)  // 10
        # start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        # end_frame_index = start_frame_index + np.random.randint(min_interval,  len(frames) - start_frame_index )
        # end_frame_index = min(end_frame_index, len(frames) - 1)

        # Get image path
        # ref_image_name = frames[start_frame_index]
        # tar_image_name = frames[end_frame_index]
        ref_image_path = os.path.join(file_name,'reference_image.png')
        back_image_path = os.path.join(file_name,'masked_raw_image.png')
        
        tar_image_path = os.path.join(file_name,'raw_image.png')
        gt_image_path = os.path.join(file_name,'ground_truth.png')
        ref_mask_path = os.path.join(file_name,'reference_mask.png')
        tar_mask_path = os.path.join(file_name,'raw_mask.png')
        gt_mask_path = os.path.join(file_name,'ground_truth_mask.png')
        
        gt_effect_path = os.path.join(file_name,'effect_mask.png')
        ini_effect_path = os.path.join(file_name,'raw_image_shadow.png')
        reference_image_albedo_path = os.path.join(file_name,'reference_image_albedo.png')
        reference_image_normal_path = os.path.join(file_name,'reference_image_normal.png')
        ground_truth_shading_path = os.path.join(file_name,'ground_truth_shading.png')
        raw_image_normal_path = os.path.join(file_name,'raw_image_normal.png')
        raw_image_shading_path = os.path.join(file_name,'raw_image_shading.png')
        
        # print(file_name)
        # raw_image_shading = cv2.imread(raw_image_shading_path)
        # raw_image_shading = cv2.cvtColor(raw_image_shading, cv2.COLOR_BGR2RGB)
        # raw_image_shading = cv2.resize(raw_image_shading, (512, 512))
        
        raw_image_shading = Image.open(raw_image_shading_path ).convert('RGB').resize((224,224))
        raw_image_shading= np.array(raw_image_shading)
        
        # raw_image_normal = cv2.imread(raw_image_normal_path)
        # raw_image_normal = cv2.cvtColor(raw_image_normal, cv2.COLOR_BGR2RGB)
        # raw_image_normal = cv2.resize(raw_image_normal, (512, 512))
        raw_image_normal = Image.open(raw_image_normal_path ).convert('RGB').resize((224,224))
        raw_image_normal= np.array(raw_image_normal)
        
        ground_truth_shading = cv2.imread(ground_truth_shading_path)
        ground_truth_shading = cv2.cvtColor(ground_truth_shading, cv2.COLOR_BGR2RGB)

        reference_image_albedo = cv2.imread(reference_image_albedo_path)
        reference_image_albedo = cv2.cvtColor(reference_image_albedo, cv2.COLOR_BGR2RGB)

        reference_image_normal = cv2.imread(reference_image_normal_path)
        reference_image_normal = cv2.cvtColor(reference_image_normal, cv2.COLOR_BGR2RGB)
                
        gt_image = cv2.imread(gt_image_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = Image.open(ref_mask_path ).convert('P')
        ref_mask= np.array(ref_mask)
        ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

        gt_mask = Image.open(gt_mask_path ).convert('P')
        gt_mask= np.array(gt_mask)
        gt_mask = np.where(gt_mask > 128, 1, 0).astype(np.uint8)        
        # ref_mask = ref_mask == int(objects_id)

        tar_mask = Image.open(tar_mask_path ).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)
        # tar_mask = tar_mask == int(objects_id)

        gt_effect = Image.open(gt_effect_path).convert('P').resize((512,512))
        gt_effect= np.array(gt_effect)
        gt_effect = np.where(gt_effect > 128, 1, 0).astype(np.uint8)

        ini_effect = Image.open(ini_effect_path).convert('P').resize((512,512))
        ini_effect= np.array(ini_effect)
        ini_effect = np.where(ini_effect > 128, 1, 0).astype(np.uint8)
        
        back_image = Image.open(back_image_path).convert('RGB').resize((512,512))
        back_image= np.array(back_image).astype(np.uint8)
        # print(1111)
        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, gt_image,gt_mask, gt_effect, ini_effect,back_image,reference_image_albedo,reference_image_normal,ground_truth_shading,raw_image_normal,raw_image_shading)
        # print(file_name)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage


