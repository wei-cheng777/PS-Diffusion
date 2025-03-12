import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A

def extract_with_mask(image, mask):
    """
    从图片中根据给定的mask提取内容，并返回裁剪后的图像和mask的包围盒
    """
    image_array = image
    mask_array = mask # 转换为灰度图
    
    extracted_image = Image.fromarray(np.where(mask_array[..., None] > 0, image_array, 0).astype(np.uint8))
    # 计算 mask 的包围盒
    bbox = get_bbox(mask_array)
    
    # 裁剪图像和mask到包围盒区域
    image_cropped = extracted_image.crop((bbox[1], bbox[0], bbox[3], bbox[2]))
    mask_cropped = (Image.fromarray(mask_array).crop((bbox[1], bbox[0], bbox[3], bbox[2]))).convert('L')
    
    return image_cropped, mask_cropped, bbox

def get_bbox(mask_array):
    """
    根据mask获得包围盒
    """
    coords = np.column_stack(np.where(mask_array > 0))
    bbox = [coords[:,0].min(), coords[:,1].min(), coords[:,0].max(), coords[:,1].max()]
    return bbox
class BaseDataset(Dataset):
    def __init__(self):
        image_mask_dict = {}
        self.data = []

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass

    
    def aug_data_back(self, image):
        transform = A.Compose([
            A.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            A.ChannelShuffle()
            ])
        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image
    
    def aug_data_mask_strong(self, image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask
    
    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            #A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask

    def aug_data_paste(self, image):
        transform = A.Compose([
            #A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag


    def __getitem__(self, idx):
        while(True):
            try:
                idx = np.random.randint(0, len(self.data)-1)
                item = self.get_sample(idx)
                return item
            except:
                idx = np.random.randint(0, len(self.data)-1)
                
    def get_sample(self, idx):
        # Implemented for each specific dataset
        pass

    def sample_timestep(self, max_step =1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
            step_end = max_step
        elif self.dynamic == 0:
            # static images
            step_start = 0 
            step_end = max_step // 2
        else:
            # fine multi-view images/videos/3Ds
            step_start = 0
            step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def check_mask_area(self, mask):
        H,W = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (H * W)
        if ratio > 0.8 * 0.8  or ratio < 0.1 * 0.1:
            return False
        else:
            return True 
    

    def process_pairs(self, ref_image, ref_mask, tar_image, tar_mask, gt_image, gt_mask, gt_effect,ini_effect,back_image,reference_image_albedo,reference_image_normal,gt_shading, raw_norm, raw_shading, max_ratio = 0.8):
        assert mask_score(ref_mask) > 0.90
        assert self.check_mask_area(ref_mask) == True
        assert self.check_mask_area(tar_mask)  == True
        ref_mask_ori = ref_mask*255
        # gt_effect = gt_effect.resize((512, 512))
        # ini_effect = ini_effect.resize((512, 512))
        # ========= Reference ===========
        '''
        # similate the case that the mask for reference object is coarse. Seems useless :(

        if np.random.uniform(0, 1) < 0.7: 
            ref_mask_clean = ref_mask.copy()
            ref_mask_clean = np.stack([ref_mask_clean,ref_mask_clean,ref_mask_clean],-1)
            ref_mask = perturb_mask(ref_mask, 0.6, 0.9)
            
            # select a fake bg to avoid the background leakage
            fake_target = tar_image.copy()
            h,w = ref_image.shape[0], ref_image.shape[1]
            fake_targe = cv2.resize(fake_target, (w,h))
            fake_back = np.fliplr(np.flipud(fake_target))
            fake_back = self.aug_data_back(fake_back)
            ref_image = ref_mask_clean * ref_image + (1-ref_mask_clean) * fake_back
        '''

        # Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        #print(self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        
        gt_mask1_box_yyxx = get_bbox_from_mask(gt_mask)
        gt_mask1_3 = np.stack([gt_mask,gt_mask,gt_mask],-1)
        masked_gt_mask1 = gt_image * gt_mask1_3 + np.ones_like(gt_image) * 255 * (1-gt_mask1_3)
        masked_gt_mask1_shading = gt_shading * gt_mask1_3 + np.ones_like(gt_shading) * 255 * (1-gt_mask1_3)
        gt_y1,gt_y2,gt_x1,gt_x2 = gt_mask1_box_yyxx
        masked_gt_mask1 = masked_gt_mask1[gt_y1:gt_y2,gt_x1:gt_x2,:]
        masked_gt_mask1_shading = masked_gt_mask1_shading[gt_y1:gt_y2,gt_x1:gt_x2,:]
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)
        masked_ref_image_albedo = reference_image_albedo * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)
        masked_ref_image_normal = reference_image_normal * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
        masked_ref_image_albedo = masked_ref_image_albedo[y1:y2,x1:x2,:]
        masked_ref_image_normal = masked_ref_image_normal[y1:y2,x1:x2,:]
        ref_mask = ref_mask[y1:y2,x1:x2]

        masked_gt_mask1 = Image.fromarray(masked_gt_mask1).resize((masked_ref_image.shape[1],masked_ref_image.shape[0]))
        masked_gt_mask1 = np.array(masked_gt_mask1)

        ratio = np.random.randint(11, 15) / 10 
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

        masked_ref_image_albedo, _ = expand_image_mask(masked_ref_image_albedo, ref_mask, ratio=ratio)
        masked_ref_image_normal, _ = expand_image_mask(masked_ref_image_normal, ref_mask, ratio=ratio)
        masked_gt_mask1, _ = expand_image_mask(masked_gt_mask1, ref_mask, ratio=ratio)
        masked_gt_mask1_shading, _ = expand_image_mask(masked_gt_mask1_shading, ref_mask, ratio=ratio)
        
        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)
        #masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (512,512) ).astype(np.uint8)

        masked_ref_image_albedo = pad_to_square(masked_ref_image_albedo, pad_value = 255, random = False)
        masked_ref_image_albedo = cv2.resize(masked_ref_image_albedo.astype(np.uint8), (224,224) ).astype(np.uint8)

        masked_ref_image_normal = pad_to_square(masked_ref_image_normal, pad_value = 255, random = False)
        masked_ref_image_normal = cv2.resize(masked_ref_image_normal.astype(np.uint8), (224,224) ).astype(np.uint8)

        masked_gt_mask1 = pad_to_square(masked_gt_mask1, pad_value = 255, random = False)
        masked_gt_mask1 = cv2.resize(masked_gt_mask1.astype(np.uint8), (224,224) ).astype(np.uint8)

        masked_gt_mask1_shading = pad_to_square(masked_gt_mask1_shading, pad_value = 255, random = False)
        masked_gt_mask1_shading = cv2.resize(masked_gt_mask1_shading.astype(np.uint8), (224,224) ).astype(np.uint8)

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
        #ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (512,512) ).astype(np.uint8)
        ref_mask = ref_mask_3[:,:,0]

        # Augmenting reference image
        #masked_ref_image_aug = self.aug_data(masked_ref_image) 
        
        # Getting for high-freqency map
        masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, ref_mask) 
        # masked_ref_image_aug = masked_ref_image_compose.copy()
        # masked_ref_image_albedo, _ =  self.aug_data_mask(masked_ref_image_albedo, ref_mask) 
        # masked_ref_image_normal, _ =  self.aug_data_mask(masked_ref_image_normal, ref_mask) 
        # masked_gt_mask1, _ =  self.aug_data_mask(masked_gt_mask1, ref_mask) 
        # masked_gt_mask1_shading, _ =  self.aug_data_mask(masked_gt_mask1_shading, ref_mask) 
        
        
        # #masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask 
        masked_ref_image_aug = masked_ref_image_compose.copy()
        masked_ref_image_albedo, _ =  masked_ref_image_albedo, ref_mask
        masked_ref_image_normal, _ =  masked_ref_image_normal, ref_mask 
        masked_gt_mask1, _ = masked_gt_mask1, ref_mask
        masked_gt_mask1_shading, _ =  masked_gt_mask1_shading, ref_mask



        ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
        

        # ========= Training Target ===========
        combined_mask = tar_mask*255 + gt_mask*255    
        
        combined_mask = np.bitwise_or(tar_mask*255, gt_mask*255)
        combined_mask = np.where((tar_mask*255 > 128) | (gt_mask*255 > 128), 255, 0).astype(np.uint8)
        tar_mask = combined_mask
        
        # image = Image.fromarray(tar_mask)
        # save_path = 'image.png'
        # image.save(save_path)      
          
        tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)
        
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
        #print(self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True)
        assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
        # print(tar_box_yyxx)
        # Cropping around the target object 
        tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop
        
        # cropped_target_image = tar_image[y1:y2,x1:x2,:]
        # cropped_tar_mask = tar_mask[y1:y2,x1:x2]
        # cropped_gt_mask = gt_mask[y1:y2,x1:x2]*255
        # cropped_gt_image = gt_image[y1:y2,x1:x2,:]
        cropped_target_image = tar_image
        cropped_tar_mask = tar_mask
        cropped_gt_mask = gt_mask*255
        cropped_gt_image = gt_image
        # print(y1,y2,x1,x2)
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1,y2,x1,x2 = tar_box_yyxx
        
        ref_image_collage_224 = ref_image_collage
        #cropped_image1, cropped_mask1, bbox1 = extract_with_mask(ref_image_collage_224, ref_mask_compose)
        #cropped_image1, cropped_mask1, bbox1 = extract_with_mask(reference_image_albedo, ref_mask_ori)
        cropped_image1, cropped_mask1, bbox1 = extract_with_mask(ref_image, ref_mask_ori)
        
        #####如果使用sobel
        # cropped_image1 = sobel(np.array(cropped_image1), np.array(cropped_mask1)/255)
        # cropped_image1 = Image.fromarray(cropped_image1)
        

        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

        collage = cropped_target_image.copy() 
        # print(x2,x1, y2,y1)
        # image = Image.fromarray(ref_image_collage)
        # save_path = 'image.png'
        # image.save(save_path)
        
        ref_image_collage = np.zeros_like(ref_image_collage)
        #collage[y1:y2,x1:x2,:] = ref_image_collage
        collage[cropped_tar_mask == 1] = 0
        


        bbox2 = get_bbox(cropped_gt_mask)
        resized_cropped_image1 = cropped_image1.resize((bbox2[3] - bbox2[1], bbox2[2] - bbox2[0]), Image.LANCZOS)
        final_image = Image.fromarray(collage.copy())
        final_image.paste(resized_cropped_image1.convert('RGBA'), (bbox2[1], bbox2[0]), resized_cropped_image1.convert('RGBA'))
        final_image = np.array(final_image)
        final_image[cropped_tar_mask == 0] = collage[cropped_tar_mask == 0]
        collage = final_image
        
        # image = Image.fromarray(collage)
        # save_path = 'image1.png'
        # image.save(save_path)

        collage_mask = np.stack([cropped_tar_mask] * 3, axis=-1)
        # collage_mask = cropped_target_image.copy() * 0.0
        # collage_mask[y1:y2,x1:x2,:] = 1.0

        # if np.random.uniform(0, 1) < 0.7: 
        #     cropped_tar_mask = perturb_mask(cropped_tar_mask)
        #     collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        cropped_gt_image = pad_to_square(cropped_gt_image, pad_value = 0, random = False).astype(np.uint8)
        
        # image = Image.fromarray(cropped_gt_image)
        # save_path = 'image2.png'
        # image.save(save_path)
        
        collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        
        # image = Image.fromarray(collage_mask*255)
        # save_path = 'image3.png'
        # image.save(save_path)
        
        H2, W2 = collage.shape[0], collage.shape[1]

        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
        cropped_gt_image = cv2.resize(cropped_gt_image.astype(np.uint8), (512,512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
        collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
        collage_mask[collage_mask == 2] = -1
        
        raw_reference = masked_ref_image_aug.copy()
        # Prepairing dataloader items
        masked_ref_image_aug = masked_ref_image_aug  / 255 
        # masked_ref_image_aug = masked_ref_image_aug
        masked_ref_image_albedo = masked_ref_image_albedo / 255
        masked_ref_image_normal = masked_ref_image_normal / 255
        masked_gt_mask1 = masked_gt_mask1/255
        masked_gt_mask1_shading = masked_gt_mask1_shading/255
        raw_norm = raw_norm / 255
        raw_shading = raw_shading / 255
        
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        cropped_gt_image = cropped_gt_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0 
        raw_collage = collage.copy()
        collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
        tar_box_yyxx_crop = (0,1024,0,1024)
        item = dict(
                #ref=masked_ref_image_albedo.copy(), 
                ref=masked_ref_image_aug.copy(), 
                ref_ori=masked_ref_image_aug.copy(),
                ref_normal=masked_ref_image_normal.copy(),
                jpg=cropped_gt_image.copy(), 
                gt_ref = masked_gt_mask1.copy(),
                gt_shading = masked_gt_mask1_shading.copy(),
                hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
                gt_effect=gt_effect.copy(),
                ini_effect=ini_effect.copy(),
                raw_collage = raw_collage.copy(),
                raw_reference =raw_reference.copy(),
                back_image = back_image.copy(),
                raw_norm = raw_norm.copy(),
                raw_shading = raw_shading.copy(),
                ) 
        return item





