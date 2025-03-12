import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets1.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image

def extract_with_mask(image, mask):

    image_array = image
    mask_array = mask 
    
    extracted_image = Image.fromarray(np.where(mask_array[..., None] > 0, image_array, 0).astype(np.uint8))
    bbox = get_bbox(mask_array)
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
save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('/home/u1120210216/wwc/psdiffusion/configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
model.eval()
ddim_sampler = DDIMSampler(model)



def aug_data_mask(image, mask):
    transform = A.Compose([
        # A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

def process_pairs(ref_image, ref_mask, tar_image, tar_mask, gt_image, gt_mask,gt_effect,ini_effect,back_image,reference_image_albedo,reference_image_normal,gt_shading,raw_norm,raw_shading, max_ratio = 0.8):
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
        masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask
        masked_ref_image_aug = masked_ref_image_compose.copy()
        masked_ref_image_albedo, _ =  masked_ref_image_albedo, ref_mask
        masked_ref_image_normal, _ =  masked_ref_image_normal, ref_mask
        masked_gt_mask1, _ =  masked_gt_mask1, ref_mask
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
        #assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
        # print(tar_box_yyxx)
        # Cropping around the target object 
        tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop
        

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
                ref=masked_ref_image_aug.copy(), 
                #ref_albedo=masked_ref_image_albedo.copy(),
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


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask,gt_image, gt_mask,gt_effect, ini_effect,back_image,reference_image_albedo,reference_image_normal,ground_truth_shading,raw_image_normal,raw_image_shading, guidance_scale = 5.0):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask,gt_image, gt_mask, gt_effect, ini_effect, back_image,reference_image_albedo,reference_image_normal,ground_truth_shading,raw_image_normal,raw_image_shading)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = 12345
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    control_mask = item['ini_effect']
    raw_collage = item['raw_collage']
    gt_mask = item['gt_effect']
    raw_reference = item['raw_reference']
    back_image = item['back_image']
    ref_normal = item['ref_normal']
    raw_norm = item['raw_norm']
    raw_shading= item['raw_shading']
    gt= item['gt_ref']
    gt_shading= item['gt_shading']
    ref_ori = item['ref_ori']

    control_mask = torch.from_numpy(control_mask.copy()).float().cuda() 
    control_mask = control_mask.unsqueeze(0)
    raw_collage = torch.from_numpy(raw_collage.copy()).float().cuda() 
    raw_collage = raw_collage.unsqueeze(0)
    gt_mask = torch.from_numpy(gt_mask.copy()).float().cuda() 
    gt_mask = gt_mask.unsqueeze(0)
    raw_reference = torch.from_numpy(raw_reference.copy()).float()
    raw_reference = raw_reference.unsqueeze(0)
    back_image = torch.from_numpy(back_image.copy()).float() 
    back_image = back_image.unsqueeze(0)

    ref_normal = torch.from_numpy(ref_normal.copy()).float().cuda() 
    ref_normal = ref_normal.unsqueeze(0).transpose(1,3)
    raw_norm = torch.from_numpy(raw_norm.copy()).float()
    raw_norm = raw_norm.unsqueeze(0).transpose(1,3)
    raw_shading = torch.from_numpy(raw_shading.copy()).float() 
    raw_shading = raw_shading.unsqueeze(0).transpose(1,3)
    gt = torch.from_numpy(gt.copy()).float()
    gt = gt.unsqueeze(0).transpose(1,3)
    gt_shading = torch.from_numpy(gt_shading.copy()).float() 
    gt_shading = gt_shading.unsqueeze(0).transpose(1,3)
    ref_ori = torch.from_numpy(ref_ori.copy()).float() 
    ref_ori = ref_ori.unsqueeze(0).transpose(1,3)
    # print(ref_ori.shape,gt_shading.shape)


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [(model.get_learned_conditioning( clip_input, raw_shading,raw_norm, ref_normal, gt, gt_shading,ref_ori))[0]],"c_crossattn1": [(model.get_learned_conditioning( clip_input, raw_shading,raw_norm, ref_normal, gt, gt_shading,ref_ori))[-1]],'c_concat_mask':[control_mask, raw_collage, gt_mask, raw_reference, back_image]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [(model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples))[0]],"c_crossattn1": [(model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples,[torch.zeros((1,3,224,224))] * num_samples))[-1]],"c_concat_mask": None if guess_mode else [control_mask, raw_collage, gt_mask, raw_reference, back_image]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates, mask_predict = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    # print(mask_predict.shape)
    # print(aaaa)
    mask_predict = mask_predict.cpu().squeeze().numpy()
    normalized_array = mask_predict*255
    #normalized_array = (mask_predict - mask_predict.min()) / (mask_predict.max() - mask_predict.min()) * 255
    normalized_array = normalized_array.astype('uint8')
    normalized_array = cv2.cvtColor(normalized_array, cv2.COLOR_GRAY2BGR)
    normalized_array = cv2.resize(normalized_array, (1024, 1024))
    # Save as image
    # image = Image.fromarray(normalized_array)    
    
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    #print(gen_image.shape,normalized_array.shape)
    return gen_image,normalized_array
def set_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__': 

    from omegaconf import OmegaConf
    import os 
    # DConf = OmegaConf.load('./configs/datasets.yaml')
    save_dir = '/home/u1120210216/wwc/psdiffusion/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # test_dir = DConf.Test.VitonHDTest.image_dir
    # image_names = os.listdir(test_dir)
    path = '/home/u1120210216/wwc/datasets/process_data2'
    for i in os.listdir(path):
    #for i in range(1,89):
        # for image_name in image_names:
        random_integer = 12345
        set_seed(random_integer)
        ref_image_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/reference_image.png'.format(i)
        tar_image_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/raw_image.png'.format(i)
        ref_mask_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/reference_mask.png'.format(i)
        tar_mask_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/raw_mask.png'.format(i)
        gt_mask_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/ground_truth_mask.png'.format(i)
        gt_image_path='/home/u1120210216/wwc/datasets/process_data2/{:}/raw_image.png'.format(i)
        back_image_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/masked_raw_image.png'.format(i)
        gt_effect_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/effect_mask.png'.format(i)
        ini_effect_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/raw_image_shadow.png'.format(i)        

        reference_image_albedo_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/reference_image_albedo.png'.format(i)
        reference_image_normal_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/reference_image_normal.png'.format(i) 
        ground_truth_shading_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/raw_image_shading.png'.format(i) 
        raw_image_normal_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/raw_image_normal.png'.format(i) 
        raw_image_shading_path = '/home/u1120210216/wwc/datasets/process_data2/{:}/raw_image_shading.png'.format(i) 

        gen_path = os.path.join(save_dir, '{:}.png'.format(i))
        if os.path.exists(gen_path):
            continue
        raw_image_shading = Image.open(raw_image_shading_path ).convert('RGB').resize((224,224))
        raw_image_shading= np.array(raw_image_shading)
        
        raw_image_normal = Image.open(raw_image_normal_path ).convert('RGB').resize((224,224))
        raw_image_normal= np.array(raw_image_normal)        

        ground_truth_shading = cv2.imread(ground_truth_shading_path)
        ground_truth_shading = cv2.cvtColor(ground_truth_shading, cv2.COLOR_BGR2RGB)

        reference_image_albedo = cv2.imread(reference_image_albedo_path)
        reference_image_albedo = cv2.cvtColor(reference_image_albedo, cv2.COLOR_BGR2RGB)

        reference_image_normal = cv2.imread(reference_image_normal_path)
        reference_image_normal = cv2.cvtColor(reference_image_normal, cv2.COLOR_BGR2RGB)
        
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

        tar_mask = Image.open(tar_mask_path ).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

        gt_mask = Image.open(gt_mask_path ).convert('P')
        gt_mask= np.array(gt_mask)
        gt_mask = np.where(gt_mask > 128, 1, 0).astype(np.uint8)

        gt_image = cv2.imread(gt_image_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        gt_effect = Image.open(gt_effect_path).convert('P').resize((512,512))
        gt_effect= np.array(gt_effect)
        gt_effect = np.where(gt_effect > 128, 1, 0).astype(np.uint8)

        ini_effect = Image.open(ini_effect_path).convert('P').resize((512,512))
        ini_effect= np.array(ini_effect)
        ini_effect = np.where(ini_effect > 128, 1, 0).astype(np.uint8)
        
        back_image = Image.open(back_image_path).convert('RGB').resize((512,512))
        back_image= np.array(back_image).astype(np.uint8)

        gen_image,mask = inference_single_image(ref_image, ref_mask, tar_image.copy(), tar_mask, gt_image,gt_mask,gt_effect, ini_effect,back_image,reference_image_albedo,reference_image_normal,ground_truth_shading,raw_image_normal,raw_image_shading)
        gen_path = os.path.join(save_dir, '{:}.png'.format(i))

        vis_image = cv2.hconcat([ref_image, tar_image, gen_image,gt_image,mask])
        cv2.imwrite(gen_path, vis_image[:,:,::-1])
    #'''

    

