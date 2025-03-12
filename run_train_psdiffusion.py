import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets1.mydataset1 import Mydataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import time
save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()
def load_state_dict_with_skipping(model, checkpoint, device='cpu'):
    model_state_dict = model.state_dict()

    new_state_dict = {}
    
    for key, param in checkpoint.items():
        if key in model_state_dict:
            if param.shape == model_state_dict[key].shape:
                new_state_dict[key] = param
            else:
                print(f"Skipping parameter '{key}' due to size mismatch: "
                      f"checkpoint shape {param.shape} vs model shape {model_state_dict[key].shape}")
        else:
            print(f"Skipping parameter '{key}' as it is not found in the model state dictionary")
    
    # 加载新的状态字典
    model.load_state_dict(new_state_dict, strict=False)
# Configs
resume_path = '/home/u1120210216/wwc/AnyDoor-main/weights/epoch=1-step=8687.ckpt'
batch_size = 8
logger_freq = 500
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches=1

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('/home/u1120210216/wwc/psdiffusion/configs/ps-diffusion.yaml').cpu()
# load_state_dict_with_skipping(model, load_state_dict(resume_path, location='cpu'), device='cpu')
model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets
DConf = OmegaConf.load('/home/u1120210216/wwc/psdiffusion/configs/datasets.yaml')
dataset1 = Mydataset(**DConf.Train.Mydataset)  

image_data = [dataset1]
# video_data = [dataset1, dataset3, dataset4, dataset7, dataset9, dataset10 ]
# tryon_data = [dataset8, dataset11]
# threed_data = [dataset5]

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints_ours",  # 指定保存路径
    filename="model-{epoch:02d}",  # 文件名格式
    save_top_k=-1,  # 保存所有检查点
    every_n_epochs=2,  # 每 10 个 epoch 保存一次
)

# The ratio of each dataset is adjusted by setting the __len__ 
dataset = ConcatDataset( image_data)
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

trainer = pl.Trainer(gpus=n_gpus, strategy="ddp_sharded", precision=16, accelerator="gpu", callbacks=[logger,checkpoint_callback], progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches,max_epochs=40)

# Train!
trainer.fit(model, dataloader)
