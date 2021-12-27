from model import *
from trainer import *
import torch
from dataloader import *
from args import *

#################################################################################################
#                                                                                               #
# TODO : save 한 model parameter 를 load 하여 test set 에서 성능을 확인해봅시다.                #
#                                                                                               #
#################################################################################################
device="cuda"
path = "/content/drive/MyDrive/Colab Notebooks/pth_path/log_epoch_7.pth"

Model = seg_model
Model.load_state_dict(torch.load(path))
Model.eval()

trainer_load = Semantic_Seg_Trainer(model=Model, opt="adam", lr=Args["lr"], has_scheduler=False, device=device).to(device)
trainer_load.test(test_loader)