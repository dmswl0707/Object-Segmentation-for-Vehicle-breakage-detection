import torchvision.models as models
from torch import nn

seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
seg_model.classifier[4]=nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
seg_model