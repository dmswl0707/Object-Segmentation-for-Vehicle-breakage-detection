from transforms import *
from dataset import *
from torch.utils.data import DataLoader

dent_train = SOCAR_Dataset(os.path.join(root,'train'), get_transform(train=True))
dent_valid = SOCAR_Dataset(os.path.join(root,'valid'), get_transform(train=False))
dent_test = SOCAR_Dataset(os.path.join(root,'test'), get_transform(train=False))

#메모리 부족으로 배치사이즈 수정은 불가해보임 ㅜㅜ...
train_loader = DataLoader(dent_train, batch_size=12, shuffle=True, drop_last=True)
valid_loader = DataLoader(dent_valid, batch_size=12, shuffle=False, drop_last=True)
test_loader = DataLoader(dent_test, batch_size=2, shuffle=False, drop_last=True)