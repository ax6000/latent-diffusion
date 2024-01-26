import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    print(os.path.isfile(dir))
    if os.path.isfile(dir):
        arr = np.genfromtxt(dir, dtype=str, encoding='utf-8')
        if arr.ndim:
            images = [i for i in arr]
        else:
            images = np.array([arr])
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

class PPG2ABPDataset_v3_base(Dataset):
    def __init__(self,data_flist,data_root = r"F:\minowa\BloodPressureEstimation\data\processed\BP_npy\1121\p00",
                 data_len=-1, size=224, loader=None):
        self.data_root = data_root
        self.data_flist = data_flist
        self.flist = make_dataset(self.data_flist)
        # if data_len > 0:
        #     self.flist = flist[:int(data_len)]
        # else:
        #     self.flist = flist
        self.tfs = transforms.ToTensor()
        self.size = size
        self.data=self.load_npys()
        if data_len > 0:
            data_index = np.arange(0,len(self.data),max(len(self.data)//int(data_len),1)).astype(int)[:int(data_len)]
            self.data = self.data[data_index]
        else:
            self.data = self.data[:len(self.data)-len(self.data)%64]
        print("data prepared:" ,self.data.shape)
    def load_npys(self):
        data = []
        for f in self.flist:
            arr = np.load(self.data_root+"\\"+str(f))
            if len(arr) != 0:
                data.append(arr)
        data = np.concatenate(data)
        return data
    
    def __getitem__(self, index):
        ret = {}
        ret['gt_image'] = self.data[index,:,0].reshape(1,-1).astype(np.float32)
        ret['cond_image'] = self.data[index,:,1].reshape(1,-1).astype(np.float32)
        ret['path'] = str(index)
        return ret

    def __len__(self):
        return self.data.shape[0]
    
class PPG2ABPDataset_v3_Train(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=-1, size=224, loader=None):
        super().__init__(data_flist = r"F:\minowa\BloodPressureEstimation\data\processed\list\train_BP.txt")

class PPG2ABPDataset_v3_Val(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=-1, size=224, loader=None):
        super().__init__(data_flist = r"F:\minowa\BloodPressureEstimation\data\processed\list\val_BP.txt")

class PPG2ABPDataset_v3_Test(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=-1, size=224, loader=None):
        super().__init__(data_flist = r"F:\minowa\BloodPressureEstimation\data\processed\list\test_BP.txt")         
    
# class PPG2ABPDataset_v3_Predict(PPG2ABPDataset_v3_base):
#     def __init__(self, data_len=-1,size=224, loader=None):
#         super().__init__(data_flist = r"F:\minowa\BloodPressureEstimation\data\processed\list\predict_BP.txt")
