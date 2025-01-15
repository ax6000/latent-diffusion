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
    def __init__(self,data_flist,data_root = None,
                 data_len=1000, size=224, loader=None):
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
        ret['gt_image'] = self.data[index,:,0][np.newaxis, :].astype(np.float32)
        ret['cond_image'] = self.data[index,:,1][np.newaxis, :].astype(np.float32)
        ret['path'] = str(index)
        return ret

    def __len__(self):
        return self.data.shape[0]

class PPG2ABPDataset_v3_Train(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=-1, size=224, loader=None, data_root=r"..\..\data\processed\BP_npy\1127_256_balanced\p00"):
        super().__init__(data_len=data_len,data_flist = r"..\..\data\processed\list\train_BP2.txt",data_root=data_root)

class PPG2ABPDataset_v3_Val(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=1000, size=224, loader=None, data_root=r"..\..\data\processed\BP_npy\1127_256_balanced\p00"):
        super().__init__(data_len=data_len,data_flist = r"..\..\data\processed\list\val_BP2.txt",data_root=data_root)

class PPG2ABPDataset_v3_Test(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=5000, size=224, loader=None, data_root=r"..\..\data\processed\BP_npy\1127_256_balanced\p00"):
        super().__init__(data_len=data_len,data_flist = r"..\..\data\processed\list\test_BP2.txt",data_root=data_root)         
    
class PPG2ABPDataset_v3_Predict(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=-1,size=224, loader=None):
        super().__init__(data_flist = r"..\..\data\processed\list\predict_BP2.txt")   

class PPG2ABPDataset_v5_base(Dataset):
    def __init__(self,data_files,data_root = r"..\..\data\processed\BP_npy\0625_256_downsampled\p00",
                 data_len=1000, size=224, loader=None):
        self.data_root = data_root
        self.data_files = data_files
        self.data_len = data_len
        self.abp = None  
        self.ppg = None
        # if data_len > 0:
        #     self.flist = flist[:int(data_len)]
        # else:
        #     self.flist = flist
        self.tfs = transforms.ToTensor()
        self.size = size
        self.load_npys()
  
        print("data prepared:" ,self.ppg.shape)
    def load_npys(self):
        self.abp = np.load(self.data_root+"\\"+self.data_files[0])
        self.ppg = np.load(self.data_root+"\\"+self.data_files[1])
        if self.data_len > 0:
            data_index = np.arange(0,len(self.abp),max(len(self.abp)//int(self.data_len),1)).astype(int)[:int(self.data_len)]
            self.abp = self.abp[data_index]
            self.ppg = self.ppg[data_index]
        else:
            self.abp = self.abp[:len(self.abp)-len(self.abp)%64]
            self.ppg = self.ppg[:len(self.ppg)-len(self.ppg)%64]
    def __getitem__(self, index):
        ret = {}
        ret['gt_image'] = self.abp[index].reshape(1,-1).astype(np.float32)
        ret['cond_image'] = self.ppg[index].reshape(3,-1).astype(np.float32)
        ret['path'] = str(index)
        return ret

    def __len__(self):
        return self.abp.shape[0]
    
class PPG2ABPDataset_v5_Train(PPG2ABPDataset_v5_base):
    def __init__(self, data_len=-1, size=224, loader=None):
        super().__init__(data_len=data_len,data_files=["train_abp.npy","train_ppg.npy"])

class PPG2ABPDataset_v5_Val(PPG2ABPDataset_v5_base):
    def __init__(self, data_len=1000, size=224, loader=None):
        super().__init__(data_len=data_len,data_files=["validate_abp.npy","validate_ppg.npy"])

class PPG2ABPDataset_v5_Test(PPG2ABPDataset_v5_base):
    def __init__(self, data_len=5000, size=224, loader=None):
        super().__init__(data_len=data_len,data_files=["test_abp.npy","test_ppg.npy"])         
    
class PPG2ABPDataset_v5_Predict(PPG2ABPDataset_v5_base):
    def __init__(self, data_len=-1,size=224, loader=None):
        super().__init__(data_files=["test_abp.npy","test_ppg.npy"])

class PPG2ABPDataset_v3CV_base(Dataset):
    def __init__(self,data_flist,data_root = r"..\..\data\processed\BP_npy\0625_256_cv\p00",
                 data_len=1000, size=224, loader=None):
        self.data_root = data_root
        self.data_flist = data_flist
        print("flist path is",data_flist)
        self.flist = make_dataset(self.data_flist)
        self.fold=int(data_flist[-5])
        self.data=self.load_npys()
        self.normalize()
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
    
    def normalize(self):
        scale = np.load(os.path.join(self.data_root,f"scale_fold{self.fold}.npy"))
        self.data[:,:,0] -= scale[0,0]
        self.data[:,:,0] /= scale[0,1]
        self.data[:,:,1] -= scale[1,0]
        self.data[:,:,1] /= scale[1,1]
        
    def __getitem__(self, index):
        ret = {}
        ret['gt_image'] = self.data[index,:,0].reshape(1,-1).astype(np.float32)
        ret['cond_image'] = self.data[index,:,1].reshape(1,-1).astype(np.float32)
        ret['path'] = str(index)
        return ret

    def __len__(self):
        return self.data.shape[0]

class PPG2ABPDataset_v3CV_Train(PPG2ABPDataset_v3CV_base):
    def __init__(self,fold=None, data_len=-1, size=224, loader=None):
        super().__init__(data_len=data_len,data_flist = r"..\..\data\processed\list\train_fold"+str(fold)+".txt")

class PPG2ABPDataset_v3CV_Val(PPG2ABPDataset_v3CV_base):
    def __init__(self,fold=None, data_len=1000, size=224, loader=None):
        super().__init__(data_len=data_len,data_flist = r"..\..\data\processed\list\val_fold"+str(fold)+".txt")

class PPG2ABPDataset_v3CV_Test(PPG2ABPDataset_v3CV_base):
    def __init__(self,fold=None, data_len=5000, size=224, loader=None):
        super().__init__(data_len=data_len,data_flist = r"..\..\data\processed\list\test_fold"+str(fold)+".txt")         
    
# class PPG2ABPDataset_v3CV_Predict(PPG2ABPDataset_v3CV_base):
#     def __init__(self,fold, data_len=-1,size=224, loader=None):
#         super().__init__(data_flist = r"..\..\data\processed\list\predict_BP2.txt")   
