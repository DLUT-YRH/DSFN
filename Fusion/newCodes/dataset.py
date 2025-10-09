from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.train_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'warp1' or data_name == 'warp2' or data_name == 'mask1' or data_name == 'mask2' or data_name == 'warp_depth1' or data_name == 'warp_depth2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):

        # load image1
        warp1 = cv2.imread(self.datas['warp1']['image'][index])
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        # load image2
        warp2 = cv2.imread(self.datas['warp2']['image'][index])
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        # load mask1
        mask1 = cv2.imread(self.datas['mask1']['image'][index])
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = np.expand_dims(mask1[:,:,0], 2) / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        # load mask2
        mask2 = cv2.imread(self.datas['mask2']['image'][index])
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = np.expand_dims(mask2[:,:,0], 2) / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

        # add depth input
        # load depth image1
        depthinput1 = cv2.imread(self.datas['warp_depth1']['image'][index],cv2.IMREAD_GRAYSCALE)
#         depthinput1 = cv2.resize(depthinput1, (self.width, self.height))
        depthinput1 = depthinput1.astype(dtype=np.float32)
        depthinput1 = (depthinput1 / 255.0)
#         depthinput1 = np.transpose(depthinput1, [2, 0, 1])
        depthinput1 = depthinput1[np.newaxis, :, :]
    
        # load depth image2
        depthinput2 = cv2.imread(self.datas['warp_depth2']['image'][index],cv2.IMREAD_GRAYSCALE)
#         depthinput2 = cv2.resize(depthinput2, (self.width, self.height))
        depthinput2 = depthinput2.astype(dtype=np.float32)
        depthinput2 = (depthinput2 / 255.0)
#         depthinput2 = np.transpose(depthinput2, [2, 0, 1])
        depthinput2 = depthinput2[np.newaxis, :, :]



        # convert to tensor
        warp1_tensor = torch.tensor(warp1)
        warp2_tensor = torch.tensor(warp2)
        mask1_tensor = torch.tensor(mask1)
        mask2_tensor = torch.tensor(mask2)
        depth_warp1_tensor = torch.tensor(depthinput1)
        depth_warp2_tensor = torch.tensor(depthinput2)

        #return (input1_tensor, input2_tensor, mask1_tensor, mask2_tensor)

        if_exchange = random.randint(0,1)
        if if_exchange == 0:
            #print(if_exchange)
            return (warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, depth_warp1_tensor, depth_warp2_tensor)
        else:
            #print(if_exchange)
            return (warp2_tensor, warp1_tensor, mask2_tensor, mask1_tensor, depth_warp2_tensor, depth_warp1_tensor)


    def __len__(self):

        return len(self.datas['warp1']['image'])

class TestDataset(Dataset):
    def __init__(self, data_path):

        self.test_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'warp1' or data_name == 'warp2' or data_name == 'mask1' or data_name == 'mask2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()

        print(self.datas.keys())

    def __getitem__(self, index):


                # load image1
        warp1 = cv2.imread(self.datas['warp1']['image'][index])
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        # load image2
        warp2 = cv2.imread(self.datas['warp2']['image'][index])
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        # load mask1
        mask1 = cv2.imread(self.datas['mask1']['image'][index])
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = np.expand_dims(mask1[:,:,0], 2) / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        # load mask2
        mask2 = cv2.imread(self.datas['mask2']['image'][index])
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = np.expand_dims(mask2[:,:,0], 2) / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

        # convert to tensor
        warp1_tensor = torch.tensor(warp1)
        warp2_tensor = torch.tensor(warp2)
        mask1_tensor = torch.tensor(mask1)
        mask2_tensor = torch.tensor(mask2)

        return (warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

    def __len__(self):

        return len(self.datas['warp1']['image'])


