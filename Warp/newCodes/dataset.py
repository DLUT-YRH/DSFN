# from torch.utils.data import Dataset
# import  numpy as np
# import cv2, torch
# import os
# import glob
# from collections import OrderedDict
# import random


# class TrainDataset(Dataset):
#     def __init__(self, data_path):

#         self.width = 512
#         self.height = 512
#         self.train_path = data_path
#         self.datas = OrderedDict()
        
#         datas = glob.glob(os.path.join(self.train_path, '*'))
#         for data in sorted(datas):
#             data_name = data.split('/')[-1]
#             if data_name == 'input1' or data_name == 'input2' :
#                 self.datas[data_name] = {}
#                 self.datas[data_name]['path'] = data
#                 self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
#                 self.datas[data_name]['image'].sort()
#         print(self.datas.keys())

#     def __getitem__(self, index):
        
#         # load image1
#         input1 = cv2.imread(self.datas['input1']['image'][index])
#         input1 = cv2.resize(input1, (self.width, self.height))
#         input1 = input1.astype(dtype=np.float32)
#         input1 = (input1 / 127.5) - 1.0
#         input1 = np.transpose(input1, [2, 0, 1])
        
#         # load image2
#         input2 = cv2.imread(self.datas['input2']['image'][index])
#         input2 = cv2.resize(input2, (self.width, self.height))
#         input2 = input2.astype(dtype=np.float32)
#         input2 = (input2 / 127.5) - 1.0
#         input2 = np.transpose(input2, [2, 0, 1])
        
#         # convert to tensor
#         input1_tensor = torch.tensor(input1)
#         input2_tensor = torch.tensor(input2)
        
#         #print("fasdf")
#         if_exchange = random.randint(0,1)
#         if if_exchange == 0:
#             #print(if_exchange)
#             return (input1_tensor, input2_tensor)
#         else:
#             #print(if_exchange)
#             return (input2_tensor, input1_tensor)

#     def __len__(self):

#         return len(self.datas['input1']['image'])

# class TestDataset(Dataset):
#     def __init__(self, data_path):

#         self.width = 512
#         self.height = 512
#         self.test_path = data_path
#         self.datas = OrderedDict()
        
#         datas = glob.glob(os.path.join(self.test_path, '*'))
#         for data in sorted(datas):
#             data_name = data.split('/')[-1]
#             if data_name == 'input1' or data_name == 'input2' :
#                 self.datas[data_name] = {}
#                 self.datas[data_name]['path'] = data
#                 self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
#                 self.datas[data_name]['image'].sort()
#         print(self.datas.keys())

#     def __getitem__(self, index):
        
#         # load image1
#         input1 = cv2.imread(self.datas['input1']['image'][index])
#         #input1 = cv2.resize(input1, (self.width, self.height))
#         input1 = input1.astype(dtype=np.float32)
#         input1 = (input1 / 127.5) - 1.0
#         input1 = np.transpose(input1, [2, 0, 1])
        
#         # load image2
#         input2 = cv2.imread(self.datas['input2']['image'][index])
#         #input2 = cv2.resize(input2, (self.width, self.height))
#         input2 = input2.astype(dtype=np.float32)
#         input2 = (input2 / 127.5) - 1.0
#         input2 = np.transpose(input2, [2, 0, 1])
        
#         # convert to tensor
#         input1_tensor = torch.tensor(input1)
#         input2_tensor = torch.tensor(input2)

#         return (input1_tensor, input2_tensor)

#     def __len__(self):

#         return len(self.datas['input1']['image'])

from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' or data_name == "depthInput1" or data_name == "depthInput2": # add depth input
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # add depth input
        # load depth image1
        depthinput1 = cv2.imread(self.datas['depthInput1']['image'][index],cv2.IMREAD_GRAYSCALE)
        depthinput1 = cv2.resize(depthinput1, (self.width, self.height))
        depthinput1 = depthinput1.astype(dtype=np.float32)
#         depthinput1 = (depthinput1 / 255.0)
#         depthinput1 = np.transpose(depthinput1, [2, 0, 1])
        depthinput1 = depthinput1[np.newaxis, :, :]
    
        # load depth image2
        depthinput2 = cv2.imread(self.datas['depthInput2']['image'][index],cv2.IMREAD_GRAYSCALE)
        depthinput2 = cv2.resize(depthinput2, (self.width, self.height))
        depthinput2 = depthinput2.astype(dtype=np.float32)
#         depthinput2 = (depthinput2 / 255.0)
#         depthinput2 = np.transpose(depthinput2, [2, 0, 1])
        depthinput2 = depthinput2[np.newaxis, :, :]
    
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        depthInput1_tensor = torch.tensor(depthinput1)
        depthInput2_tensor = torch.tensor(depthinput2)

        #print("fasdf")
        if_exchange = random.randint(0,1)
        if if_exchange == 0:
            #print(if_exchange)
            return (input1_tensor, input2_tensor, depthInput1_tensor, depthInput2_tensor)
        else:
            #print(if_exchange)
            return (input2_tensor, input1_tensor, depthInput2_tensor, depthInput1_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])

class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        #input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        #input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])


class TestDepthOutDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' or data_name == "depthInput1" or data_name == "depthInput2": # add depth input
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # add depth input
        # load depth image1
        depthinput1 = cv2.imread(self.datas['depthInput1']['image'][index],cv2.IMREAD_GRAYSCALE)
        depthinput1 = cv2.resize(depthinput1, (self.width, self.height))
        depthinput1 = depthinput1.astype(dtype=np.float32)
#         depthinput1 = (depthinput1 / 255.0)
#         depthinput1 = np.transpose(depthinput1, [2, 0, 1])
        depthinput1 = depthinput1[np.newaxis, :, :]
    
        # load depth image2
        depthinput2 = cv2.imread(self.datas['depthInput2']['image'][index],cv2.IMREAD_GRAYSCALE)
        depthinput2 = cv2.resize(depthinput2, (self.width, self.height))
        depthinput2 = depthinput2.astype(dtype=np.float32)
#         depthinput2 = (depthinput2 / 255.0)
#         depthinput2 = np.transpose(depthinput2, [2, 0, 1])
        depthinput2 = depthinput2[np.newaxis, :, :]

        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        depthInput1_tensor = torch.tensor(depthinput1)
        depthInput2_tensor = torch.tensor(depthinput2)


        return (input1_tensor, input2_tensor, depthInput1_tensor, depthInput2_tensor)


    def __len__(self):

        return len(self.datas['input1']['image'])



