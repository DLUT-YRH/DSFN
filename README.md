# Depth-Supervised Fusion Network for Seamless-Free Image Stitching

<center>Zhiying Jiang<sup>1</sup> &ensp; Ruhao Yan<sup>2</sup> &ensp; Zengxi Zhang<sup>2</sup> &ensp; Bowei Zhang<sup>2</sup> &ensp; Jinyuan Liu<sup>2</sup>

<sup>1</sup>College of Information Science and Technology, Dalian Maritime University

<sup>2</sup>School of Software Technology, Dalian University of Technology

<img src="./fig.png">
</center>

## Code
**Requirements**

Linux<br>
Python 3<br>
numpy >= 1.19.5<br>
pytorch >= 1.7.1<br>
CUDA >= 11.0<br>
We use an NVIDIA RTX 3090 GPU to achieve this task. If you are using hardware of other models, please adapt the versions of tools such as CUDA and pytorch by yourself.

**Installation**

You can create a Conda environment through the following command:
```
conda env create -f environment.yml
```

## Datasets

Please download UDIS-D dataset([Google Drive](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY) or [Baidu Yun](https://pan.baidu.com/share/init?surl=3KZ29e487datgtMgmb9laQ?pwd=1234)) and IVSD dataset([Google Drive](https://drive.google.com/file/d/1EFS0O-3KujvRJvcRx_Me5W2fdn9jRKGc) or [Baidu Yun](https://pan.baidu.com/share/init?surl=ZP4hgBovXnsLHcOReCGnrg&pwd=ssfv)) for training and testing.

## How to Run

**Training**

Please first use [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2/tree/main) to generate the depth map of the dataset.

Please set the training dataset path in Warp/newCodes/train.py and Fusion/newCodes/train.py, and run the following command under the corresponding path to start the training.

```
python train.py
```
***Please note that the training in the fusion phase depends on the result of the warp phase. Pay attention to the training sequence. After completing the training in the warp phase, warp the training set to generate the fusion training set.***


**Testing**

Please set the training dataset path in Warp/newCodes/test_output.py and Fusion/newCodes/test.py, and run the following command under the corresponding path to start the training.


>For Warp:
>```
>python test_output.py
>```
>For Fusion:
>```
>python test.py
>```
***Please note that, similar to the training phase, the testing phase still has sequential nature.***

## Download
Please download the pre-trained model([Google Drive](https://drive.google.com/drive/folders/1qsvIj7iN62gdHJLytTAzfvrOZaGaqXzR) or [Baidu Yun](https://pan.baidu.com/s/1VpBGxePAbQoL7IbNspQMuA?pwd=2025)) and place it in the corresponding path.




