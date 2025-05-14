## 环境配置

```
git clone https://github.com/Thislu13/CellSeg.git
cd cellseg
```

* 创建 cellpose，SAM，StarDist，DeepCell的环境
* 注意：如果你希望使用[PyTorch](https://pytorch.org/)请根据自己的GPU版本下载对应的版本

```
# python=3.8 env
conda create -n sellseg python=3.8
conda activate sellseg
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

* 使用命令安装MEDIAR和cellprofile的环境 ，在cell_seg.py中指向该环境(添加conda路径)

```
conda env create -f old_model/MEDIAR/environment.yaml
conda env create -f old_model/cellprofiler/environment.yaml
```

* 下载模型文件，并移动到 old_model/models 

[sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 

[MEDIAR](https://drive.google.com/drive/folders/1eZLGuQkxF5ouBgTA2UuH0beLcm635ADS)  

## 模型推理脚本

 运行脚本

```
python cell_seg.py -i your_inputpath -o your_outputpath -m  cellpose3 sam -t ss -g True  
```

* -i  图片输入路径
* -o 输出掩码路径
* -m 要使用的算法（cellpose cellpose3 MEDIAR deepcell stardist cellprofiler）可指定多个
* -t 染色类型 （ss/he/dapi/mif）
* -g 是否使用gpu （True/False 或者索引）

## 模型对比脚本

```
python eval/multi_cell_eval.py -g gt_path -d dt_path -o result_path
```

* -g 标签位置
* -d 模型推理文件夹
* -o 输出文件夹

## Cellpose推理脚本

```
python new_model/cellpose_fine/pre_cellpose_fine.py -m model_path -i input_dir -o out_dir
```

* -m 权重位置
* -i 输入文件夹
* -o 输出文件夹

## 参考

>  [cellpose](https://github.com/MouseLand/cellpose)  
>
>  [cellpose3](https://github.com/MouseLand/cellpose)  
>
>  [deepcell](https://github.com/vanvalenlab/deepcell-tf)  
>
>  [sam](https://github.com/facebookresearch/segment-anything)  
>
>  [mediar](https://github.com/Lee-Gihun/MEDIAR)  
>
>  [stardist](https://github.com/stardist/stardist)  
>
>  [cellprofiler](https://github.com/CellProfiler)  