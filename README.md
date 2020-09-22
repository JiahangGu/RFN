# Residual Fractal Network for Single Image Super Resolution by Widening and Deepening
This repo is for RFN introduced in the following paper

Residual Fractal Network for Single Image Super Resolution by Widening and Deepening, ICPR 2020

The code is built on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN(PyTorch)](https://github.com/yulunzhang/RCAN), and trained/tested on Ubuntu 16.04 environment (Python3.6, PyTorch1.1.0, cudatoolkit9.0) with RTX1080Ti.

### Contents

1. Introduction
2. Train
3. Test
4. Results
5. Acknowledgements

#### Introduction

The architecture of the convolutional neural network (CNN) plays an important role in single image super-resolution (SISR). However, most models proposed in recent years usually transplant methods or architectures that perform well in other vision fields. Thence they do not combine the characteristics of super-resolution (SR) and ignore the key information brought by the recurring texture feature in the image. To utilize patch-recurrence in SR and the high correlation of texture, we propose a residual fractal convolutional block (RFCB) and expand its depth and width to obtain residual fractal network (RFN), which contains deep residual fractal network (DRFN) and wide residual fractal network (WRFN). RFCB is recursive
with multiple branches of magnified receptive field. Through the phased feature fusion module, the network focuses on extracting high-frequency texture feature that repeatedly appear in the
image. We also introduce residual in residual (RIR) structure to RFCB that enables abundant low-frequency feature feed into deeper layers and reduce the difficulties of network training. RFN is the first supervised learning method to combine the patch-recurrence characteristic in SISR into network design. Extensive experiments demonstrate that RFN outperforms state-of- the-art SISR methods in terms of both quantitative metrics and visual quality, while the amount of parameters has been greatly optimized.

#### Train

1. Download DIV2K training data from [dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and specify '--dir_data' corresponds to your own data path in option.py. For more information, please refer to [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch). 
2. Download models for RFN from [Baidu Netdisk](https://pan.baidu.com/s/1O0SkU9ixT0lkHS-Y7pgbKg) with extraction code 4p3e and place them in '/EDSR/experiment'.
3. Cd to 'EDSR/src', run train_drfn.sh or train_wrfn.sh to train models. You can also use following scripts to train models. Option '--degradation' in option.py must be changed to 'BD' when training BDX3 model and set blur-downsampled images under 'EDSR/data'.

`# BI, scale 2, 3, 4`

`# DRFN_BIX2`

`CUDA_VISIBLE_DEVICES=0 python main.py --model DRFN --save DRFN_BIX2 --scale 2 --save_results --epochs 1000 --patch_size 96 --decay '200-400-600-800-1000' --reset`

`# DRFN_BIX3`

`CUDA_VISIBLE_DEVICES=0 python main.py --model DRFN --save DRFN_BIX3 --scale 3 --save_results --epochs 1000 --patch_size 144 --decay '200-400-600-800-1000' --reset`

`# DRFN_BIX4`

`CUDA_VISIBLE_DEVICES=0 python main.py --model DRFN --save DRFN_BIX4 --scale 4 --save_results --epochs 1000 --patch_size 192 --decay '200-400-600-800-1000' --reset`

`# WRFN_BIX2`

`CUDA_VISIBLE_DEVICES=0 python main.py --model WRFN --save WRFN_BIX2 --scale 2 --save_results --epochs 1000 --patch_size 96 --decay '200-400-600-800-1000' --reset`

`# WRFN_BIX3`

`CUDA_VISIBLE_DEVICES=0 python main.py --model WRFN --save WRFN_BIX3 --scale 3 --save_results --epochs 1000 --patch_size 144 --decay '200-400-600-800-1000' --reset`

`# WRFN_BIX4`

`CUDA_VISIBLE_DEVICES=0 python main.py --model WRFN --save WRFN_BIX4 --scale 4 --save_results --epochs 1000 --patch_size 192 --decay '200-400-600-800-1000' --reset`

#### Test

1. Download 5 benckmark datasets (Set5, Set14, Urban100, B100, Manga109) and place them in 'RCAN_TestCode/HR'. Run Prepare_TestData_HR_LR.m to obtain test data.
2. Download trained models for RFN from [Baidu Netdisk](https://pan.baidu.com/s/1O0SkU9ixT0lkHS-Y7pgbKg) with extraction code 4p3e and place them in 'RCAN_TestCode/model'.
3. Cd to 'RCAN_TestCode/code', run testDRFN.sh or testWRFN.sh to generate super-resolution results. You can also run following scripts to generate results. Add option '--self-ensemble' in scripts to get ensembled results corresponds to plus (+) in out paper. To test WRFN models please replace DRFN with WRFN and change corresponding model path.

`# BIX2`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x2.pt --test_only --save_results --chop --save 'DRFN_test' --testpath ../LR/LRBI --testset Set5
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x2.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Set14
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x2.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Urban100
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x2.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset B100
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x2.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Manga109`

`#BIX3`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x3.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Set5
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x3.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Set14
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x3.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Urban100
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x3.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset B100
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x3.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Manga109`

`#BIX4`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x4.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Set5
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x4.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Set14
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x4.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Urban100
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x4.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset B100
`

`CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model RFN --n_feats 64 --pre_train /home/wxr/server2/wxr/gjh/RCAN/RCAN_TestCode/model/DRFN/model_best_x4.pt --test_only --save_results --chop --save 'DRFN_final' --testpath ../LR/LRBI --testset Manga109`

4. Run Evaluate_PSNR_SSIM.m to evaluate PSNR/SSIM values for paper.

#### Results

##### Quantitative Results with BI degradation model. Best and second best results are highlighted and underlined

![image-20200922114502189](C:\Users\fengzhizi\AppData\Roaming\Typora\typora-user-images\image-20200922114502189.png)

##### Quantitative Results with BD degradation model. Best and second best results are highlighted and underlined

![image-20200922114443378](C:\Users\fengzhizi\AppData\Roaming\Typora\typora-user-images\image-20200922114443378.png)

##### Visual Results

![wrfn_visual_1](D:\learning\academic\ICPR2020\figs\wrfn_visual_1.png)

![wrfn_visual_2](D:\learning\academic\ICPR2020\figs\wrfn_visual_2.png)

![wrfn_visual_3](D:\learning\academic\ICPR2020\figs\wrfn_visual_3.png)

For more results, please download from [Baidu Netdisk](https://pan.baidu.com/s/1O0SkU9ixT0lkHS-Y7pgbKg) with extraction code 4p3e. The SR results produced by previous methods  are also provided.

#### Acknowledgements

This code is built on [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN](https://github.com/yulunzhang/RCAN). Please cite the related papers if you find the code helpful in your research or work.