CUDA_VISIBLE_DEVICES=1 python main.py --model WRFN --save WRFN_BIX2_val10 --scale 2 --save_results --epochs 600 --patch_size 96 --decay '100-200-300-400-500' --reset
