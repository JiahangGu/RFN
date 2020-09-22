CUDA_VISIBLE_DEVICES=1 python main.py --model DRFN --save DRFN_BIX2_val10 --scale 2 --save_results --epochs 1000 --patch_size 96 --decay '200-400-600-800-1000' --reset
