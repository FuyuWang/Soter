CUDA_VISIBLE_DEVICES=1 python main.py --optim_obj latency --epochs 10 --accelerator Simba --workload resnet50 --layer_id 43 --batch_size 1
cd ../