python train.py \
--pre_train True \
--save_mode 'epoch' \
--save_by_epoch 1 \
--save_by_iter 100000 \
--save_name_mode True \
--load_name '' \
--multi_gpu False \
--gpu_ids '0, 1, 2, 3' \
--cudnn_benchmark True \
--epochs 40 \
--batch_size 32 \
--lr_g 0.0002 \
--lr_d 0.0001 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0 \
--lr_decrease_mode 'epoch' \
--lr_decrease_epoch 10 \
--lr_decrease_iter 200000 \
--lr_decrease_factor 0.5 \
--z_dim 8 \
--random_type 'gaussian' \
--random_var 1.0 \
--num_workers 8 \
--lambda_l1 10 \
--lambda_gan 1 \
--lambda_ms 1 \
--pad 'reflect' \
--norm 'bn' \
--in_channels 1 \
--out_channels 3 \
--start_channels 32 \
--init_type 'normal' \
--init_gain 0.02 \
--gan_mode 'LSGAN' \
--additional_training_d 1 \
--task 'colorization' \
--baseroot '../ILSVRC2012_train_256' \
