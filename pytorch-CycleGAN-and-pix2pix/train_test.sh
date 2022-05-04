python train.py --dataroot $1 \
--name $2 --model var_cycle_gan \
--load_size 80 --crop_size 64 --ngf 16 --ndf 16 --n_epochs 50 \
--n_epochs_decay 50

python test.py --dataroot $1 \
--name $2 --model var_cycle_gan \
--load_size 80 --crop_size 64 --ngf 16 --ndf 16 --nlevels 2 \
--latent_dim 64