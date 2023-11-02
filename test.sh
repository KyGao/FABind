
ckpt_path=/home/t-kaiyuangao/workspace/data/epoch_400/pytorch_model.bin
data_path=~/workspace/data/fabind
python fabind/test_fabind.py \
    --batch_size 4 \
    --data-path $data_path \
    --resultFolder ./results \
    --exp-name test_exp \
    --ckpt $ckpt_path \
#    --local-eval