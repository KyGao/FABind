ckpt_path=/home/t-kaiyuangao/workspace/data/fabind-neurips/best_model.bin
data_path=/home/t-kaiyuangao/workspace/data/fabind-neurips
python fabind/test_fabind.py \
    --batch_size 4 \
    --data-path $data_path \
    --resultFolder ./results \
    --exp-name test_exp \
    --ckpt $ckpt_path \
#    --local-eval