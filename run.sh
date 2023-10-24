data_path=~/workspace/data/fabind

python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
accelerate launch fabind/main_fabind.py \
    --batch_size 5 \
    -d 0 \
    -m 5 \
    --data-path $data_path \
    --label baseline \
    --addNoise 5 \
    --resultFolder ./results \
    --use-compound-com-cls \
    --total-epochs 500 \
    --exp-name fabind-onlydocking-from-scratch-dismap15 \
    --coord-loss-weight 1.0 \
    --pair-distance-loss-weight 1.0 \
    --pair-distance-distill-loss-weight 1.0 \
    --pocket-cls-loss-weight 1.0 \
    --pocket-distance-loss-weight 0.05 \
    --lr 5e-05 --lr-scheduler poly_decay \
    --distmap-pred mlp \
    --n-iter 8 --mean-layers 4 \
    --refine refine_coord \
    --coordinate-scale 5 \
    --hidden-size 512 \
    --geometry-reg-step-size 0.001 \
    --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer \
    --noise-for-predicted-pocket 0 \
    --clip-grad \
    --random-n-iter \
    --pocket-idx-no-noise \
    --pocket-cls-loss-func bce \
    --use-esm2-feat --disable-validate --dis-map-thres 15 --onlydocking-from-scratch