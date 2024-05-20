CUDA_VISIBLE_DEVICES=$1 python train_MOA.py \
    --epochs 200 \
    --batch-size 64 \
    --trigger-path 'data/trigger_set/pics' \
    --emb-path 'data/trigger_set/emb_pics' \
    --key-type 'image' \
    --train-private \
    --use-trigger-as-passport \
    --exp-id 1 \
    --arch $2 \
    --dataset $3 \
    --norm-type $4\
    --passport-config $5 \
