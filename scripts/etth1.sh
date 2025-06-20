    python -u run_longExp.py \
    --data ETTh1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --enc_in 7 \
    --seq_len 720 \
    --pred_len 96 \
    --e_layers 3 \
    --d_model 32 \
    --dropout 0.2 \
    --patch_len 8 \
    --stride 8 \
    --pad_size 1 \
    --local_size 2\
    --batch_size 1024 \
    --patience 20 \
    --learning_rate 0.001 \
    --use_multi_gpu \
    --devices 0,1

    python -u run_longExp.py \
    --data ETTh1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --enc_in 7 \
    --seq_len 720 \
    --pred_len 192 \
    --e_layers 1 \
    --d_model 32 \
    --dropout 0.2 \
    --patch_len 8 \
    --stride 8 \
    --pad_size 1 \
    --local_size 2\
    --batch_size 1024 \
    --patience 20 \
    --learning_rate 0.005 \
    --use_multi_gpu \
    --devices 0,1

    python -u run_longExp.py \
    --data ETTh1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --enc_in 7 \
    --seq_len 720 \
    --pred_len 336 \
    --e_layers 1 \
    --d_model 32 \
    --dropout 0.2 \
    --patch_len 8 \
    --stride 8 \
    --pad_size 1 \
    --local_size 2\
    --batch_size 1024 \
    --patience 20 \
    --learning_rate 0.005 \
    --use_multi_gpu \
    --devices 0,1

    python -u run_longExp.py \
    --data ETTh1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --enc_in 7 \
    --seq_len 720 \
    --pred_len 720 \
    --e_layers 3 \
    --d_model 64 \
    --dropout 0.2 \
    --patch_len 8 \
    --stride 8 \
    --pad_size 1 \
    --local_size 2\
    --batch_size 1024 \
    --patience 20 \
    --learning_rate 0.001 \
    --use_multi_gpu \
    --devices 0,1