# iRL

## Run eLSTM on the prediction task:  
python eLSTM_copy_task_main.py --data_dir 'utils/data_copy_task' --level 500 --model_type 11 --no_embedding --num_layer 1 --hidden_size 2048 --dropout 0.0 --batch_size 128 --learning_rate 3e-5 --clip 1.0 --grad_cummulate 1 --num_epoch 50  --seed 1

## Run rtu-prediction branch on google colab (run these in separate cells):  
!git clone -b rtu-prediction https://github.com/CMPUT-469-iRL/iRL.git
!python iRL/eLSTM_copy_task_main.py --data_dir 'iRL/utils/data_copy_task' --level 500 --model_type 11 --no_embedding --num_layer 1 --hidden_size 2048 --dropout 0.0 --batch_size 128 --learning_rate 3e-5 --clip 1.0 --grad_cummulate 1 --num_epoch 50  --seed 1
