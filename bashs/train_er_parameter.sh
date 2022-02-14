# python train.py --model gnn --cuda_id -1 --data_path data/er_parameter/beta5/train_data/SIR_100.pkl    
# python train.py --model gnn --cuda_id -1 --data_path data/er_parameter/gamma5/train_data/SIR_100.pkl 
python er_parameter.py --model gnn --diff SIR --parameter beta --num_status 3 
python er_parameter.py --model gnn --diff SIR --parameter gamma --num_status 3  

# python train.py --model nedmp --cuda_id -1 --data_path data/er_parameter/beta5/train_data/SIR_100.pkl         
# python train.py --model nedmp --cuda_id -1 --data_path data/er_parameter/gamma5/train_data/SIR_100.pkl 
python er_parameter.py --model nedmp --diff SIR --parameter beta --num_status 3 
python er_parameter.py --model nedmp --diff SIR --parameter gamma --num_status 3  