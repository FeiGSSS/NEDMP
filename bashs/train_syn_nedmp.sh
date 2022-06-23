Traing
python train.py --diff SIR --model nedmp --cuda_id -1 --data_path data/synthetic/tree/train_data/SIR_200.pkl       
python train.py --diff SIR --model nedmp --cuda_id -1 --data_path data/synthetic/barbell/train_data/SIR_200.pkl    
python train.py --diff SIR --model nedmp --cuda_id -1 --data_path data/synthetic/complete/train_data/SIR_200.pkl   
python train.py --diff SIR --model nedmp --cuda_id -1 --data_path data/synthetic/er03/train_data/SIR_200.pkl
python train.py --diff SIR --model nedmp --cuda_id -1 --data_path data/synthetic/er05/train_data/SIR_200.pkl       
python train.py --diff SIR --model nedmp --cuda_id -1 --data_path data/synthetic/er08/train_data/SIR_200.pkl       
python train.py --diff SIR --model nedmp --cuda_id -1 --data_path data/synthetic/grid/train_data/SIR_200.pkl       
python train.py --diff SIR --model nedmp --cuda_id -1 --data_path data/synthetic/regular_graph/train_data/SIR_200.pkl

# Testing--diff SIR 
python train.py --diff SIR --testing --model nedmp --cuda_id -1 --data_path data/synthetic/tree/train_data/SIR_200.pkl       
python train.py --diff SIR --testing --model nedmp --cuda_id -1 --data_path data/synthetic/barbell/train_data/SIR_200.pkl    
python train.py --diff SIR --testing --model nedmp --cuda_id -1 --data_path data/synthetic/complete/train_data/SIR_200.pkl   
python train.py --diff SIR --testing --model nedmp --cuda_id -1 --data_path data/synthetic/er03/train_data/SIR_200.pkl
python train.py --diff SIR --testing --model nedmp --cuda_id -1 --data_path data/synthetic/er05/train_data/SIR_200.pkl       
python train.py --diff SIR --testing --model nedmp --cuda_id -1 --data_path data/synthetic/er08/train_data/SIR_200.pkl       
python train.py --diff SIR --testing --model nedmp --cuda_id -1 --data_path data/synthetic/grid/train_data/SIR_200.pkl       
python train.py --diff SIR --testing --model nedmp --cuda_id -1 --data_path data/synthetic/regular_graph/train_data/SIR_200.pkl

