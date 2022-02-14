# # Traing
# python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/dolphins/train_data/SIR_150.pkl     >logs/dolphins_gnn.log     
# python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/fb-food/train_data/SIR_150.pkl      >logs/fb-food_gnn.log      
# python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/fb-social/train_data/SIR_150.pkl    >logs/fb-social_gnn.log    
# python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/norwegain/train_data/SIR_150.pkl    >logs/norwegain_gnn.log    
# python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/openflights/train_data/SIR_150.pkl  >logs/openflights_gnn.log  
python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/top-500/train_data/SIR_150.pkl      >logs/top-500_gnn.log 
 
python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/dolphins/train_data/SIR_150.pkl     --testing 
python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/fb-food/train_data/SIR_150.pkl      --testing 
python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/fb-social/train_data/SIR_150.pkl    --testing 
python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/norwegain/train_data/SIR_150.pkl    --testing 
python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/openflights/train_data/SIR_150.pkl  --testing 
python train.py --model gnn --cuda_id 0 --diff SIR --data_path data/realnets/top-500/train_data/SIR_150.pkl      --testing