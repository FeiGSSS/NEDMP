# # Traing
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/dolphins/train_data/SIR_150.pkl     >logs/dolphins_nedmp.log     
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/fb-food/train_data/SIR_150.pkl      >logs/fb-food_nedmp.log      
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/fb-social/train_data/SIR_150.pkl    >logs/fb-social_nedmp.log    
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/norwegain/train_data/SIR_150.pkl    >logs/norwegain_nedmp.log    
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/openflights/train_data/SIR_150.pkl  >logs/openflights_nedmp.log  
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/top-500/train_data/SIR_150.pkl      >logs/top-500_nedmp.log 
 
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/dolphins/train_data/SIR_150.pkl     --testing 
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/fb-food/train_data/SIR_150.pkl      --testing 
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/fb-social/train_data/SIR_150.pkl    --testing 
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/norwegain/train_data/SIR_150.pkl    --testing 
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/openflights/train_data/SIR_150.pkl  --testing 
python train.py --model nedmp --cuda_id 1 --diff SIR --data_path data/realnets/top-500/train_data/SIR_150.pkl      --testing