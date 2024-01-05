python experiments/FAUST/train_faust.py --model FAUST --layer $1 --seed 5 --epochs 200 --lr 0.001 --exp_name "results/faust0_$1/" -save_model
python experiments/FAUST/train_faust.py --model FAUST --layer $1 --seed 6 --epochs 200 --lr 0.001 --exp_name "results/faust1_$1/" -save_model
python experiments/FAUST/train_faust.py --model FAUST --layer $1 --seed 7 --epochs 200 --lr 0.001 --exp_name "results/faust2_$1/" -save_model
python experiments/FAUST/train_faust.py --model FAUST --layer $1 --seed 8 --epochs 200 --lr 0.001 --exp_name "results/faust3_$1/" -save_model
python experiments/FAUST/train_faust.py --model FAUST --layer $1 --seed 9 --epochs 200 --lr 0.001 --exp_name "results/faust4_$1/" -save_model
