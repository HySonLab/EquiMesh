python experiments/TOSCA/train_tosca.py --model TOSCA --layer $1 --seed 0 --epochs 100 --lr 0.0005 --exp_name "results/tosca_0_$1" -save_model
python experiments/TOSCA/train_tosca.py --model TOSCA --layer $1 --seed 1 --epochs 100 --lr 0.0005 --exp_name "results/tosca_1_$1" -save_model
python experiments/TOSCA/train_tosca.py --model TOSCA --layer $1 --seed 2 --epochs 100 --lr 0.0005 --exp_name "results/tosca_2_$1" -save_model
python experiments/TOSCA/train_tosca.py --model TOSCA --layer $1 --seed 3 --epochs 100 --lr 0.0005 --exp_name "results/tosca_3_$1" -save_model
python experiments/TOSCA/train_tosca.py --model TOSCA --layer $1 --seed 4 --epochs 100 --lr 0.0005 --exp_name "results/tosca_4_$1" -save_model