1. Select Poincare/Lorentz model of hyperbolic space first.

2. Before running, we preprocess the datasets.

cd data/DATA_NAME
python3 get_history_graph.py

3. Pretrain the global model.

python3 pretrain.py -d DATA_NAME --n-hidden 200 --lr 1e-3 --max-epochs 100 --batch-size 1024

4. Train the model.

python3 train.py -d DATA_NAME --n-hidden 200 --lr 1e-3 --max-epochs 100 --batch-size 1024

5. Finally, test the model.

python3 test.py -d DATA_NAME --n-hidden 200