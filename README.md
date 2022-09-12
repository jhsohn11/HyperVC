# HyperVC

## Abstract

Recently there is an increasing scholarly interest in time-varying knowledge graphs, or temporal knowledge graphs (TKG). Previous research suggests diverse approaches to TKG reasoning that uses historical information. However, less attention has been given to the hierarchies within such information at different timestamps. Given that TKG is a sequence of knowledge graphs based on time, the chronology in the sequence derives hierarchies between the graphs. Furthermore, each knowledge graph has its hierarchical level which may differ from one another. To address these hierarchical characteristics in TKG, we propose HyperVC, which utilizes hyperbolic space that better encodes the hierarchies than Euclidean space. The chronological hierarchies between knowledge graphs at different timestamps are represented by embedding the knowledge graphs as vectors in a common hyperbolic space. Additionally, diverse hierarchical levels of knowledge graphs are represented by adjusting the curvatures of hyperbolic embeddings of their entities and relations. Experiments on four benchmark datasets show substantial improvements, especially on the datasets with higher hierarchical levels.



## Installation

1. Select Poincare/Lorentz model of hyperbolic space first.

2. Before running, we preprocess the datasets.

>cd data/DATA_NAME \
>python3 get_history_graph.py


3. Pretrain the global model.

>python3 pretrain.py -d DATA_NAME --n-hidden 200 --lr 1e-3 --max-epochs 100 --batch-size 1024

4. Train the model.

>python3 train.py -d DATA_NAME --n-hidden 200 --lr 1e-3 --max-epochs 100 --batch-size 1024

5. Finally, test the model.

>python3 test.py -d DATA_NAME --n-hidden 200