This is the official implementation of the PMFISyn model.



Five-fold cross validation is performed on the O'Neil dataset:
generate processed dataset: python create\_data.py
train: python training.py



Leave-one-out experiments:
generate processed dataset: python splite\_leave\_out\_data.py 
python train\_PMFISyn\_leave\_out.py --leave\_type leave\_drug --dropout\_rate 0.1 --device\_num 0
python train\_PMFISyn\_leave\_out.py --leave\_type leave\_comb --dropout\_rate 0.1 --device\_num 0
python train\_PMFISyn\_leave\_out.py --leave\_type leave\_cell --dropout\_rate 0.1 --device\_num 0



Generalization test:
python training\_generalization.py

labels.csv is the O'Neil dataset, independent.csv is the AstraZeneca dataset.

