This is the official implementation of the PMFISyn model.


Five-fold cross validation is performed on the O'Neil dataset:
generate processed dataset: python create_data.py (files are generated)
train: python training.py


Leave-one-out experiments:
generate processed dataset: python splite_leave_out_data.py (files are generated)
python train_PMFISyn_leave_out.py --leave_type leave_drug --dropout_rate 0.1 --device_num 0
python train_PMFISyn_leave_out.py --leave_type leave_comb --dropout_rate 0.1 --device_num 0
python train_PMFISyn_leave_out.py --leave_type leave_cell --dropout_rate 0.1 --device_num 0



Generalization test:
python training_generalization.py

labels.csv is the O'Neil dataset, independent.csv is the AstraZeneca dataset.