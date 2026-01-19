import warnings

import pandas as pd
import torch_geometric.deprecation

# 忽略特定的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric.deprecation')
import random
import torch.nn.functional as F
import torch.nn as nn
from models.PMFISyn import PMFISyn
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
EARLY_STOP_PATIENCE = 30
MIN_EPOCHS = 80

# training function at each epoch
# 改为（混合精度版本）
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch, scaler):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        y = data[0].y.view(-1, 1).long().to(device).squeeze(1)

        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast():
            output = model(data1, data2)
            loss = loss_fn(output, y)

        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % LOG_INTERVAL == 0:
            processed_samples = batch_idx * TRAIN_BATCH_SIZE
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                processed_samples,
                len(drug1_loader_train.dataset),
                100. * processed_samples / len(drug1_loader_train.dataset),
                loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


modeling = PMFISyn

TRAIN_BATCH_SIZE = 2048
TEST_BATCH_SIZE = 2048
LR = 0.0003
LOG_INTERVAL = 50
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

print(f'Early stopping patience: {EARLY_STOP_PATIENCE}')  # 显示早停参数
print(f'Minimum epochs: {MIN_EPOCHS}')  # 显示最小训练轮数

# 数据文件
train_datafile = 'labels'
test_datafile = 'independent'

# CPU or GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

# 数据预处理（如果需要）
cellfile = 'data/cell_features.csv'

# 检查并创建训练数据
processed_train_file_drug1 = f'data/processed/{train_datafile}_drug1.pt'
if not os.path.isfile(processed_train_file_drug1):
    from creat_data import creat_data
    creat_data(train_datafile, cellfile)

# 检查并创建测试数据
processed_test_file_drug1 = f'data/processed/{test_datafile}_drug1.pt'
if not os.path.isfile(processed_test_file_drug1):
    from creat_data import creat_data
    creat_data(test_datafile, cellfile)

# 加载训练数据
drug1_data_train = TestbedDataset(root='data', dataset=train_datafile + '_drug1')
drug2_data_train = TestbedDataset(root='data', dataset=train_datafile + '_drug2')

# 加载测试数据
drug1_data_test = TestbedDataset(root='data', dataset=test_datafile + '_drug1')
drug2_data_test = TestbedDataset(root='data', dataset=test_datafile + '_drug2')

lenth_train = len(drug1_data_train)
lenth_test = len(drug1_data_test)

print('Train data length:', lenth_train)
print('Test data length:', lenth_test)

# 运行5次独立的训练-测试（类似稳定性和多次运行）
k = 5
for i in range(k):
    print(f'\nStarting run {i + 1}/{k}')

    # 打乱训练数据顺序（可选，用于数据增强）
    random_num_train = random.sample(range(0, lenth_train), lenth_train)
    drug1_data_train_shuffled = drug1_data_train[random_num_train]
    drug2_data_train_shuffled = drug2_data_train[random_num_train]

    # 构建训练DataLoader
    drug1_loader_train = DataLoader(drug1_data_train_shuffled, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    drug2_loader_train = DataLoader(drug2_data_train_shuffled, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 构建测试DataLoader（不打乱）
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 初始化模型
    model = modeling(
        n_output=2,
        gat_dims=(32, 64, 128),
        num_features_xd=78,
        num_features_xt=954,
        fingerprint_dim=2048,
        output_dim=128,
        dropout=0.2,
        num_mpgr_layers=2,
        heads=1
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=3e-4
    )



    # 结果文件夹
    result_folder = 'result/Generalization'
    os.makedirs(result_folder, exist_ok=True)

    file_AUCs = f'{result_folder}/PMFISyn_generalization_{i}.txt'
    AUCs_header = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs_header + '\n')

    best_auc = 0
    no_improve_count = 0
    best_epoch = 0
    best_model_state = None

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1, scaler)

        train_T, train_S, train_Y = predicting(model, device, drug1_loader_train, drug2_loader_train)
        train_AUC = roc_auc_score(train_T, train_S)

        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
        # T:真实标签, S:预测分数, Y:预测标签

        # 计算指标
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y, zero_division=1)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)

        print(f'Epoch {epoch + 1}: Train AUC: {train_AUC:.4f}, Test AUC: {AUC:.4f}')

        # 早停逻辑
        if AUC > best_auc:
            best_auc = AUC
            no_improve_count = 0
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            print(f' New best AUC: {best_auc:.4f} at epoch {epoch + 1}')

        else:
            no_improve_count += 1
            print(f' No improvement for {no_improve_count}/{EARLY_STOP_PATIENCE} epochs, best AUC: {best_auc:.4f}')

        # 保存每个epoch的结果
        AUCs = [epoch + 1, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
        save_AUCs(AUCs, file_AUCs)

        # 早停检查
        if epoch >= MIN_EPOCHS and no_improve_count >= EARLY_STOP_PATIENCE:
            print(f' Early stopping triggered at epoch {epoch + 1}!')
            print(f' Best AUC: {best_auc:.4f} achieved at epoch {best_epoch + 1}')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(' Loaded best model weights')
            break
    else:
        print(f' Completed all {NUM_EPOCHS} epochs for run {i + 1}')
        print(f' Best AUC: {best_auc:.4f} achieved at epoch {best_epoch + 1}')

    with open(file_AUCs, 'a') as f:
        f.write(f'best_auc:{best_auc}\n')

    print(f'Finished run {i + 1}/{k}\n')

print('All generalization runs completed!')