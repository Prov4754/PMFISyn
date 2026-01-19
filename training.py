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

EARLY_STOP_PATIENCE = 50
MIN_EPOCHS = 200

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

TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LR = 0.0003
LOG_INTERVAL = 50
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

print(f'Early stopping patience: {EARLY_STOP_PATIENCE}')
print(f'Minimum epochs: {MIN_EPOCHS}')

datafile = 'labels'


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

lenth = len(drug1_data)
k = 5
pot = int(lenth/k)
print('lenth', lenth)
print('pot', pot)


random_num = random.sample(range(0, lenth), lenth)
for i in range(k):
    test_num = random_num[pot*i:pot*(i+1)]
    train_num = random_num[:pot*i] + random_num[pot*(i+1):]

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 初始化模型
    model = modeling(
        n_output=2,
        gat_dims=(32, 64, 128),
        num_features_xd=78,
        num_features_xt=954,
        fingerprint_dim=2048,
        output_dim=128,
        dropout=0.1,
        num_mpgr_layers=2,
        heads=1
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()
    # 在 training.py 中，把 optimizer 改成这样
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)



    best_auc = 0
    no_improve_count = 0
    best_epoch = 0
    best_model_state = None

    # 训练测试
    #model_file_name = 'result/PMFISyn(DrugA_DrugB)' + str(i) + '--model_' + datafile +  '.model'

    result_file_name = 'result/labels/PMFISyn(DrugA_DrugB)' + str(i) + '--result_' + datafile +  '.csv'
    file_AUCs = 'result/labels/PMFISyn(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile + '.txt'
    model_file_name = 'result/labels/PMFISyn(DrugA_DrugB)' + str(i) + '--model_' + datafile + '.pth'  # 模型保存路径


    # 确保结果目录存在
    os.makedirs('result', exist_ok=True)
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1, scaler)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
        # T:真实标签, S:预测分数, Y:预测标签

        # 计算相关得分
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

        # === 早停逻辑 ===
        if AUC > best_auc:
            # 性能提升，更新最佳结果
            best_auc = AUC
            no_improve_count = 0
            best_epoch = epoch
            best_model_state = model.state_dict().copy()  # 保存最佳模型状态
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'fold': i
            }, model_file_name)

            print(f' New best AUC: {best_auc:.4f} at epoch {epoch + 1}, model saved!')
        else:
            # 性能没有提升
            no_improve_count += 1
            print(f' No improvement for {no_improve_count}/{EARLY_STOP_PATIENCE} epochs, best AUC: {best_auc:.4f}')

            # 保存每个epoch的结果
        AUCs = [epoch + 1, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
        save_AUCs(AUCs, file_AUCs)

        # 计算其他指标（如果需要）
        # ret = [rmse(T, S), mse(T, S), pearson(T, S), spearman(T, S), ci(T, S)]

        # === 早停检查 ===
        if epoch >= MIN_EPOCHS and no_improve_count >= EARLY_STOP_PATIENCE:
            print(f' Early stopping triggered at epoch {epoch + 1}!')
            print(f' Best AUC: {best_auc:.4f} achieved at epoch {best_epoch + 1}')

            # 加载最佳模型状态
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(' Loaded best model weights for final evaluation')
            break
    else:
        # 如果正常完成所有epoch（没有触发break）
        print(f' Completed all {NUM_EPOCHS} epochs for fold {i + 1}')
        print(f' Best AUC: {best_auc:.4f} achieved at epoch {best_epoch + 1}')

    print(f'Finished fold {i + 1}/{k}\n')

print(' All folds completed!')



