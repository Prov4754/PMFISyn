import random
import torch.nn.functional as F
import torch.nn as nn
from models.PMFISyn import PMFISyn
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score
from sklearn import metrics
import pandas as pd
import numpy as np
import csv
import datetime
import argparse
import os
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx

parser = argparse.ArgumentParser(description='PMFISyn Leave-out Experiment')

parser.add_argument('--leave_type', type=str, help='The type of leaveout:  leave_drug, leave_comb, leave_cell')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='The dropout rate')
parser.add_argument('--device_num', type=int, default=0, help='The number of device')

args = parser.parse_args()

leave_type = args.leave_type
dropout_rate = args.dropout_rate
device_num = args.device_num

result_name = 'PMFISyn_' + str(leave_type) + "_drop_rate=" + str(dropout_rate)

modeling = PMFISyn


# ============ 数据处理辅助函数 ============
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}: ".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        print(f"警告:  无法解析SMILES:  {smiles}")
        return np.zeros(n_bits)


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def load_data(datafile, drug_smiles_file, cellfile):
    """
    加载数据并返回训练所需的所有信息

    Args:
        datafile: 数据文件路径 (CSV格式)
        drug_smiles_file: 药物SMILES文件路径
        cellfile: 细胞特征文件路径

    Returns:
        drug1, drug2, cell, label, smile_graph, cell_features, fingerprint_dict
    """
    # 读取细胞特征文件
    cell_features = []
    with open(cellfile) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    print('cell_features shape:', cell_features.shape)

    # 读取药物分子的SMILES字符串
    compound_iso_smiles = []
    df_smiles = pd.read_csv(drug_smiles_file)
    compound_iso_smiles += list(df_smiles['smile'])
    compound_iso_smiles = set(compound_iso_smiles)

    # 生成摩根指纹
    print("开始生成摩根指纹...")
    fingerprint_dict = {}
    for smile in compound_iso_smiles:
        fp = generate_morgan_fingerprint(smile)
        fingerprint_dict[smile] = fp.tolist()

    os.makedirs('./data/processed/', exist_ok=True)
    with open('./data/processed/morgan_fingerprints.json', 'w') as f:
        json.dump(fingerprint_dict, f)
    print(f"摩根指纹生成完成！共处理 {len(fingerprint_dict)} 个分子")

    # 构建smile_graph
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    # 读取数据文件
    df = pd.read_csv(datafile)
    drug1 = np.asarray(list(df['drug1']))
    drug2 = np.asarray(list(df['drug2']))
    cell = np.asarray(list(df['cell']))
    label = np.asarray(list(df['label']))

    print(f'数据加载完成:  {len(drug1)} 条记录')

    return drug1, drug2, cell, label, smile_graph, cell_features, fingerprint_dict


# ============ 训练和预测函数 ============
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples... '.format(len(drug1_loader_train.dataset)))
    model.train()

    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).float().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        if output.dim() > 1 and output.size(1) == 2:
            output = F.softmax(output, dim=1)[:, 1]
        else:
            output = output.squeeze()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * len(data1.x),
                                                                            len(drug1_loader_train.dataset),
                                                                            100. * batch_idx / len(drug1_loader_train),
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
            if output.dim() > 1 and output.size(1) == 2:
                ys = F.softmax(output, dim=1).to('cpu').data.numpy()
                predicted_labels = list(map(lambda x: np.argmax(x), ys))
                predicted_scores = list(map(lambda x: x[1], ys))
            else:
                ys = output.to('cpu').data.numpy()
                predicted_labels = list(map(lambda x: int(x > 0.5), ys))
                predicted_scores = list(map(lambda x: x, ys))
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


# ============ 主程序 ============
# CPU or GPU
if torch.cuda.is_available():
    device = torch.device('cuda:' + str(device_num))
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!! !')

TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LR = 0.001
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs:  ', NUM_EPOCHS)

cellfile = 'data/cell_features.csv'
drug_smiles_file = 'data/smiles.csv'

if leave_type == 'leave_drug':
    train_datafile = ['data/leave_drug/leave_d00.csv',
                      'data/leave_drug/leave_d11.csv',
                      'data/leave_drug/leave_d22.csv',
                      'data/leave_drug/leave_d33.csv',
                      'data/leave_drug/leave_d44.csv']
    train_pt_dataset = ['leave_drug_d00', 'leave_drug_d11', 'leave_drug_d22', 'leave_drug_d33', 'leave_drug_d44']
    test_datafile = ['data/leave_drug/d00.csv',
                     'data/leave_drug/d11.csv',
                     'data/leave_drug/d22.csv',
                     'data/leave_drug/d33.csv',
                     'data/leave_drug/d44.csv']
    test_pt_result_dataset = ['drug_d00', 'drug_d11', 'drug_d22', 'drug_d33', 'drug_d44']
    fold_num = 5
elif leave_type == 'leave_comb':
    train_datafile = ['data/leave_comb/leave_c00.csv',
                      'data/leave_comb/leave_c11.csv',
                      'data/leave_comb/leave_c22.csv',
                      'data/leave_comb/leave_c33.csv',
                      'data/leave_comb/leave_c44.csv']
    train_pt_dataset = ['leave_comb_c00', 'leave_comb_c11', 'leave_comb_c22', 'leave_comb_c33', 'leave_comb_c44']
    test_datafile = ['data/leave_comb/c00.csv',
                     'data/leave_comb/c11.csv',
                     'data/leave_comb/c22.csv',
                     'data/leave_comb/c33.csv',
                     'data/leave_comb/c44.csv']
    test_pt_result_dataset = ['comb_c00', 'comb_c11', 'comb_c22', 'comb_c33', 'comb_c44']
    fold_num = 5
elif leave_type == 'leave_cell':
    train_datafile = ['data/leave_cell/leave_breast.csv',
                      'data/leave_cell/leave_colon.csv',
                      'data/leave_cell/leave_lung.csv',
                      'data/leave_cell/leave_melanoma.csv',
                      'data/leave_cell/leave_ovarian.csv',
                      'data/leave_cell/leave_prostate.csv']
    train_pt_dataset = ['leave_cell_breast', 'leave_cell_colon', 'leave_cell_lung', 'leave_cell_melanoma',
                        'leave_cell_ovarian', 'leave_cell_prostate']
    test_datafile = ['data/leave_cell/breast.csv',
                     'data/leave_cell/colon.csv',
                     'data/leave_cell/lung.csv',
                     'data/leave_cell/melanoma.csv',
                     'data/leave_cell/ovarian.csv',
                     'data/leave_cell/prostate.csv']
    test_pt_result_dataset = ['cell_breast', 'cell_colon', 'cell_lung', 'cell_melanoma', 'cell_ovarian',
                              'cell_prostate']
    fold_num = 6
else:
    raise ValueError(f"Unknown leave_type: {leave_type}. Choose from:  leave_drug, leave_comb, leave_cell")

for i in range(fold_num):
    # 使用新的load_data函数，返回7个值（包含fingerprint_dict）
    train_drug1, train_drug2, train_cell, train_label, smile_graph, cell_features, fingerprint_dict = load_data(
        train_datafile[i], drug_smiles_file, cellfile
    )

    # 创建TestbedDataset时传入fingerprint_dict
    train_drug1_data = TestbedDataset(root='data', dataset=train_pt_dataset[i] + '_drug1', xd=train_drug1,
                                      xt=train_cell,
                                      y=train_label, smile_graph=smile_graph, xt_featrue=cell_features,
                                      fingerprint_dict=fingerprint_dict)
    train_drug2_data = TestbedDataset(root='data', dataset=train_pt_dataset[i] + '_drug2', xd=train_drug2,
                                      xt=train_cell,
                                      y=train_label, smile_graph=smile_graph, xt_featrue=cell_features,
                                      fingerprint_dict=fingerprint_dict)
    print('train_drug1_data[0]', train_drug1_data[0])
    lenth = len(train_drug1_data)
    random_num = random.sample(range(0, lenth), lenth)
    drug1_data = train_drug1_data[random_num]
    drug2_data = train_drug2_data[random_num]

    # 测试集也需要加载fingerprint_dict
    test_drug1, test_drug2, test_cell, test_label, smile_graph, cell_features, fingerprint_dict = load_data(
        test_datafile[i], drug_smiles_file, cellfile
    )

    test_drug1_data = TestbedDataset(root='data', dataset=test_pt_result_dataset[i] + '_drug1', xd=test_drug1,
                                     xt=test_cell,
                                     y=test_label, smile_graph=smile_graph, xt_featrue=cell_features,
                                     fingerprint_dict=fingerprint_dict)
    test_drug2_data = TestbedDataset(root='data', dataset=test_pt_result_dataset[i] + '_drug2', xd=test_drug2,
                                     xt=test_cell,
                                     y=test_label, smile_graph=smile_graph, xt_featrue=cell_features,
                                     fingerprint_dict=fingerprint_dict)
    lenth = len(test_drug1_data)


    drug1_loader_train = DataLoader(drug1_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None, drop_last=True)
    drug2_loader_train = DataLoader(drug2_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None, drop_last=True)


    drug1_loader_test = DataLoader(test_drug1_data, batch_size=TEST_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(test_drug2_data, batch_size=TEST_BATCH_SIZE, shuffle=None)

    model = modeling(n_output=2, dropout=dropout_rate).to(device)

    loss_fn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H-%M-%S")

    folder_path = './result/' + result_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_AUCs = folder_path + '/' + result_name + '_' + str(i) + '--AUCs--' + test_pt_result_dataset[
        i] + '_' + time_str + '.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
        # T is correct label
        # S is predict score
        # Y is predict label

        # compute performance
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall_val = recall_score(T, Y)

        # save data
        if best_auc < AUC:
            best_auc = AUC
            AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA]
            save_AUCs(AUCs, file_AUCs)
        print('best_auc', best_auc)
    save_AUCs("best_auc:" + str(best_auc), file_AUCs)