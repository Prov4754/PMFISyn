"""
数据预分割脚本：生成留出药物、留出药物对、留出细胞系的训练/测试集
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold


def load_synergy_data(synergy_file):

    df = pd.read_csv(synergy_file)
    return df


def split_leave_drug(df, n_folds=5, output_dir='data/leave_drug', seed=42):

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # 获取所有唯一药物
    all_drugs = set(df['drug1'].unique()) | set(df['drug2'].unique())
    all_drugs = list(all_drugs)
    np.random.shuffle(all_drugs)

    # 将药物分成n_folds组
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    drug_array = np.array(all_drugs)

    for fold_idx, (train_drug_idx, test_drug_idx) in enumerate(kf.split(drug_array)):
        test_drugs = set(drug_array[test_drug_idx])

        # 测试集：包含留出药物的所有样本
        test_mask = df['drug1'].isin(test_drugs) | df['drug2'].isin(test_drugs)
        test_df = df[test_mask]
        train_df = df[~test_mask]

        # 保存
        train_df.to_csv(os.path.join(output_dir, f'leave_d{fold_idx}{fold_idx}. csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, f'd{fold_idx}{fold_idx}.csv'), index=False)

        print(
            f"Fold {fold_idx}:  Train samples={len(train_df)}, Test samples={len(test_df)}, Test drugs={len(test_drugs)}")


def split_leave_comb(df, n_folds=5, output_dir='data/leave_comb', seed=42):

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # 获取所有唯一药物对
    df['drug_pair'] = df.apply(lambda x: tuple(sorted([x['drug1'], x['drug2']])), axis=1)
    unique_pairs = df['drug_pair'].unique()
    np.random.shuffle(unique_pairs)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_pair_idx, test_pair_idx) in enumerate(kf.split(unique_pairs)):
        test_pairs = set(unique_pairs[test_pair_idx])

        test_mask = df['drug_pair'].isin(test_pairs)
        test_df = df[test_mask].drop(columns=['drug_pair'])
        train_df = df[~test_mask].drop(columns=['drug_pair'])

        train_df.to_csv(os.path.join(output_dir, f'leave_c{fold_idx}{fold_idx}.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, f'c{fold_idx}{fold_idx}.csv'), index=False)

        print(
            f"Fold {fold_idx}: Train samples={len(train_df)}, Test samples={len(test_df)}, Test pairs={len(test_pairs)}")


    if 'drug_pair' in df.columns:
        df.drop(columns=['drug_pair'], inplace=True)


def split_leave_cell(df, output_dir='data/leave_cell', seed=42):
    """
    留出细胞系分割：按细胞系类型分割（如breast, colon, lung等）
    """
    os.makedirs(output_dir, exist_ok=True)
    cell_type_mapping = {
        'A2058': 'melanoma', 'A375': 'melanoma', 'HT144': 'melanoma',
        'RPMI7951': 'melanoma', 'SKMEL30': 'melanoma',
        'A2780': 'ovarian', 'CAOV3': 'ovarian', 'ES2': 'ovarian',
        'OV90': 'ovarian', 'SKOV3': 'ovarian',
        'HCT116': 'colon', 'HT29': 'colon', 'LOVO': 'colon',
        'RKO': 'colon', 'SW620': 'colon', 'SW837': 'colon',
        'A427': 'lung', 'NCIH1650': 'lung', 'NCIH2122': 'lung',
        'NCIH23': 'lung', 'NCIH460': 'lung', 'NCIH520': 'lung', 'SKMES1': 'lung',
        'KPL1': 'breast', 'MDAMB436': 'breast', 'T47D': 'breast',
        'LNCAP': 'prostate',
        'MSTO': 'mesothelioma'
    }

    # 按癌症类型分组
    cancer_types = ['breast', 'colon', 'lung', 'melanoma', 'ovarian', 'prostate']

    for cancer_type in cancer_types:
        cells_of_type = [cell for cell, ctype in cell_type_mapping.items() if ctype == cancer_type]

        if len(cells_of_type) == 0:
            continue

        test_mask = df['cell'].isin(cells_of_type)
        test_df = df[test_mask]
        train_df = df[~test_mask]

        if len(test_df) > 0:
            train_df.to_csv(os.path.join(output_dir, f'leave_{cancer_type}.csv'), index=False)
            test_df.to_csv(os.path.join(output_dir, f'{cancer_type}.csv'), index=False)

            print(
                f"{cancer_type}: Train samples={len(train_df)}, Test samples={len(test_df)}, Test cells={cells_of_type}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Split data for leave-out experiments')
    parser.add_argument('--synergy_file', type=str, default='data/labels.csv',
                        help='Path to synergy data file')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for drug/comb split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print("Loading synergy data...")
    df = load_synergy_data(args.synergy_file)

    print("\n=== Splitting for Leave-Drug ===")
    split_leave_drug(df, n_folds=args.n_folds, seed=args.seed)

    print("\n=== Splitting for Leave-Comb ===")
    split_leave_comb(df, n_folds=args.n_folds, seed=args.seed)

    print("\n=== Splitting for Leave-Cell ===")
    split_leave_cell(df, seed=args.seed)

    print("\nData splitting completed!")