import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WARNINGS"] = "0"
os.environ["THREADPOOLCTL_DISABLE"] = "1"
os.environ["SCIKIT_LEARN_DATA"] = os.getcwd() # 增强对 sklearn 环境变量的隔离

import argparse
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from loss import *
from models import GMAE_MVC

from utils.dataloader import dataset_with_info
from utils import Logger
from utils.metric import compute_metric
from utils.plot import plot_metric



def train_one_epoch(args, model, mse_loss_fn, optimizer, view_num, nbr_idx, neg_idx, train_loader, device):
    model.train()
    for x, y, idx, pu in train_loader:
        optimizer.zero_grad()

        for v in range(view_num):
            x[v] = x[v].to(device)

        z_c, z_nc_list, hidden, recs, z_c_views = model(x)

        loss_rec = 0.0
        for v in range(view_num):
            loss_rec += mse_loss_fn(recs[v], x[v])

        loss_disentangle = advanced_disentangle_loss(z_c.detach(), z_nc_list, z_c_views)

        if args.do_contrast and nbr_idx is not None:
            loss_topo = Topological_contrast_loss(args, z_c, nbr_idx, neg_idx, idx)
        else:
            loss_topo = 0.0

        total_loss = loss_rec + args.lambda_dis * loss_disentangle + args.lambda_topo * loss_topo
        total_loss.backward()
        optimizer.step()


def evaluate_model(model, nc, view_num, test_loader, device):
    model.eval()
    with torch.no_grad():
        for x, y, idx, pu in test_loader:
            for v in range(view_num):
                x[v] = x[v].to(device)

            z_c, z_nc_list, hidden, recs, _ = model(x)
            z_c_norm = F.normalize(z_c, p=2, dim=-1).cpu().numpy()

            # 【已修复】加入防崩溃重试机制，免疫 Windows 下 threadpoolctl 的模块扫描异常
            max_retries = 3
            y_pred = None
            for attempt in range(max_retries):
                try:
                    y_pred = KMeans(n_clusters=nc, n_init=20, random_state=42).fit_predict(z_c_norm)
                    break
                except OSError as e:
                    if "GetModuleFileNameEx failed" in str(e) and attempt < max_retries - 1:
                        continue
                    else:
                        raise e

            label = np.array(y)
            ACC, NMI, Purity, ARI, F_score, Precision, Recall = compute_metric(label, y_pred)
            return ACC, NMI, Purity, ARI, F_score, Precision, Recall


def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_neighbors(dataset, ins_num, view_num, device):
    if ins_num <= 5:
        return None, None

    neighbors_num = max(2, min(int(ins_num / 4), ins_num - 1))
    pos_num = max(2, min(21, neighbors_num - 1))
    neg_num = max(1, int((neighbors_num - pos_num - 1) / 2))

    nbr_idx, neg_idx = [], []

    for v in range(view_num):
        X_np = np.array(dataset.features[0][v])
        nbrs_v = np.zeros((ins_num, pos_num - 1), dtype=np.int64)
        neg_v = np.zeros((ins_num, neg_num), dtype=np.int64)

        nbrs = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(X_np)
        dis, idx = nbrs.kneighbors(X_np)

        for i in range(ins_num):
            nbrs_v[i][:] = idx[i][1:pos_num]
            neg_v[i][:] = idx[i][-neg_num:]

        nbr_idx.append(torch.LongTensor(nbrs_v))
        neg_idx.append(torch.LongTensor(neg_v))

    nbr_idx = torch.cat(nbr_idx, dim=-1).to(device)
    neg_idx = torch.cat(neg_idx, dim=-1).to(device)
    return nbr_idx, neg_idx


# ===================================================================
# 3. 主程序入口
# ===================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topology-InfoMax Multi-View Clustering')
    parser.add_argument('--logs_path', default='1.logs_clustering', type=str, help='Path to save logs')
    parser.add_argument('--imgs_path', default='2.imgs_clustering', type=str, help='Path to save imgs')
    parser.add_argument('--folder_path', default='dataset', type=str, help='Dataset folder path')
    parser.add_argument('--do_plot', default=True, type=bool, help='Whether to plot the results')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use for training')

    parser.add_argument('--train_epoch', default=500, type=int, help='Number of training epochs')
    parser.add_argument('--eval_interval', default=10, type=int, help='Interval for evaluation')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dimensions')
    parser.add_argument('--lambda_dis', default=0.05, type=float, help='Lambda for disentangle loss')
    parser.add_argument('--lambda_topo', default=0.01, type=float, help='Lambda for Topological contrast loss')
    parser.add_argument('--pos_num', default=21, type=int, help='Positive sample number')
    parser.add_argument('--do_contrast', default=True, type=bool, help='Whether to use contrastive loss')

    args = parser.parse_args()
    seed_setting(args.seed)

    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)

    file_datasetInfo = f'{args.logs_path}/datasetInfo.csv'
    if not os.path.exists(file_datasetInfo):
        headers = ['Dataname', 'number of data', 'views', 'clusters', 'each view']
        with open(file_datasetInfo, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    file_names = os.listdir(args.folder_path)
    data_iter = 1
    for dataset_name in file_names:
        if dataset_name.endswith(".mat"):
            dataset_name_clean = dataset_name[:-4]
            print(
                f'----------------------------------{dataset_name_clean}[{data_iter}]----------------------------------')
            seed_setting(args.seed)
            try:
                logger = Logger.get_logger(__file__, dataset_name_clean, args.logs_path)
                dataset, ins_num, view_num, nc, input_dims, _ = dataset_with_info(
                    dataset_name_clean, file_datasetInfo, args.folder_path)

                model = GMAE_MVC(input_dims, view_num, args.feature_dim, h_dims=[500, 200], num_classes=nc).to(
                    args.device)
                mse_loss_fn = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                nbr_idx, neg_idx = prepare_neighbors(dataset, ins_num, view_num, args.device)
                acc_list, nmi_list, pur_list, ari_list = [], [], [], []

                test_loader = DataLoader(dataset, batch_size=ins_num, shuffle=False)
                train_loader = DataLoader(dataset, batch_size=ins_num, shuffle=False)

                for epoch in tqdm(range(args.train_epoch)):
                    train_one_epoch(args, model, mse_loss_fn, optimizer, view_num, nbr_idx, neg_idx, train_loader,
                                    args.device)

                    if (epoch + 1) % args.eval_interval == 0:
                        acc, nmi, pur, ari, _, _, _ = evaluate_model(model, nc, view_num, test_loader, args.device)
                        acc_list.append(acc)
                        nmi_list.append(nmi)
                        pur_list.append(pur)
                        ari_list.append(ari)

                        info = {"epoch": epoch + 1, "acc": acc, "nmi": nmi, "ari": ari, "pur": pur}
                        logger.info(str(info))

                if args.do_plot and len(acc_list) > 0:
                    plot_metric(acc_list, dataset_name_clean, 'acc', args.imgs_path)

            except Exception as e:
                print(f"Error processing dataset {dataset_name_clean}: {e}")
                import traceback

                traceback.print_exc()

            finally:
                if 'model' in locals():
                    del model
                if 'optimizer' in locals():
                    del optimizer
                torch.cuda.empty_cache()

        else:
            print(f'Non-MAT file skipped.')
        data_iter += 1