import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt

# import spektral
import io
import pandas as pd
import dill
import os

# from sklearn.model_selection import train_test_split

# import gekko

path = "/content/drive/Shareddrives/Proyecto Gas Natural 2023/Cristian/GNN/data_generation/"

options = ["dummy_wey", "col_wey", "dummy_lineal", "col_lineal"]
select_option = options[3]

if select_option == "dummy_wey":
    folder_path = path + "/dummy_random/MPCC/"
elif select_option == "col_wey":
    folder_path = path + "/col_random/MPCC/"
elif select_option == "dummy_lineal":
    folder_path = path + "/No_weymouth/Dummy/MPCC/"
elif select_option == "col_lineal":
    folder_path = path + "/No_weymouth/Col/MPCC/"


# folder_path = '/content/drive/Shareddrives/Proyecto Gas Natural 2023/Cristian/GNN/data_generation/final_files/'
# folder_path = '/content/drive/Shareddrives/Proyecto Gas Natural 2023/Cristian/GNN/data_generation/dummy_random/MPCC/'
# folder_path = '/content/drive/Shareddrives/Proyecto Gas Natural 2023/Cristian/GNN/data_generation/col_random/MPCC/'
folder_path = "./results/"
folder_path = "../manipulate_results/Results/col_weymouth/MPCC/"
files = os.listdir(folder_path)
files = [os.path.join(folder_path, f) for f in files if f.endswith(".pkl")]


def data_loader(files_list):
    extended_list = []
    MPCC_list = []
    SOC_list = []
    Taylor_list = []
    All_list = []
    for filename in tqdm(files_list):
        with open(filename, "rb") as f:
            # try:
            data = dill.load(f)
            # except:
            #     pass
        wey_type = data.weymouth_type
        All_list.append(data)
        if data.weymouth_type == "MPCC":
            MPCC_list.append(data)
        elif data.weymouth_type == "SOC":
            SOC_list.append(data)
        elif data.weymouth_type == "Taylor":
            Taylor_list.append(data)
    return MPCC_list, SOC_list, Taylor_list, All_list


MPCC_list, SOC_list, Taylor_list, All_list = data_loader(files)


# from logging import FileHandler
def get_values(X):
    result = []
    for i in X.flatten():
        try:
            result.append(i[0])
        except:
            result.append(i)
    return np.array(result).reshape(X.shape)


def extend_pkl_files(folder_path):
    extended_list = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as f:
                # data_list = dill.load(f)
                # extended_list.extend(data_list)
                extended_list.append(dill.load(f))

    return extended_list


def input_node_features(p):
    well_data = np.zeros((p.N, 2))
    well_data[p.Minc[:, : p.W].sum(axis=1) > 0.5, 0] = p.well.Imin.values
    well_data[p.Minc[:, : p.W].sum(axis=1) > 0.5, 1] = p.well.Imax.values
    #   well_data[p.Minc[:,0]>0.5,0] = p.well.Imin.values
    #   well_data[p.Minc[:,0]>0.5,1] = p.well.Imax.values

    user_data = p.loads_gas.reshape(-1, 1)
    user_data[6] += 0.01
    node_data = np.hstack(
        (
            p.node_info["Pmin"].values.reshape(-1, 1),
            p.node_info["Pmax"].values.reshape(-1, 1),
        )
    )
    node_features = np.hstack((well_data, user_data, node_data))

    return node_features


def input_edge_features(p):
    pipe_data = np.zeros((p.P, 4))
    pipe_data[:, 0] = p.Kij.reshape(
        -1,
    )
    pipe_data[:, 2] = -p.pipe.Fg_max.values.reshape(
        -1,
    )
    pipe_data[:, 3] = p.pipe.Fg_max.values.reshape(
        -1,
    )
    # print(pipe_data)

    comp_data = np.zeros((p.C, 4))
    comp_data[:, 1] = p.max_ratio.values.reshape(
        -1,
    )
    comp_data[:, 3] = p.comp.fmaxc.values.reshape(
        -1,
    )
    # print(comp_data)

    X = np.block(
        [
            [
                pipe_data,
            ],
            [comp_data],
        ]
    )
    return X


def output_node_features(p):
    # f_well = get_values(p.X_gas[0,:p.W])
    f_well = np.zeros((p.N,))
    f_well[p.Minc[:, : p.W].sum(1) > 0.5] = get_values(p.X_gas[0, : p.W]).reshape(
        -1,
    )
    press = get_values(p.press)[0]
    y = np.hstack(
        [
            f_well.reshape(-1, 1),
            press.reshape(-1, 1),
        ]
    )
    return f_well.reshape(-1, 1), press.reshape(-1, 1)


def output_edge_features(p):
    f_plus = get_values(p.X_gas[0, p.W : p.W + p.P])
    f_minus = get_values(p.X_gas[0, p.W + p.P : p.W + 2 * p.P])
    f_pipe = f_plus + f_minus
    f_comp = get_values(p.X_gas[0, p.W + 2 * p.P : p.W + 2 * p.P + p.C])
    f = np.hstack([f_pipe, f_comp])

    return f.reshape(-1, 1)


def incidence_matrix(p):
    A = np.hstack(
        [p.Minc[:, p.W : p.W + p.P], p.Minc[:, p.W + 2 * p.P : p.W + 2 * p.P + p.C]]
    )

    return np.abs(A)


def incidence_matrix_real(p):
    A = np.hstack(
        [p.Minc[:, p.W : p.W + p.P], p.Minc[:, p.W + 2 * p.P : p.W + 2 * p.P + p.C]]
    )

    return A


def node_laplacian(A):
    # A = incidence_matrix(p)
    A_ = A @ A.T
    A_ -= np.diag(np.diag(A_))
    return spektral.utils.convolution.gcn_filter(A_)


def edge_laplacian(A):
    # A = incidence_matrix(p)
    A_ = A.T @ A
    A_ -= np.diag(np.diag(A_))
    return spektral.utils.convolution.gcn_filter(A_)
