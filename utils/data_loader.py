import numpy as np
from spektral.utils.convolution import gcn_filter


class data_loader:
    def __init__(self, folder_path, method_folder, method):
        self.folder_path = folder_path
        self.method_folder = method_folder
        self.method = method
        self.create_padding_data()

    def create_padding_data(
        self,
    ):

        X_train_list, X_test_list, X_val_list = [], [], []
        An_train_list, An_test_list, An_val_list = [], [], []
        Ae_train_list, Ae_test_list, Ae_val_list = [], [], []
        T_train_list, T_test_list, T_val_list = [], [], []
        Z_train_list, Z_test_list, Z_val_list = [], [], []
        fw_train_list, fw_test_list, fw_val_list = [], [], []
        p_train_list, p_test_list, p_val_list = [], [], []
        ye_train_list, ye_test_list, ye_val_list = [], [], []
        IR_train_list, IR_test_list, IR_val_list = [], [], []

        cases = self.method
        for c in cases:
            (
                X_train,
                X_test,
                X_val,
                An_train,
                An_test,
                An_val,
                Ae_train,
                Ae_test,
                Ae_val,
                T_train,
                T_test,
                T_val,
                Z_train,
                Z_test,
                Z_val,
                fw_train,
                fw_test,
                fw_val,
                p_train,
                p_test,
                p_val,
                ye_train,
                ye_test,
                ye_val,
                IR_train,
                IR_test,
                IR_val,
            ) = self.load_data(self.folder_path, self.method_folder, c)

            # Add the data to the corresponding lists
            X_train_list.append(X_train)  # check - feat
            X_test_list.append(X_test)  # check - feat
            X_val_list.append(X_val)  # check - feat

            An_train_list.append(An_train)  # check - topology
            An_test_list.append(An_test)  # check - topology
            An_val_list.append(An_val)  # check - topology

            Ae_train_list.append(Ae_train)  # check - topology
            Ae_test_list.append(Ae_test)  # check - topology
            Ae_val_list.append(Ae_val)  # check - topology

            T_train_list.append(T_train)  # check - incidence
            T_test_list.append(T_test)  # check - incidence
            T_val_list.append(T_val)  # check - incidence

            Z_train_list.append(Z_train)  # check - feat
            Z_test_list.append(Z_test)  # check - feat
            Z_val_list.append(Z_val)  # check - feat

            fw_train_list.append(fw_train)  # check - NE_out
            fw_test_list.append(fw_test)  # check - NE_out
            fw_val_list.append(fw_val)  # check - NE_out

            p_train_list.append(p_train)  # check - NE_out
            p_test_list.append(p_test)  # check - NE_out
            p_val_list.append(p_val)  # check - NE_out

            ye_train_list.append(ye_train)  # check - NE_out
            ye_test_list.append(ye_test)  # check - NE_out
            ye_val_list.append(ye_val)  # check - NE_out

            IR_train_list.append(IR_train)  # check - incidence
            IR_test_list.append(IR_test)  # check - incidence
            IR_val_list.append(IR_val)  # check - incidence

        self.X_train_padding = self.build_padding_array(X_train_list, "feat")
        self.X_test_padding = self.build_padding_array(X_test_list, "feat")
        self.X_val_padding = self.build_padding_array(X_val_list, "feat")

        self.An_train_padding = self.build_padding_array(An_train_list, "topology")
        self.An_test_padding = self.build_padding_array(An_test_list, "topology")
        self.An_val_padding = self.build_padding_array(An_val_list, "topology")

        self.Ae_train_padding = self.build_padding_array(Ae_train_list, "topology")
        self.Ae_test_padding = self.build_padding_array(Ae_test_list, "topology")
        self.Ae_val_padding = self.build_padding_array(Ae_val_list, "topology")

        self.T_train_padding = self.build_padding_array(T_train_list, "incidence")
        self.T_test_padding = self.build_padding_array(T_test_list, "incidence")
        self.T_val_padding = self.build_padding_array(T_val_list, "incidence")

        self.Z_train_padding = self.build_padding_array(Z_train_list, "feat")
        self.Z_test_padding = self.build_padding_array(Z_test_list, "feat")
        self.Z_val_padding = self.build_padding_array(Z_val_list, "feat")

        self.fw_train_padding = self.build_padding_array(fw_train_list, "NE_out")
        self.fw_test_padding = self.build_padding_array(fw_test_list, "NE_out")
        self.fw_val_padding = self.build_padding_array(fw_val_list, "NE_out")

        self.p_train_padding = self.build_padding_array(p_train_list, "NE_out")
        self.p_test_padding = self.build_padding_array(p_test_list, "NE_out")
        self.p_val_padding = self.build_padding_array(p_val_list, "NE_out")

        self.ye_train_padding = self.build_padding_array(ye_train_list, "NE_out")
        self.ye_test_padding = self.build_padding_array(ye_test_list, "NE_out")
        self.ye_val_padding = self.build_padding_array(ye_val_list, "NE_out")

        self.IR_train_padding = self.build_padding_array(IR_train_list, "incidence")
        self.IR_test_padding = self.build_padding_array(IR_test_list, "incidence")
        self.IR_val_padding = self.build_padding_array(IR_val_list, "incidence")

    def load_data(self, folder_path, method_folder, method):
        X_train = np.load(
            folder_path + method_folder[method] + "X_train.npy", allow_pickle=True
        )
        An_train = np.load(
            folder_path + method_folder[method] + "An_train.npy", allow_pickle=True
        )
        Ae_train = np.load(
            folder_path + method_folder[method] + "Ae_train.npy", allow_pickle=True
        )
        T_train = np.load(
            folder_path + method_folder[method] + "T_train.npy", allow_pickle=True
        )
        Z_train = np.load(
            folder_path + method_folder[method] + "Z_train.npy", allow_pickle=True
        )
        fw_train = np.load(
            folder_path + method_folder[method] + "fw_train.npy", allow_pickle=True
        )
        p_train = np.load(
            folder_path + method_folder[method] + "p_train.npy", allow_pickle=True
        )
        ye_train = np.load(
            folder_path + method_folder[method] + "ye_train.npy", allow_pickle=True
        )
        IR_train = np.load(
            folder_path + method_folder[method] + "IR_train.npy", allow_pickle=True
        )

        An_train = gcn_filter(An_train)
        Ae_train = gcn_filter(Ae_train)

        X_test = np.load(
            folder_path + method_folder[method] + "X_test.npy", allow_pickle=True
        )
        An_test = np.load(
            folder_path + method_folder[method] + "An_test.npy", allow_pickle=True
        )
        Ae_test = np.load(
            folder_path + method_folder[method] + "Ae_test.npy", allow_pickle=True
        )
        T_test = np.load(
            folder_path + method_folder[method] + "T_test.npy", allow_pickle=True
        )
        Z_test = np.load(
            folder_path + method_folder[method] + "Z_test.npy", allow_pickle=True
        )
        fw_test = np.load(
            folder_path + method_folder[method] + "fw_test.npy", allow_pickle=True
        )
        p_test = np.load(
            folder_path + method_folder[method] + "p_test.npy", allow_pickle=True
        )
        ye_test = np.load(
            folder_path + method_folder[method] + "ye_test.npy", allow_pickle=True
        )
        IR_test = np.load(
            folder_path + method_folder[method] + "IR_test.npy", allow_pickle=True
        )

        An_test = gcn_filter(An_test)
        Ae_test = gcn_filter(Ae_test)

        X_val = np.load(
            folder_path + method_folder[method] + "X_val.npy", allow_pickle=True
        )
        An_val = np.load(
            folder_path + method_folder[method] + "An_val.npy", allow_pickle=True
        )
        Ae_val = np.load(
            folder_path + method_folder[method] + "Ae_val.npy", allow_pickle=True
        )
        T_val = np.load(
            folder_path + method_folder[method] + "T_val.npy", allow_pickle=True
        )
        Z_val = np.load(
            folder_path + method_folder[method] + "Z_val.npy", allow_pickle=True
        )
        fw_val = np.load(
            folder_path + method_folder[method] + "fw_val.npy", allow_pickle=True
        )
        p_val = np.load(
            folder_path + method_folder[method] + "p_val.npy", allow_pickle=True
        )
        ye_val = np.load(
            folder_path + method_folder[method] + "ye_val.npy", allow_pickle=True
        )
        IR_val = np.load(
            folder_path + method_folder[method] + "IR_val.npy", allow_pickle=True
        )

        # IR_val = 1.0*np.abs(IR_val!=0)
        An_val = gcn_filter(An_val)
        Ae_val = gcn_filter(Ae_val)

        return_files = (
            X_train,
            X_test,
            X_val,
            An_train,
            An_test,
            An_val,
            Ae_train,
            Ae_test,
            Ae_val,
            T_train,
            T_test,
            T_val,
            Z_train,
            Z_test,
            Z_val,
            fw_train,
            fw_test,
            fw_val,
            p_train,
            p_test,
            p_val,
            ye_train,
            ye_test,
            ye_val,
            IR_train,
            IR_test,
            IR_val,
        )

        return return_files

    def build_padding_array(self, arrays, mode):
        arrays = np.array(arrays, dtype=object)

        if mode == "feat":
            F = arrays[0].shape[2]
            if not np.all([arr.shape[2] == F for arr in arrays]):
                raise ValueError("All arrays must have the same value for F")
            NE_max = max(arr.shape[1] for arr in arrays)
            T_p = sum(arr.shape[0] for arr in arrays)
            result_array = np.zeros((T_p, NE_max, F))

        elif mode == "topology":
            E1_max = max(arr.shape[1] for arr in arrays)
            E2_max = max(arr.shape[2] for arr in arrays)
            if E1_max != E2_max:
                raise ValueError("All arrays must have the same value for E")
            T_p = sum(arr.shape[0] for arr in arrays)
            result_array = np.zeros((T_p, E1_max, E2_max))

        elif mode == "incidence":
            N_max = max(arr.shape[1] for arr in arrays)
            E_max = max(arr.shape[2] for arr in arrays)
            T_p = sum(arr.shape[0] for arr in arrays)
            result_array = np.zeros((T_p, N_max, E_max))

        elif mode == "NE_out":
            N_max = max(arr.shape[1] for arr in arrays)
            T_p = sum(arr.shape[0] for arr in arrays)
            result_array = np.zeros((T_p, N_max))

        else:
            raise ValueError("Unknown {} mode".format(mode))

        current_T_index = 0
        for arr in arrays:
            if mode == "NE_out":
                T = arr.shape[0]
                N = arr.shape[1]
                result_array[current_T_index : current_T_index + T, :N] = arr
            else:
                T, dim1, dim2 = arr.shape
                result_array[current_T_index : current_T_index + T, :dim1, :dim2] = arr
            current_T_index += T

        return result_array
