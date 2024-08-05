import numpy as np
import pandas as pd
import gekko
import dill
import argparse
from IS_montecarlo_data_generator import InterconnectedEnergySystemOptimizer

parser = argparse.ArgumentParser()

parser.add_argument("argument_gas", type=str)
parser.add_argument("argument_power", type=str)
parser.add_argument("argument_random", type=float)
parser.add_argument("argument_savefolder", type=str)
parser.add_argument("argument_weymouth", type=str)

args = parser.parse_args()


def modify_values(df, column_list, pct_modification=args.argument_random):
    """
    Modifies values in a DataFrame column with random noise.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        column_name (str): The name of the column to modify.

    Returns:
        pandas.DataFrame: The DataFrame with modified values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_modified = df.copy()

    # Generate random noise between -pct_modification and pct_modification
    noise = np.random.uniform(
        low=-pct_modification, high=pct_modification, size=len(df)
    )

    # Add noise to the original values
    for col in column_list:
        noise = np.random.uniform(
            low=-pct_modification, high=pct_modification, size=len(df)
        )
        df_modified[col] = df[col] * (1 + noise)

    return df_modified


# file_path = '/content/ng_case8_gnn.xlsx'
def randomize_values(file_path, savepath):
    node_info = modify_values(
        pd.read_excel(file_path, sheet_name="node.info"), ["Pmax", "Pmin"]
    )

    node_dem = modify_values(
        pd.read_excel(file_path, sheet_name="node.dem"), ["Res", "Ind"]
    )

    node_demcost = pd.read_excel(file_path, sheet_name="node.demcost")

    node_deminitial = pd.read_excel(file_path, sheet_name="node.deminitial")

    well_df = modify_values(
        pd.read_excel(file_path, sheet_name="well"), ["Imax", "Imin"]
    )

    pipe_df = modify_values(
        pd.read_excel(file_path, sheet_name="pipe"), ["Kij", "Fg_max", "Fg_min"]
    )

    comp_df = modify_values(
        pd.read_excel(file_path, sheet_name="comp"), ["ratio", "fmaxc"]
    )

    sto_df = pd.read_excel(file_path, sheet_name="sto")
    with pd.ExcelWriter(savepath, engine="xlsxwriter") as writer:
        node_info.to_excel(writer, sheet_name="node.info", index=False)
        node_dem.to_excel(writer, sheet_name="node.dem", index=False)
        node_demcost.to_excel(writer, sheet_name="node.demcost", index=False)
        node_deminitial.to_excel(writer, sheet_name="node.deminitial", index=False)

        well_df.to_excel(writer, sheet_name="well", index=False)
        pipe_df.to_excel(writer, sheet_name="pipe", index=False)
        comp_df.to_excel(writer, sheet_name="comp", index=False)
        sto_df.to_excel(writer, sheet_name="sto", index=False)

    return


# 118IEEE - 48 node gas

trials = 1000
# /home/usuario/Work/NGP/GNN_gen_data/results

gas_file = args.argument_gas
power_file = args.argument_power
savepath_folder = args.argument_savefolder


for i in range(1, trials):
    try:
        savepath = (
            "./"
            + savepath_folder
            + "/results_random_df/ng_case_col_random_"
            + str(i)
            + ".xlsx"
        )
        randomize_values(gas_file, savepath)
        path_power = power_file
        name = "gnn_samples"

        df_power = pd.ExcelFile(path_power)
        df_gas = pd.ExcelFile(savepath)

        print("flag")

        p = InterconnectedEnergySystemOptimizer(
            data_gas=df_gas,
            data_power=df_power,
            T=1,
            weymouth_type=args.argument_weymouth,
        )
        if p.m.options.objfcnval > 1:
            file_ = open(
                "./"
                + savepath_folder
                + "/results_pkl/"
                + "gnn_sample_"
                + str(i)
                + ".pkl",
                "wb",
            )
            dill.dump(p, file_)
            file_.close()
            print("The file was saved")
        else:
            print("Cannont save")

    except Exception as e:
        print(f"An error occurred: {e}")
