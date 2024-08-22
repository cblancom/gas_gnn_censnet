import os
import shutil


final_path = "./results/"
number_file = 0
folders = sorted(os.listdir("."))
for folder in folders:
    if folder.startswith("gen"):
        files = sorted(os.listdir("./" + folder + "/results_pkl/"))
        for file in files:
            shutil.copy(
                "./" + folder + "/results_pkl/" + file,
                final_path + "gnn_sample_" + str(number_file) + ".pkl",
            )

            number_file += 1
