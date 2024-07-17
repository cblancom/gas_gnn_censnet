import os
import shutil


final_path = './results/'
number_file = 1201
folders = sorted(os.listdir('.'))
for folder in folders:
    if folder.startswith('gen'):
        files = sorted(os.listdir('./'+folder+'/MPCC/'))
        for file in files:
            shutil.move('./'+folder+'/MPCC/'+file,
                        final_path+'gnn_sample_'+str(number_file)+'.pkl')

            number_file += 1
