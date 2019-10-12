import shutil
import os
import numpy as np 
import random
def subset_folders_having_number_images(path,number,des):
    files = 0
    folders = 0
    for _, dirnames_p, filenames_p in os.walk(path):
        folders += len(dirnames_p)
        for i in dirnames_p:
            path1 = os.path.join(path, i)
            for _, dirnames_p1, filenames_p1 in os.walk(path1):
                if(len(filenames_p1) > number-1):
                    files +=1
                    idx = random.sample(range(0,len(filenames_p1)),number)
                    #destination = 'data\\subset1\\' + i
                    destination = os.path.join(des, i)
                    os.makedirs(destination)
                    for j in idx:
                        path2 = os.path.join(path1, filenames_p1[j])
                        shutil.copy(path2,destination)
    print("{:,} folders".format(files))

def split_subset(path,number1,number2,des1,des2):
    for _, dirnames_p, filenames_p in os.walk(path):
        for i in dirnames_p:
            path1 = os.path.join(path, i)
            for _, dirnames_p1, filenames_p1 in os.walk(path1):
                idx1 = random.sample(range(0,len(filenames_p1)),number1)
                idx2 = []
                for k in np.arange(number2):
                    if (k not in idx1):
                        idx2.append(k)
                destination1 = os.path.join(des1, i)
                destination2 = os.path.join(des2, i)
                os.makedirs(destination1)
                os.makedirs(destination2)
                for j in idx1:
                    path2 = os.path.join(path1, filenames_p1[j])
                    shutil.copy(path2,destination1)
                for m in idx2:
                    path3 = os.path.join(path1, filenames_p1[m])
                    shutil.copy(path3,destination2)
