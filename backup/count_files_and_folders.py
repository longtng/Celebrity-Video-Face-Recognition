import os
from distutils.dir_util import copy_tree
import numpy as np 
import random
import shutil
def count_files_and_folders(path):
    files = folders = 0
    for _, dirnames, filenames in os.walk(path):
        files += len(filenames)
        folders += len(dirnames)
    print("{:,} files, {:,} folders".format(files, folders)) 
def subset_contain_folders_having_at_least_number_images(path, number):
    folders = 0
    files = 0
    for _, dirnames_p, filenames_p in os.walk(path):
        for i in dirnames_p:
            path1 = os.path.join(path,i)
            for _, dirnames_p1, filenames_p1 in os.walk(path1):
                if(len(filenames_p1) > number):
                    folders +=1
                    files +=len(filenames_p1)
                    destination = os.path.join('data/subset/',i)
                    os.makedirs(destination)
                    copy_tree(path1,destination) 
    print("{:,} files, {:,} folders".format(files, folders))
def split_subset_from_1_directory(path,number,des1,des2):
    for _, dirnames_p, filenames_p in os.walk(path):
        for i in dirnames_p:
            path1 = os.path.join(path, i)
            for _, dirnames_p1, filenames_p1 in os.walk(path1):
                idx1 = random.sample(range(0,len(filenames_p1)),number)
                idx2 = []
                for k in np.arange(len(filenames_p1)):
                    if (k not in idx1):
                        idx2.append(k)
                destination1 = os.path.join(des1,i)
                destination2 = os.path.join(des2,i)
                os.makedirs(destination1)
                os.makedirs(destination2)
                for j in idx1:
                    path2 = os.path.join(path1, filenames_p1[j])
                    shutil.copy(path2,destination1)
                for m in idx2:
                    path3 = os.path.join(path1, filenames_p1[m])
                    shutil.copy(path3,destination2)
