# compare the number of files in subfolders with same name in two different folders

import os


def compare_folders(folder1, folder2):
    for root, dirs, files in os.walk(folder1):
      for dir in sorted(dirs):
        print(dir)
        print(len(os.listdir(os.path.join(folder1, dir))))
        print(len(os.listdir(os.path.join(folder2, dir, "pcl_seq"))))
        print("")



if __name__ == '__main__':
    folder1 ="DFAUST_mesh/data"
    folder2 ="DFAUST_mesh/data_processed_shape"
    compare_folders(folder1, folder2)