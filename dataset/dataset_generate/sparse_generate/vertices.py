# check the vertices number of each mesh in the folder DFAUST_mesh/data_subdivision

import os
import numpy as np
import trimesh
import glob


# list all the files in the folder
path = 'DFAUST_mesh/subdivision_shape'
files = os.listdir(path)
files.sort()

# for obj in each folder in files, check the vertices number
for file in files:
    file_path = os.path.join(path, file)
    objs = os.listdir(file_path)
    objs.sort()
    for obj in objs[:1]:
      obj_path = os.path.join(path, file, obj)
      # print(obj_path)
      mesh = trimesh.load(obj_path)
      # print the path
      print(obj_path)
      print(mesh.faces.shape[0])
      

