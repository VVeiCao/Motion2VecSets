import os
import subprocess
from multiprocessing import Pool, cpu_count

# 输入和输出目录
input_dir = "DFAUST_mesh/data"
output_dir = "DFAUST_mesh/subdivision_shape"

# MeshLab 脚本的路径
mlx_script = "DFAUST_mesh/preprocessing/05_3.mlx"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 初始化一个空的文件列表变量
obj_files = []

# 遍历 data 文件夹下的所有子文件夹
for subdir, _, _ in sorted(os.walk(input_dir)):
    # 在每个子文件夹中查找 .obj 文件
    for item in os.listdir(subdir):
        if item.endswith('.obj'):
            obj_files.append(os.path.join(subdir, item))

def process_file(input_file):
    filename = os.path.basename(input_file)
    filename_noext = os.path.splitext(filename)[0]
    subdir_name = os.path.basename(os.path.dirname(input_file))
    
    output_subdir = os.path.join(output_dir, subdir_name)
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f"{filename_noext}.obj")
    
    # 检查输出文件是否已存在，如果存在则跳过
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping.")
        return
    
    command = ["meshlabserver", "-i", input_file, "-o", output_file, "-s", mlx_script]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode == 0:
        print(f"Successfully processed {input_file} -> {output_file}")
    else:
        print(f"Failed to process {input_file} -> {output_file}")
        print(result.stderr.decode('utf-8'))

if __name__ == "__main__":
    # 使用所有可用的CPU核心
    with Pool(cpu_count()) as p:
        p.map(process_file, obj_files)
