import os 

angles = [45, 90, 0, 30]

for angle in angles:
    setting = f"rotate{angle}"
    print(f"Generating depth for setting: {setting}")
    os.system(f"python generate_depth_dataset.py --angle {angle}")
    print(f"Done generating depth for setting: {setting}")
