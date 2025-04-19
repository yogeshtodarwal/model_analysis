import os
import subprocess

results_dir = "/mnt/home/yogtod@ad.cmm.se/old-projects/patdbpredict/results"
#results_dir = "/mnt/home/yogtod@ad.cmm.se/projects/patdbpredict/yogesh/patdbpredict/results"
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".pklz") and "_" in filename:
                hash_id = filename.split("_")[0]
                cmd = ["python", "analysis.py", folder_path, hash_id, "."]
                subprocess.run(cmd)