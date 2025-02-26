# TDGC analysis code, parameters

import yaml
from pathlib import Path

conf = yaml.safe_load(Path("./analysis_config.yaml").read_text())  # YAML to dictionary
DATA_FOLDER = conf["paths"]["exp_folder"]
SPLIT_TASK = 2  # 2, 3, 4. split task to n parts, open n terminals to run parallel.

# step name, 1 is run, 0 is skip
STEP_RUN = {
    "1": 0,
    "2": 0,
    "3": 1,
    "4": 1,
    "5A": 1,
    "5B": 1,
    "6": 1,
    "7": 1,
    "8": 1,
    "9": 1,
    "10": 1,
    "11": 1,
    "14": 1,
}





# @author: Yilin Shi | 2025.2.14
# shiyilin890@gmail.com
# Bog the Fat Crocodile vvvvvvv
#                       ^^^^^^^
