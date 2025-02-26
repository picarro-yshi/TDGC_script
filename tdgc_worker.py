# TDGC analysis code, worker thread
# the main task is divided into 4 piece, 3 workers run parallel with the main

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")

from constants import DATA_FOLDER, STEP_RUN, SPLIT_TASK
from step3 import notebook3_part
from step4 import notebook4_part
from step5a import notebook5a_part
from step5b import notebook5b_part
from step9 import notebook9_part

# wait until all sub-tasks are finished, then go to next task
def wait(stepname):
    """args:
      stepname: string, like "3", "5A"
      returns none"""

    while 1:
        with open(p, "r") as f:
            log = f.read()
            if ("Step %s finished" % stepname) in log:
                break
        time.sleep(60)


if __name__ == "__main__":    
    # p = os.path.join(DATA_FOLDER, "log.txt")
    p = "log.txt"
    n = int(sys.argv[1])  # sub-task 2, 3, 4 (1 is with the main thread)

    if STEP_RUN["3"]:
        wait("2")
        notebook3_part(n)
        with open(p, "a") as f:
            f.write("...3-%s/%s" % (n, SPLIT_TASK))

    if STEP_RUN["4"]:
        wait("3")
        notebook4_part(n)
        with open(p, "a") as f:
            f.write("...4-%s/%s" % (n, SPLIT_TASK))

    if STEP_RUN["5A"]:
        wait("4")
        notebook5a_part(n)
        with open(p, "a") as f:
            f.write("...5A-%s/%s" % (n, SPLIT_TASK))

    if STEP_RUN["5B"]:
        wait("5A")
        notebook5b_part(n)
        with open(p, "a") as f:
            f.write("...5B-%s/%s" % (n, SPLIT_TASK))

    if STEP_RUN["9"]:
        wait("8")
        notebook9_part(n)
        with open(p, "a") as f:
            f.write("...9-%s/%s" % (n, SPLIT_TASK))


# @author: Yilin Shi | 2025.2.14
# shiyilin890@gmail.com
# Bog the Fat Crocodile vvvvvvv
#                       ^^^^^^^
