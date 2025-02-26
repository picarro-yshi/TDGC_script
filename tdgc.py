# TDGC analysis code, main thread, last updated: 2025.2.24
# combine Chris's 10 notebook together
# how to use:
# 1. update 'analysis_config.yaml'
# 2. update 'constants.py' if needed
# 3. start this file, wait until step 1, 2 is finished, check if the chromatogram is ok, otherwise adjust yaml
# 'Misc_results/ChromatogramSummary.png'
# 4. open n terminals and run n workers:
# $ python tdgc_worker.py 2 (or 3, 4)
# the main task is divided into 4 pieces, 3 workers run parallel with the main


import time
import os
import shutil
import sys
print(sys.version)
import warnings
warnings.filterwarnings("ignore")

from constants import DATA_FOLDER, SPLIT_TASK, STEP_RUN
# notebooks
from utility import notebook1, notebook2, notebook6, notebook7
from step3 import notebook3_part
from step4 import notebook4_part
from step5a import notebook5a_part
from step5b import notebook5b_part
from utility2 import notebook8, notebook10, notebook11, notebook14
from step9 import notebook9_part


def print_time(t):
    """args:
      t: float/int, epoch time
      returns string, human understandable time"""

    if t <= 60:
        text = "%.2f s  " % t
    elif 60 < t <= 3600:
        text = "%.2f min" % (t / 60)
    else:
        text = "%.2f h  " % (t / 3600)
    return text


# synconize sub-tasks for long steps (3, 4, 5A, 5B, 9)
def synco(stepname, t0):
    """args:
      stepname: string, like "3", "5A"
      t0: float/int, epoch time
      returns none """

    while 1:
        with open(p, "r") as f:
            log = f.read()
            condition = (
                ("%s-1" % stepname in log)
                and ("%s-2" % stepname in log)
                and ("%s-3" % stepname in log)
                and ("%s-4" % stepname in log)
            )

            if condition:
                break
        time.sleep(60)

    text = print_time(time.time() - t0)
    info = "\nStep %s finished: %s \nTook: %s\n\n" % (stepname, time.ctime(), text)
    with open(p, "a") as f:
        f.write(info)
    time.sleep(1)


if __name__ == "__main__":    
    t_start = time.time()
    pp = os.path.join(DATA_FOLDER, "log.txt")
    p = "log.txt"
    with open(p, "w") as f:
        f.write("*** TDGC Analysis Log ***\nStarted: %s\n\n" % time.ctime())

    if STEP_RUN["1"]:
        t = notebook1()
        text = print_time(t)
        info = "Step 1: PoolDataAndCreateZarrStore\nStep 1 finished: %s \nTook: %s\n\n" % (
            time.ctime(),
            text,
        )
    else:
        info = "Step 1: PoolDataAndCreateZarrStore\nStep 1 finished: skipped\n\n"
    with open(p, "a") as f:
        f.write(info)

    if STEP_RUN["2"]:
        t = notebook2()
        text = print_time(t)
        info = "Step 2: FindChromatogramTransitions\nStep 2 finished: %s \nTook: %s\n\n" % (
            time.ctime(),
            text,
        )
    else:
        info = "Step 2: FindChromatogramTransitions\nStep 2 finished: skipped\n\n"
    with open(p, "a") as f:
        f.write(info)
    # exit()
    # stop here to check if (1) chromatogram is good (2) 'zero_mask' error

    if STEP_RUN["3"]:
        t0 = time.time()
        with open(p, "a") as f:
            f.write("Step 3: ComputeAverageSpectraInBin\n....")
        notebook3_part(1)
        with open(p, "a") as f:
            f.write("...3-1/%s" % SPLIT_TASK)
        synco("3", t0)
    else:
        with open(p, "a") as f:
            f.write("Step 3: ComputeAverageSpectraInBin\nStep 3 finished: skipped\n\n")

    if STEP_RUN["4"]:
        t0 = time.time()
        with open(p, "a") as f:
            f.write("Step 4: ComputeCorrelationSpectraInBin\n....")
        notebook4_part(1)
        with open(p, "a") as f:
            f.write("...4-1/%s" % v)
        synco("4", t0)
    else:
        with open(p, "a") as f:
            f.write("Step 4: ComputeCorrelationSpectraInBin\nStep 4 finished: skipped\n\n")        

    if STEP_RUN["5A"]:
        t0 = time.time()
        with open(p, "a") as f:
            f.write("Step 5A: FittingAverageSpectra\n....")
        notebook5a_part(1)
        with open(p, "a") as f:
            f.write("...5A-1/%s" % SPLIT_TASK)
        synco("5A", t0)
    else:
        with open(p, "a") as f:
            f.write("Step 5A: FittingAverageSpectra\nStep 5A finished: skipped\n\n") 

    if STEP_RUN["5B"]:
        t0 = time.time()
        with open(p, "a") as f:
            f.write("Step 5B: FittingCorrelationSpectra\n....")
        notebook5b_part(1)
        with open(p, "a") as f:
            f.write("...5B-1/%s" % SPLIT_TASK)
        synco("5B", t0)
    else:
        with open(p, "a") as f:
            f.write("Step 5B: FittingCorrelationSpectra\nStep 5B finished: skipped\n\n") 

    if STEP_RUN["6"]:
        t = notebook6()
        text = print_time(t)
        info = (
            "Step 6: SurveyFitResultsAverageAndCorr\nStep 6 finished: %s \nTook: %s\n\n"
            % (time.ctime(), text)
        )
    else:
        info = "Step 6: SurveyFitResultsAverageAndCorr\nStep 6 finished: skipped\n\n"
    with open(p, "a") as f:
        f.write(info)

    if STEP_RUN["7"]:
        t = notebook7()
        text = print_time(t)
        info = "Step 7: HuntingDigester\nStep 7 finished: %s \nTook: %s\n\n" % (
            time.ctime(),
            text,
        )
    else:
        info = "Step 7: HuntingDigester\nStep 7 finished: skipped\n\n"
    with open(p, "a") as f:
        f.write(info)

    if STEP_RUN["8"]:
        t = notebook8()
        text = print_time(t)
        info = "Step 8: CreateFitRecipesInTime\nStep 8 finished: %s \nTook: %s\n\n" % (
            time.ctime(),
            text,
        )
    else:
        info = "Step 8: CreateFitRecipesInTime\nStep 8 finished: skipped\n\n"
    with open(p, "a") as f:
        f.write(info)

    if STEP_RUN["9"]:
        t0 = time.time()
        with open(p, "a") as f:
            f.write("Step 9: RefittingData\n....")
        notebook9_part(1)
        with open(p, "a") as f:
            f.write("...9-1/%s" % SPLIT_TASK)
        synco("9", t0)
    else:
        with open(p, "a") as f:
            f.write("Step 9: RefittingData\nStep 9 finished: skipped\n\n") 
            
    if STEP_RUN["10"]:
        t = notebook10()
        text = print_time(t)
        info = "Step 10: CreateLSQSummaryFile\nStep 10 finished: %s \nTook: %s\n\n" % (
            time.ctime(),
            text,
        )
    else:
        info = "Step 10: CreateLSQSummaryFile\nStep 10 finished: skipped\n\n"
    with open(p, "a") as f:
        f.write(info)

    if STEP_RUN["11"]:
        t = notebook11()
        text = print_time(t)
        info = "Step 11: PlotLSQResults\nStep 11 finished: %s \nTook: %s\n\n" % (
            time.ctime(),
            text,
        )
    else:
        info = "Step 11: PlotLSQResults\nStep 11 finished: skipped\n\n"
    with open(p, "a") as f:
        f.write(info)
            
    if STEP_RUN["14"]:
        t = notebook14()
        text = print_time(t)
        info = "Step 14: FitConcentrationPeaks\nStep 14 finished: %s \nTook: %s\n\n" % (
            time.ctime(),
            text,
        )
    else:
        info = "Step 14: FitConcentrationPeaks\nStep 14 finished: skipped\n\n"
    with open(p, "a") as f:
        f.write(info)

    text = print_time(time.time() - t_start)
    info = "*** All analysis finished: %s \n Took: %s\n\n" % (time.ctime(), text)
    with open(p, "a") as f:
        f.write(info)
    print(info)
    
    try:
        shutil.copyfile("log.txt", pp)
    except:
        print("error copy log to r-drive.")
    


# @author: Yilin Shi | 2025.2.14
# shiyilin890@gmail.com
# Bog the Fat Crocodile vvvvvvv
#                       ^^^^^^^
