""" Scrapes the save directory and outputs the 
    results of each training run into a run_info.csv file
"""

import os
import datetime
import pandas
import json

import subprocess

def read_args(lines):
    json_str = "{"
    i = 1
    while "}" not in lines[i]:
        json_str += lines[i]
        i += 1
    json_str += "}"

    args = json.loads(json_str)

    return args

if __name__ == "__main__":
    files = os.listdir("save")

    run_info = dict() 
    for filename in files:
        print(filename)

        try:
            with open(f'save/{filename}/log_train.txt') as file_:
            
                lines = file_.readlines()
                args = read_args(lines)
                args["run_date"] = datetime.datetime.fromtimestamp(os.stat(f'save/{filename}').st_ctime)

                try:
                    output = str(subprocess.check_output(["python", "train.py", "--do-eval", "--save-dir", f'save/{filename}', "--eval-dir", "datasets/oodomain_val"]))
                    metrics = output.split("[")[4][19:].strip().split(',')
                    args["F1"] = metrics[0][9:]
                    args["EM"] = metrics[1][5:]
                    args["composite_qa_loss"] = metrics[2][20:]
                    args["avg_kl_divergence"] = metrics[3][20:25]
                except:
                    pass

                run_info[filename] = args
        except:
            pass

    dataframe = pandas.DataFrame.from_dict(run_info, orient='index')
    dataframe.index.name = 'Run Name'

    dataframe.to_csv(f'save/run_info.csv', sep='\t')