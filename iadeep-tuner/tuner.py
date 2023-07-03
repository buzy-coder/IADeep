import gc
import os
import random
import logging
import numpy as np
from typing import List
from itertools import product
from flask import Flask, request
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ml_agent import Agent
from iadeep_agent import GPLCB

import logset
logset.set_logging()

app = Flask(__name__)


# generate the candidate cloud configuration
def generated_configurations(batchsizes):
    loop_val = []
    for batchsize in batchsizes:
        lower = int(batchsize // 1.5) // 2 * 2
        upper = int(batchsize * 2)
        step = max((upper - lower) // 32, 1)
        loop_val.append(np.arange(lower, upper, step))
    p = product(*loop_val)
    return np.array(list(p))

TUNING_RECORDER = {}

@app.route('/', methods=['POST'])
def get_gp():
    jresp = request.get_json()
    job_names = jresp['Job_names']
    pod_names = jresp['Pod_names']
    init_batchsizes = jresp['Init_batchsizes']
    input_batchsizes = jresp['Batchsize_info']
    target_info = jresp['PD_info']
    dev_id = jresp['DevId']

    training_set_size = 10

    return gplcb_tuning(dev_id, job_names, init_batchsizes, input_batchsizes, target_info)

def gplcb_tuning(dev_id, job_names, init_batchsizes, input_batchsizes, target_info):
    x = generated_configurations(init_batchsizes)
    logging.info(f"Tuning job names are {job_names}")
    logging.info(f"Init batchsizes are {init_batchsizes}")
    logging.info(f"Input batchsizes are {input_batchsizes}")
    logging.info(f"Target info are {target_info}")
    logging.info(f"Generated configurations are {x}")
    # todo filter x
    # mem_matrix = np.array()
    # for job_index, col in enumerate(np.array(self.X_grid).T):
    #     mem_matrix = np.append(mem_matrix, cubic_regression(*self.cubic_models[job_index], col))
    # mem_sum = mem_matrix.sum(axis=0)
    # x_grid = [x for i, x in enumerate(self.X_grid) if mem_sum[i] <= DEVICE_MEM_LIMITATION]

    # logging.debug(f"x.T is: {x.T}")
    target_info_2d = np.array(target_info).reshape(-1, 1)
    scaler = MinMaxScaler().fit(target_info_2d)
    target_info = scaler.transform(target_info_2d)
    target_info = target_info.flatten()
    mu = np.mean(target_info)
    mu = np.array([mu for _ in range(x.shape[0])])
    sigma = np.array([0.5 for _ in range(x.shape[0])])
    x = x.tolist()
    agent = GPLCB(x, mu, sigma)
    for i in range(len(input_batchsizes)):
        agent.X.append([input_batchsizes[i]])
        agent.T.append(target_info[i])

    res_batchsizes = init_batchsizes[0]
    max_iterations = 50
    patience = 10
    count = 0
    for iteration in range(max_iterations):
        index = agent.learn(iteration+1)
        logging.info(f"Device {dev_id} iteration {iteration}: {agent.X[-1]}, {agent.T[-1]}")
        if len(agent.T) > 2 and abs(agent.T[-2] - agent.T[-1]) <= 0.01:
            count += 1
            if count >= patience:
                res_batchsizes = agent.X[-1]
                break
            
    response = {"err": '', "result": res_batchsizes, "round": iteration+1}
    logging.info(f"final iteration: {iteration}")
    logging.info(f"Device {dev_id} final selection: {res_batchsizes}")
    gc.collect()
    return response

if __name__ == '__main__':
    # PORT = os.environ['PORT']
    handler = logging.FileHandler('flask.log')
    app.logger.addHandler(handler)
    app.run(host="0.0.0.0", port='8888', debug=True)
