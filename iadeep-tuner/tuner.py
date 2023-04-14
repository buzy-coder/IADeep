import gc
import os
import random
import logging
import numpy as np
from typing import List
from itertools import product
from flask import Flask, request
from ml_agent import Agent
from iadeep_agent import GPLCB


app = Flask(__name__)


# generate the candidate cloud configuration
def generated_configurations(batchsizes):
    loop_val = []
    for batchsize in batchsizes:
        lower = batchsize // 2
        upper = int(batchsize * 1.5)
        step = max((upper - lower) // 16, 1)
        loop_val.append(np.arange(lower, upper, step))
    p = product(*loop_val)
    return np.array(list(p))


TUNING_RECORDER = {}


@app.route('/', methods=['POST'])
def get_gp():
    max_iterations = 50
    jresp = request.get_json()
    job_names = jresp['Job_names']
    pod_names = jresp['Pod_names']
    init_batchsizes = jresp['Init_batchsizes']
    input_batchsizes = jresp['Batchsize_info']
    target_info = jresp['PD_info']
    dev_id = jresp['DevId']

    training_set_size = 10
    agent_type = os.environ.get("AGENT", "gplcb")
    if agent_type != "gplcb":
        return ml_tuning(
            dev_id,
            job_names,
            init_batchsizes,
            agent_type,
            input_batchsizes,
            training_set_size,
            target_info
        )
    else:
        return gplcb_tuning(
            dev_id, job_names, init_batchsizes, input_batchsizes, target_info
        )

def ml_tuning(
    dev_id, job_names, init_batchsizes, agent_type, input_batchsizes, training_set_size, target_info
):
    if dev_id not in TUNING_RECORDER:
        # If device id is not recorded yet, init related data
        # and randomly generate a training set
        _configs = generated_configurations(init_batchsizes).tolist()
        agent = Agent(_configs, agent_type)
        train_batchsizes = [input_batchsizes]
        train_batchsizes.extend(random.sample(list(_configs), training_set_size - 1))
        TUNING_RECORDER[dev_id] = {
            "agent": agent,
            "train_batchsizes": train_batchsizes,
            "real_batchsizes": [input_batchsizes],
            "pd_info": target_info,
            "rounds": 0,
            "finished": False
        }

        result = TUNING_RECORDER[dev_id]["train_batchsizes"][1]
        print(f"INFO: Device {dev_id} init! Agent type: {agent_type}, Training set size: {training_set_size}")
    elif TUNING_RECORDER[dev_id]["finished"]:
        del TUNING_RECORDER[dev_id]
        print(f"INFO: Tuning on device {dev_id} finished!")
        response = {"err": "", "result": []}
        gc.collect()
        return response
    elif TUNING_RECORDER[dev_id]["rounds"] < training_set_size - 1:
        # If this device  is still under training process,
        # record PD info and return next training batchsizes
        dev_recorder = TUNING_RECORDER[dev_id]
        train_batchsizes = dev_recorder["train_batchsizes"]
        real_batchsizes: List = dev_recorder["real_batchsizes"]
        pd_inf: List = dev_recorder["pd_info"]
        passed_rounds = dev_recorder["rounds"]

        pd_inf.extend(target_info)
        real_batchsizes.append(input_batchsizes)

        result = train_batchsizes[passed_rounds + 1]
        print(f"INFO: Device {dev_id} training! Current round: {passed_rounds}, Batchsizes: {result}")
    else:
        # We have collected enough data on this device, now we are
        # going to estimate PD and check its performance
        dev_recorder = TUNING_RECORDER[dev_id]
        agent: Agent = dev_recorder["agent"]
        dev_recorder["real_batchsizes"].append(input_batchsizes)
        dev_recorder["pd_info"].extend(target_info)
        # Train agent using training data
        X = dev_recorder["real_batchsizes"]
        Y = dev_recorder["pd_info"]
        agent.train(X, Y)
        print(f"INFO: Device {dev_id} trained! Agent type: {agent_type}")
        # Select the optimized batchsizes, predict its PD
        result = agent.select()
        TUNING_RECORDER[dev_id]["finished"] = True
        print(f"INFO: Device {dev_id} final selection: {result}")
    TUNING_RECORDER[dev_id]["rounds"] += 1
    response = {"err": "", "result": result}
    gc.collect()
    return response

def gplcb_tuning(dev_id, job_names, init_batchsizes, input_batchsizes, target_info):
    x = generated_configurations(init_batchsizes)
    # todo filter x
    # mem_matrix = np.array()
    # for job_index, col in enumerate(np.array(self.X_grid).T):
    #     mem_matrix = np.append(mem_matrix, cubic_regression(*self.cubic_models[job_index], col))
    # mem_sum = mem_matrix.sum(axis=0)
    # x_grid = [x for i, x in enumerate(self.X_grid) if mem_sum[i] <= DEVICE_MEM_LIMITATION]
    print("x.T is: ", x.T)
    mu = np.mean(target_info)
    mu = np.array([mu for _ in range(x.shape[0])])
    sigma = np.array([0.5 for _ in range(x.shape[0])])
    x = x.tolist()
    input_x = []
    input_x.append(input_batchsizes)
    if dev_id in TUNING_RECORDER:
        old_agent = TUNING_RECORDER[dev_id]['agent']
        if len(old_agent.X[-1]) == len(input_batchsizes):
            input_x = old_agent.X + input_x
            target_info = old_agent.T + target_info
            mu = old_agent.mu
            sigma = old_agent.sigma
            TUNING_RECORDER[dev_id]['round'] += 1
        else:
            TUNING_RECORDER[dev_id] = {
                "round": 1
            }
    else:
        TUNING_RECORDER[dev_id] = {
            "round": 1
        }

    agent = GPLCB(x, mu, sigma)

    agent.X.extend(input_x)
    agent.T.extend(target_info)

    index = agent.learn(1)
    res_batchsizes = x[index]

    TUNING_RECORDER[dev_id]['agent'] = agent

    if len(agent.T) > 2 and len(agent.T) <= 500:
        print(agent.T[-1])
        if abs(agent.T[-2] - agent.T[-1]) <= 0.01:
            res_batchsizes = []
            if dev_id in TUNING_RECORDER:
                del TUNING_RECORDER[dev_id]

    response = {"err": '', "result": res_batchsizes}
    gc.collect()
    return response


if __name__ == '__main__':
    # PORT = os.environ['PORT']
    handler = logging.FileHandler('flask.log')
    app.logger.addHandler(handler)
    app.run(host="0.0.0.0", port='8888', debug=True)
