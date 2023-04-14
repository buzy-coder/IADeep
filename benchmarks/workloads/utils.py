import numpy as np
import torch
import logging
from estimator import PerformanceDegradation
from etcdctl import etcd_wraper

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def check_if_tuning(pod_name):
    tuning = etcd_wraper.get(pod_name, 'tuning')
    if tuning == 1:
        return True
    return False
    
def reset_tuning(pod_name):
    etcd_wraper.put(pod_name, 'tuning', -1)

def get_grad_sqr(optimizer):
    grads = []
    mixed_precision_scale = 1.0
    for group in optimizer.param_groups:
        grads.append([])
        for param in group["params"]:
            # print("param.grad is:", param.grad)
            if param.grad is None:
                grads[-1].append(None)
                continue
            grad = param.grad.detach().float()
            # print("grad is: ", grad)
            grads[-1].append(
                grad / mixed_precision_scale)
            # print("param.grad is: ", param.grad)   
    preconditioner = get_preconditioner(optimizer)
    grads_normsqr = normsqr_groups(grads, preconditioner)
    print("grad_sqr is: ", float(np.sum(grads_normsqr)))
    return float(np.sum(grads_normsqr))

def get_preconditioner(optimizer):
    out = []
    for idx, group in enumerate(optimizer.param_groups):
        pinvs = []
        for param in group["params"]:
            pinv = calculate_preconditioner(idx, param)
            pinvs.append(pinv)
        out.append(pinvs)
    return out

def calculate_preconditioner(idx, param):
    return torch.ones_like(param, memory_format=torch.preserve_format)


def normsqr_groups(grads, pinvs):
    ret = []
    for group, pinv_group in zip(grads, pinvs):
        normsqr = [(g / pinv).pow(2).sum(dtype=torch.float64)
                for g, pinv in zip(group, pinv_group) if g is not None]
        ret.append(sum(normsqr).item() if normsqr else 0.0)
    return np.array(ret)   


COUNT_SAMPLE = 5
class TrainRecorder:
    def __init__(self, pod_name, optimizer):
        self.pod_name = pod_name
        self.optimizer = optimizer
        self.minibatch_times = []
        self.tuning_cost = 0
    
    def finish(self, epoch, final_sco, target_sco, success):
        """Record data to etcd after training done.

        Args:
            epoch (int): Number of epochs used
            final_sco (float): Score/accuracy of the last epoch
            target_sco (float): Score/accuracy that the model want to reach
            success (bool): Is the training reach the target?
        """        
        etcd_wraper.put(self.pod_name, 'complete', "1")
        etcd_wraper.put(self.pod_name, 'epoch_times', epoch)
        etcd_wraper.put(self.pod_name, 'train_success', 1 if success else -1)
        etcd_wraper.put(self.pod_name, 'final_sco', final_sco)
        etcd_wraper.put(self.pod_name, 'target_sco', target_sco)

    def after_minibatch(self, minibatch_time_start, minibatch_time_end):
        """After each minibatch, check if current job is under tuning.
        If yes, record related data to etcd, otherwise, do nothing.

        Args:
            minibatch_time_start (float): timestamp of current minibatch start time
            minibatch_time_end (float): timestamp of current minibatch end time

        Returns:
            bool: Should current epoch continue? True -> keep on going, False -> break the epoch
        """        
        minibatch_time = minibatch_time_end - minibatch_time_start
        if check_if_tuning(self.pod_name):
            print("tuning!")
            self.minibatch_times.append(minibatch_time)
            if len(self.minibatch_times) == 1: 
                self.tuning_start_time = minibatch_time_start
                self.small_grad_sqr = get_grad_sqr(self.optimizer)
                return True

            if len(self.minibatch_times) == 2: 
                self.big_grad_sqr = get_grad_sqr(self.optimizer)
                return True

            if len(self.minibatch_times) == COUNT_SAMPLE:
                etcd_wraper.put(
                    self.pod_name,
                    "minibatch_time",
                    np.average(self.minibatch_times)
                )
                self.minibatch_times = []
                PD = PerformanceDegradation(
                    self.pod_name,
                    self.optimizer,
                    self.small_grad_sqr,
                    self.big_grad_sqr
                )
                pd_val = PD.get_performance_degradation()
                print("pd is: ", pd_val)
                reset_tuning(self.pod_name)
                tuning_end_time = minibatch_time_end
                tuning_time = tuning_end_time - self.tuning_start_time
                print("Tuning cost: ", tuning_time)
                self.tuning_cost += tuning_time
                etcd_wraper.put(self.pod_name, "tuning_consumption", self.tuning_cost)
                return False
        return True