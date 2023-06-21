import torch
import numpy as np
import functools
import traceback
import logging

from etcdctl import ETCD_WRAPER
etcd_wraper = ETCD_WRAPER()

import logset
logset.set_logging()


def print_exc(function):
    """
    A decorator that wraps the passed in function and prints any exceptions.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            raise
    return wrapper    

class GradientNoiseScale(object):

    def __init__(self, pod_name, optimizer, small_grad_sqr, big_grad_sqr):
        self.mixed_precision_scale = 1.0
        self.optimizer = optimizer
        self._small_batchsize = etcd_wraper.get(pod_name, "tuned_batch_size")
        self._big_batchsize = self._small_batchsize * 2
        self.small_grad_sqr = small_grad_sqr
        self.big_grad_sqr = big_grad_sqr
    
    def _get_preconditioner(self):
        out = []
        for idx, group in enumerate(self._pre_optimizer.param_groups):
            pinvs = []
            for param in group["params"]:
                pinv = self._calculate_preconditioner(idx, param)
                pinvs.append(pinv)
            out.append(pinvs)
        return out

    def _calculate_preconditioner(self, idx, param):
        return torch.ones_like(param, memory_format=torch.preserve_format)

    def sqr_avg(self, val):
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared).

        Returns (float): Estimate of squared l2-norm.
        """
        return float(np.sum(np.maximum(val, 0.0)))

    def var_avg(self, val):
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared).

        Returns (float): Estimate of trace of the covariance.
        """
        return float(np.sum(np.maximum(val, 1e-6)))
    
    # Current estimate of the squared l2-norm of the true gradients 
    def get_grad_sqr(self):
        small_grad_sqr = self.small_grad_sqr
        big_grad_sqr = self.big_grad_sqr
        logging.debug("1/(self._big_batchsize - self._small_batchsize) is: ", 1/(self._big_batchsize - self._small_batchsize))
        logging.debug("self._big_batchsize * big_grad_sqr - self._small_batchsize * small_grad_sqr is: ", self._big_batchsize * big_grad_sqr - self._small_batchsize * small_grad_sqr)
        
        grad_sqr = 1/(self._big_batchsize - self._small_batchsize) * \
            (self._big_batchsize * big_grad_sqr - self._small_batchsize * small_grad_sqr)
        grad_sqr = self.sqr_avg(abs(grad_sqr))    
        logging.debug("grad_sqr is: ", grad_sqr)
        return grad_sqr    

    # Estimate of the trace of the covariance
    def get_grad_var(self):
        small_grad_sqr = self.small_grad_sqr
        big_grad_sqr = self.big_grad_sqr
        # grad_var = 1/(1/self._small_batchsize - 1/self._big_batchsize) * (self._small_grad_sqr - self._big_grad_sqr + 10)
        grad_var = 1/(1/self._small_batchsize - 1/self._big_batchsize) * (small_grad_sqr - big_grad_sqr)
        grad_var = self.var_avg(grad_var)
        logging.debug("grad_var is: ", grad_var)
        return grad_var

    def get_gns(self):
        return self.get_grad_var() / self.get_grad_sqr()

class PerformanceDegradation(object):

    def __init__(self, pod_name, optimizer, small_grad_sqr, big_grad_sqr):
        self.optimizer = optimizer
        self.pod_name = pod_name
        self.small_grad_sqr = small_grad_sqr
        self.big_grad_sqr = big_grad_sqr
        self.efficiency = 0.0
        self.performance_degradation = 0.0
        self.m_0 = etcd_wraper.get(pod_name, "init_batch_size")
        self.m = etcd_wraper.get(pod_name, "tuned_batch_size")
        logging.debug("m_0 is: ", self.m_0)
        logging.debug("m is: ", self.m)

    def get_efficiency(self):
        GNS = GradientNoiseScale(self.pod_name, self.optimizer, self.small_grad_sqr, self.big_grad_sqr)

        logging.debug("GNS.gns is: ", GNS.get_gns())
        etcd_wraper.put(self.pod_name, "gns", str(GNS.get_gns()))
        efficiency = (GNS.get_gns() + self.m_0)/(GNS.get_gns() + self.m)
        return efficiency

    # calculate PD 
    def get_performance_degradation(self):
        t_co = self.get_avg_minibatch_time("minibatch_time")
        t_m0 = self.get_avg_minibatch_time("mini_batch_time_m0")
        logging.debug(f"t_co is:{t_co}, t_m0 is {t_m0}, m0 is {self.m_0}, m is {self.m}")
        logging.debug("efficiency is: ", self.get_efficiency())
        res = (t_co/t_m0)*(self.m_0/self.m)*self.get_efficiency()
        logging.debug("pd is: ", res)
        etcd_wraper.put(self.pod_name, "pd", str(res))
        return res 

    def get_avg_minibatch_time(self, key):
        avg_time = etcd_wraper.get(self.pod_name, key)
        return avg_time                  