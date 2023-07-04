import os, logging, random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import model
import config
import evaluate
import data_utils
import sys
sys.path.append("/workspace/workloads/")

import utils, logset
from etcdctl import etcd_wraper

logset.set_logging()

def train_neumf(args, model, device, train_loader, test_loader, optimizer, kwargs):
    if config.model == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None
    
    import model
    factor_num = 32
    num_layers = 3
    dropout = 0.0
    
    model = model.NCF(user_num, item_num, factor_num, num_layers, 
                            dropout, config.model, GMF_model, MLP_model)
    model.cuda()
    loss_function = nn.BCEWithLogitsLoss()
    
    lr = 0.001
    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    count = 0
    recorder = utils.TrainRecorder(args.pod_name, optimizer)
    train_success = False
    global cur_batch_size
    for epoch in range(1, args.epochs+1):
        tuned_batch_size = etcd_wraper.get(args.pod_name, "tuned_batch_size")
        if tuned_batch_size is not None and utils.check_if_tuning(args.pod_name) and tuned_batch_size != args.batch_size:
            kwargs.update({"batch_size": tuned_batch_size}, )
            logging.debug(f"Epoch {epoch} kwargs are: {kwargs}")
            train_loader = DataLoader(train_dataset, **kwargs)
            cur_batch_size = tuned_batch_size

        model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()
 
        i = 0
        for user, item, label in train_loader:
            minibatch_time_start = time.time()
            optimizer.zero_grad()
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step() 

            minibatch_time_end = time.time()
            if not recorder.after_minibatch(minibatch_time_start, minibatch_time_end):
                break
        
        model.eval()
        top_k = 10
        HR, NDCG = evaluate.metrics(model, test_loader, top_k)

        elapsed_time = time.time() - start_time
        logging.info("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        logging.info("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))    
                                        
        if HR >= args.valid_acc:
            count += 1
        if count >= args.patience:
            train_success = True
            break  
    recorder.finish(epoch, HR, args.valid_acc, train_success)
    torch.cuda.empty_cache()            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--pod_name', type=str, default="neumf",
                        help="pod name in kubernetes cluster")
    parser.add_argument('--trained_count', type=int, default=0,
                        help='count the epoch once the training model achieves the accuracy')
    parser.add_argument('--mini_batch_steps', type=int, default=-1,
                        help='a tag to control python process to start or stop')
    parser.add_argument('--data_path', type=str, default="/workspace/dataset/cifar10", 
                        help="data path for training anf testing")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')  
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 4000)') 
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training') 
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')          
    parser.add_argument('--valid_acc', type=float, default=0.69, metavar='Acc',
                        help='valid accuracy that you want to achieve') 
    parser.add_argument('--patience', type=int, default=10, metavar='P',
                        help='earlystopping patience to achieve valid_acc')

    args = parser.parse_args()
    pod_name = args.pod_name

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.debug("device is: %s", str(device))
    if os.environ.get("SCHEDULER") == "IADEEP":
        cur_batch_size = args.batch_size
    else:
        cur_batch_size = random.randrange(256, 1024, 2)
    logging.info(f"cur_batch_size is: {cur_batch_size}")     
    kwargs = {'batch_size': cur_batch_size}    
    if use_cuda:
        kwargs.update({'num_workers': 0,
                       'pin_memory': False,
                       'shuffle': True},)

    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    num_ag = 4
    test_num_ng = 99
    train_dataset = data_utils.NCFData(
            train_data, item_num, train_mat, 4, True)
    test_dataset = data_utils.NCFData(
            test_data, item_num, train_mat, 0, False)
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=test_num_ng+1, shuffle=False, num_workers=0)

    if config.model == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None
    
    factor_num = 32
    num_layers = 3
    dropout = 0.0

    model = model.NCF(user_num, item_num, factor_num, num_layers, 
                                dropout, config.model, GMF_model, MLP_model)
    model.cuda()
    loss_function = nn.BCEWithLogitsLoss()
    

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) 

    try:
        utils.setup_seed(0)
        train_neumf(args, model, device, train_loader, test_loader, optimizer, kwargs)
    except RuntimeError as exception:
        # if "Out of memory" in str(exception):
        logging.info("Warning: out of memory due to a big batchsize!")
        torch.cuda.empty_cache()
        args.batch_size = int(args.batch_size/2)
        logging.debug(f"args.batch_size is: {args.batch_size}")
        train_neumf(args, model, device, train_loader, test_loader, optimizer, kwargs)
        torch.cuda.empty_cache()

                        
