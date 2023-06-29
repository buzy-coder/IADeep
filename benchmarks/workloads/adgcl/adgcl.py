import time
import os
import sys
import argparse
import logging
import warnings
import torch
from sklearn.svm import LinearSVC, SVC
from torch_geometric.data import DataLoader
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter

# sys.path.append("datasets")
from datasets.tu_dataset import TUDataset, TUEvaluator
from unsupervised.embedding_evaluation import EmbeddingEvaluation
from unsupervised.encoder import TUEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight, initialize_node_features, set_tu_dataset_y_shape
from unsupervised.view_learner import ViewLearner

sys.path.append("/workspace/workloads/")
from estimator import PerformanceDegradation
import utils, logset
from etcdctl import etcd_wraper
warnings.filterwarnings("ignore")

logset.set_logging()

class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: Batch.from_data_list(
                data_list, follow_batch),
            **kwargs)



class DataListLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a python list.

    .. note::

        This data loader should be used for multi-gpu support via
        :class:`torch_geometric.nn.DataParallel`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataListLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: data_list,
            **kwargs)



class DenseDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    .. note::

        To make use of this data loader, all graphs in the dataset needs to
        have the same shape for each its attributes.
        Therefore, this data loader should only be used when working with
        *dense* adjacency matrices.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        def dense_collate(data_list):
            batch = Batch()
            for key in data_list[0].keys:
                batch[key] = default_collate([d[key] for d in data_list])
            return batch

        super(DenseDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=dense_collate, **kwargs)


def run(args, model, device, dataloader, optimizer, kwargs):

    view_learner = ViewLearner(TUEncoder(num_dataset_features=1, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)
    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True), evaluator, dataset.task_type, dataset.num_tasks,
                             device, param_search=False)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=False)

    model.eval()
    train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score,
                                                                                         test_score))

    model_losses = []
    view_losses = []
    view_regs = []  
    valid_curve = []
    test_curve = []     
    train_curve = []        

    count = 0
    recorder = utils.TrainRecorder(args.pod_name, optimizer)
    train_success = False
    for epoch in range(1, args.epochs + 1):
        
        tuned_batch_size = etcd_wraper.get(args.pod_name, "tuned_batch_size")
        if tuned_batch_size is not None and tuned_batch_size != args.batch_size:
            kwargs.update({"batch_size": tuned_batch_size}, )
            logging.debug(f"Epoch {epoch} kwargs are: {kwargs}")
            dataloader = DataLoader(dataset, **kwargs)

        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0
        
        i = -1
        for batch in dataloader:
            i = i + 1
            minibatch_time_start = time.time()
            # set up
            batch = batch.to(device)

            # train view to maximize contrastive loss
            view_learner.train()
            view_learner.zero_grad()
            model.eval()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)

            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            # regularization

            row, col = batch.edge_index
            edge_batch = batch.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            reg = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
            reg = reg.mean()

            view_loss = model.calc_loss(x, x_aug) - (args.reg_lambda * reg)
            view_loss_all += view_loss.item() * batch.num_graphs
            reg_all += reg.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()

            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            # standard gradient descent formulation
            model_loss.backward()
            optimizer.step()

            minibatch_time_end = time.time()
            if not recorder.after_minibatch(minibatch_time_start, minibatch_time_end):
                break

        fin_model_loss = model_loss_all / len(dataloader)
        fin_view_loss = view_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)

        logging.info('Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(epoch, fin_model_loss, fin_view_loss, fin_reg))
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)
        if epoch % args.eval_interval == 0:
            model.eval()

            train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)

            logging.info(
                "Metric: {} Train: {} Val: {} Test: {}".format(evaluator.eval_metric, train_score, val_score, test_score))

            train_curve.append(train_score)
            valid_curve.append(val_score)
            test_curve.append(test_score)
        logging.info("val_acc is : {:.2f}".format(val_score))
        
        if val_score >= args.valid_acc:
            count += 1
        if count >= args.patience:
            train_success = True
            break  
    recorder.finish(epoch, val_score, args.valid_acc, train_success)
    torch.cuda.empty_cache()            

    logging.info('FinishedTraining!')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default='REDDIT-MULTI-12K',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=5,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--pod_name', type=str, default="adgcl",
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
                        help='number of epochs to train (default: 14)') 
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training') 
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')  
    parser.add_argument('--valid_acc', type=float, default=0.40, metavar='Acc',
                        help='valid accuracy that you want to achieve') 
    parser.add_argument('--patience', type=int, default=10, metavar='P',
                        help='earlystopping patience to achieve valid_acc') 
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout Ratio / Probability')  
    parser.add_argument('--reg_lambda', type=float, default=5.0, 
                        help='View Learner Edge Perturb Regularization Strength')
    parser.add_argument('--eval_interval', type=int, default=1, 
                        help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear", 
                        help="Downstream classifier is linear or non-linear")


    args = parser.parse_args()
    init_batchsize = args.batch_size
    pod_name = args.pod_name
    trained_count = args.trained_count
    mini_batch_steps = 5 if args.mini_batch_steps >= 1 else args.mini_batch_steps
    check_points_path = '/workspace/workloads/result/' + pod_name + '/'
    if not os.path.exists(check_points_path):
        os.makedirs(check_points_path)     

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.debug("device is: %s", str(device))

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 0,
                       'pin_memory': False,
                       'shuffle': True},)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print("Using Device: %s" % device)
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % 0)
    logging.info(args)

    evaluator = TUEvaluator()
    my_transforms = Compose([initialize_node_features, initialize_edge_weight, set_tu_dataset_y_shape])
    dataset = TUDataset("/workspace/dataset/", args.dataset, transform=my_transforms)

    dataloader = DataLoader(dataset, **kwargs)

    model = GInfoMinMax(
        TUEncoder(num_dataset_features=1, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        args.emb_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    try:
        utils.setup_seed(0)
        run(args, model, device, dataloader, optimizer, kwargs)
    except RuntimeError as exception:
        # if "Out of memory" in str(exception):
        logging.info("Warning: out of memory due to a big batchsize!")
        torch.cuda.empty_cache()
        args.batch_size = int(args.batch_size/2)
        logging.debug(f"args.batch_size is: {args.batch_size}")
        run(args, model, device, dataloader, optimizer, kwargs)
        torch.cuda.empty_cache()
