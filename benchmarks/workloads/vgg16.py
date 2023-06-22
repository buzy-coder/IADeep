import time
import argparse
import logging
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import utils, logset
from etcdctl import etcd_wraper
logset.set_logging()

def train(args, model, device, train_loader, optimizer, epoch, recorder: utils.TrainRecorder):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        minibatch_time_start = time.time()

        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        
        minibatch_time_end = time.time()
        if not recorder.after_minibatch(minibatch_time_start, minibatch_time_end):
            break
     
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.6f}%)]\tTrain Loss: {:.6f}'.format(
            epoch, batch_idx*args.batch_size, len(train_loader.dataset),
            100.0 * batch_idx / len(train_loader), loss.item() / args.batch_size)) 

    train_loss /= len(train_loader.dataset)
    accuracy = 100. * train_correct / len(train_loader.dataset)
    logging.info('Train Epoch: {} Train set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(epoch, train_loss, train_correct, len(train_loader.dataset), accuracy))       

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    accuracy = 100. * correct / len(test_loader.dataset)
    logging.info('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))     
    return accuracy      

def train_vgg16(args, model, device, train_loader, test_loader, optimizer, kwargs):
    count = 0
    recorder = utils.TrainRecorder(args.pod_name, optimizer)
    train_success = False
    for epoch in range(1, args.epochs + 1):
        tuned_batch_size = etcd_wraper.get(args.pod_name, "tuned_batch_size")
        if tuned_batch_size is not None and tuned_batch_size != args.batch_size:
            kwargs.update({"batch_size": tuned_batch_size})
            logging.info(f"Epoch {epoch} kwargs are: {kwargs}")
            train_loader = DataLoader(train_dataset, **kwargs)

        train(args, model, device, train_loader, optimizer, epoch, recorder)
        accuracy = test(model, device, test_loader)
  
        scheduler.step()  

        if accuracy/100 >= args.valid_acc:
            count += 1
        if count >= args.patience:
            train_success = True
            break

    recorder.finish(epoch, accuracy / 100, args.valid_acc, train_success)
    torch.cuda.empty_cache()            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--pod_name', type=str, default="vgg16",
                        help="pod name in kubernetes cluster")
    parser.add_argument('--data_path', type=str, default="/workspace/dataset/cifar10", 
                        help="data path for training anf testing")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')  
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)') 
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training') 
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')                                                                                                                              
    parser.add_argument('--valid_acc', type=float, default=0.80, metavar='Acc',
                        help='valid accuracy that you want to achieve') 
    parser.add_argument('--patience', type=int, default=10, metavar='P',
                        help='earlystopping patience to achieve valid_acc')                          

    args = parser.parse_args()
    init_batchsize = args.batch_size
    pod_name = args.pod_name  

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.debug("device is: %s", str(device))

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': False,
                       'shuffle': True},)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.CIFAR10(args.data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(args.data_path, train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)

    model = models.vgg16(pretrained=False)
    model.load_state_dict(torch.load('/workspace/cache/vgg16.pth'))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) 

    try:
        utils.setup_seed(0)
        train_vgg16(args, model, device, train_loader, test_loader, optimizer, kwargs)
    except RuntimeError as exception:
        if "Out of memory" in str(exception):
            logging.info("Warning: out of memory due to a big batchsize!")
            torch.cuda.empty_cache()
            args.batch_size = int(args.batch_size/2)
            logging.debug(f"args.batch_size is: {args.batch_size}")
            train_vgg16(args, model, device, train_loader, test_loader, optimizer, kwargs)
            torch.cuda.empty_cache()
