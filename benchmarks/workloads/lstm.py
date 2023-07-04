#coding=utf-8
import re, math, time, logging, argparse, torch, random, os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchtext.legacy.data import Field, BPTTIterator
from torchtext.legacy.datasets import WikiText2
import torchtext.vocab as vocab

import utils, logset
from etcdctl import etcd_wraper

logset.set_logging()

class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, lnorm=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp) # Shape: (vocabulary length, number of features).
        self.lnorm = None
        if lnorm:
            self.lnorm = nn.LayerNorm(ninp)        
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)  # Shape: (number of features, hidden state units, layers).        
        self.decoder = nn.Linear(nhid, ntoken) # Conversion to vocabulary tokens for final multilabel classification task.
        self.init_weights()        
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):        
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden=None):
        emb = self.drop(self.encoder(x)) # Output shape: (sequence length, batch size, number of features)
        if self.lnorm is not None:
            emb = self.lnorm(emb)        
        output, hidden = self.rnn(emb, hidden) 
        # output shape: (sequence length, batch size, hidden size)
        # hidden shape: (2 * (number of layers), batch size, hidden size). 1st for hidden state, 2nd for cell state.
        output = self.drop(output)
        # decoder input shape: (batch size * sequence length, hidden size).
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        # decoder output shape: (batch size * sequence length, vocabulary length).
        # returns: (sequence length, batch size, vocabulary length).
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):        
        weight = next(self.parameters()).data        
        return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_())        


def evaluate(data_iter):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(args.batch_size)
    for i, batch_data in enumerate(data_iter):
        data, targets = batch_data.text, batch_data.target # data and targets from torchtext.data.BPTTIterator.        
        data, targets = data.to(device), targets.to(device)
        output, hidden = model(data) # Net ouput.
        output_flat = output.view(-1, ntokens)
        total_loss += criterion(output_flat, targets.view(-1)).item() # Cumulative loss. 
         
    logging.debug(f'len: {len(data_iter)}') 
    return total_loss / len(data_iter) # returns mean loss.

# Train function.
def train(args, model, device, train_iter, optimizer, epoch, recorder: utils.TrainRecorder):
    model.train()
    total_loss = 0    
    for batch, batch_data in enumerate(train_iter):
        minibatch_time_start = time.time()
        optimizer.zero_grad()        
        data, targets = batch_data.text, batch_data.target 
        data, targets = data.to(device), targets.to(device)       
        output, hidden = model(data) # Net output.        
        loss = criterion(output.view(-1, ntokens), targets.view(-1)) # Loss and backprop.
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Clipping gradient to counter RNNs exploding gradient problem.        
        optimizer.step()

        minibatch_time_end = time.time()
        if not recorder.after_minibatch(minibatch_time_start, minibatch_time_end):
            break

        # Logging.
        total_loss += loss.item()        
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            logging.info("| epoch {:3d} | {:5d}/{:5d} batches | loss {:5.2f} | ppl {:8.2f}".format(
                epoch, batch, len(train_iter), cur_loss, math.exp(cur_loss)))
            total_loss = 0
    cur_loss = total_loss / (batch+1)
    logging.info("| epoch {:3d} | {:5d}/{:5d} batches | loss {:5.2f} | ppl {:8.2f}".format(
        epoch, batch+1, len(train_iter), cur_loss, math.exp(cur_loss)))

# Sequence generation function.
def generate(n=50, temp=1.):
    model.eval()
    # First random symbol.
    x = torch.rand(1, 1).mul(ntokens).long()
    x = x.to(device)
    hidden = None
    out = []
    # Making n length sequence.
    for i in range(n):
        output, hidden = model(x, hidden)       
        # output, hidden = output.to(device), hidden.to(device)
        s_weights = output.squeeze().data.div(temp).exp() # Gets distribution (with temperature) for next symbol.        
        s_idx = torch.multinomial(s_weights, 1)[0] # Samples next symbol index.
        x.data.fill_(s_idx)        
        s = TEXT.vocab.itos[s_idx] # Index to symbol and appends sequence.
        out.append(s)
    # returns string.
    return "".join(out)

# Train, validation and samples output.
# with torch.no_grad():
    # print("sample:\n", generate(50), "\n") # prints generated sample.


def train_lstm(args, model, device, train_iter, valid_iter, test_iter, optimizer, kwargs):
    count = 0
    recorder = utils.TrainRecorder(args.pod_name, optimizer)
    train_success = False
    global cur_batch_size
    for epoch in range(1, args.epochs+1):
        tuned_batch_size = etcd_wraper.get(args.pod_name, "tuned_batch_size")
        if tuned_batch_size is not None and utils.check_if_tuning(args.pod_name) and tuned_batch_size != cur_batch_size:
            kwargs.update({"batch_size": tuned_batch_size}, )
            logging.debug(f"Epoch {epoch} kwargs are: {kwargs}")
            train_iter, valid_iter, test_iter = BPTTIterator.splits((train_dataset, valid_dataset, test_dataset),
                                                            bptt_len=args.sequence_length, **kwargs)
            cur_batch_size = tuned_batch_size

        train(args, model, device, train_iter, optimizer, epoch, recorder)
        val_loss = evaluate(valid_iter)
        
        # scheduler.step()  

        logging.info("| end of epoch {:3d} | valid loss {:5.2f} | valid ppl {:8.2f}".format(
            epoch, val_loss, math.exp(val_loss)))

        logging.info("-" * 89)    
        
        valid_ppl = math.exp(val_loss)
        if valid_ppl <= args.valid_acc:
            count += 1
        if count >= args.patience:
            train_success = True
            break

    recorder.finish(epoch, valid_ppl, args.valid_acc, train_success)
    torch.cuda.empty_cache()            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--pod_name', type=str, default="lstm",
                        help="pod name in kubernetes cluster")
    parser.add_argument('--data_path', type=str, default="/workspace/dataset/wikitext-2", 
                        help="data path for training anf testing")
    parser.add_argument('--lr', type=float, default=1e+2, metavar='LR',
                        help='learning rate (default: 1.0)')  
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 14)') 
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training') 
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')   
    parser.add_argument('--sequence_length', type=int, default=30, metavar='N',
                        help='sequence length')                                                                                                                            
    parser.add_argument('--valid_acc', type=float, default=4.0, metavar='Acc',
                        help='valid accuracy that you want to achieve') 
    parser.add_argument('--patience', type=int, default=10, metavar='P',
                        help='earlystopping patience to achieve valid_acc')                                              
                         

    args = parser.parse_args()
    init_batchsize = args.batch_size
    pod_name = args.pod_name

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # print("||||", use_cuda)
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cuda")
    logging.debug("device is: %s", str(device))

    if os.environ.get("SCHEDULER") == "IADEEP":
        cur_batch_size = args.batch_size
    else:
        cur_batch_size = random.randrange(128, 512, 2)
    logging.info(f"cur_batch_size is: {cur_batch_size}") 
    kwargs = {'batch_size': cur_batch_size, 'shuffle': True}

    # num_workers': 0,
    # Tokenizer for splitting texts.
    tokenize = lambda x: re.findall(".", x)
    # torchtext.data.Field variable, necessary for preprocessing.
    TEXT = Field(sequential=True, tokenize=tokenize, eos_token="<eos>", lower=True)
    # Splitting WikiText2 dataset to train, validation and test sets.
    train_dataset, valid_dataset, test_dataset = WikiText2.splits(TEXT, root=args.data_path)
    # Building of vocabulary for embeddings.
    # TEXT.build_vocab(train_dataset, vectors="glove.6B.200d")
    glove = vocab.Vectors("/workspace/.vector_cache/glove.6B.200d.txt")
    TEXT.build_vocab(train_dataset, vectors=glove)
    # Vocabulary check. Each symbol assigned an id.
    # print("Vocabulary length: ", len(list(TEXT.vocab.stoi.items())))
    # print(list(TEXT.vocab.stoi.items())[:30])

    # Makes BPTTIterator for splitting corpus to sequential batches with target, shifted by 1.
    train_iter, valid_iter, test_iter = BPTTIterator.splits((train_dataset, valid_dataset, test_dataset),
                                                            bptt_len=args.sequence_length, **kwargs    
                                                            )

    grad_clip = 0.1
    best_val_loss = None

    weight_matrix = TEXT.vocab.vectors
    ntokens = weight_matrix.shape[0] # Number of tokens for embedding.
    nfeatures = weight_matrix.shape[1]

    model = RNNModel(ntokens, 128, 128, 2, 0.3, lnorm=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=0.)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) 

    try:
        utils.setup_seed(0)
        train_lstm(args, model, device, train_iter, valid_iter, test_iter, optimizer, kwargs)
    except RuntimeError as exception:
        # if "Out of memory" in str(exception):
        logging.info("Warning: out of memory due to a big batchsize!")
        torch.cuda.empty_cache()
        args.batch_size = int(args.batch_size/2)
        logging.debug(f"args.batch_size is: {args.batch_size}")
        train_lstm(args, model, device, train_iter, valid_iter, test_iter, optimizer, kwargs)
        torch.cuda.empty_cache()