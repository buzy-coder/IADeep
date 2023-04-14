import sys, torch, json, time, argparse
sys.path.append("/workspace/workloads/")
import utils
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizerFast
from torch.nn import DataParallel
from bert.dataset import SquadDataset
from bert.model import BertForSQuAD
from bert.util import *
from torch.utils.data import DataLoader
from etcdctl import etcd_wraper

print(1)

parser = argparse.ArgumentParser(description='SQuAD train & test options')
        
parser.add_argument('--pod_name', type=str, default="bert",help="pod name in kubernetes cluster")
parser.add_argument('--model', type=str, default='bert', help='model of the experiment.(bert or xlnet)')

parser.add_argument('--devices', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--doload', default='', type=str, required=False, help='if load ')
parser.add_argument('--checkpoint_path', type=str, default='.', help='models are saved here')
parser.add_argument('--sink',  type=bool,  default=False, help='if load model from last checkpoint')
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model')


parser.add_argument("--batch_size", type=int, dest="batch_size", default=8, help="Mini-batch size")
parser.add_argument("--lr", type=float, dest="lr", default=5e-5, help="Base Learning Rate")
parser.add_argument("--epochs", type=int, dest="epochs", default=100, help="Number of iterations")
parser.add_argument('--visualize', type=bool, default=True, help='if open tensorboardX')
parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
parser.add_argument('--max_grad_norm', default=1, type=float, required=False, help='max grad norm')
parser.add_argument('--num_workers', default=0, type=int, required=False)
parser.add_argument('--f1_sco', type=float, default=0.88, metavar='Sco',help='f1 score that you want to achieve')
parser.add_argument('--patience', type=int, default=10, metavar='P',
                help='earlystopping patience to achieve f1_sco')  
        
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"device is: {device}")


os.environ["TOKENIZERS_PARALLELISM"] = "true"

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_dataset = SquadDataset(tokenizer, opt, split = 'train')
valid_dataset = SquadDataset(tokenizer, opt, split = 'valid')

train_loader = DataLoader(train_dataset,batch_size = opt.batch_size,shuffle = True,num_workers = opt.num_workers)
valid_loader = DataLoader(valid_dataset,batch_size = opt.batch_size,shuffle = False,num_workers = opt.num_workers)
check_path = opt.checkpoint_path


with open("/workspace/dataset/squad-2/SQuAD2/dev-v2.0.json") as file:
    qas = {
        qa["id"]: {key: qa[key] for key in qa if key != "id"}
        for a in json.load(file)["data"]
        for p in a["paragraphs"]
        for qa in p["qas"]
    }

epoch = 0
best_loss = 1e20
epochs = opt.epochs
model = BertForSQuAD.from_pretrained('bert-base-uncased')
name = opt.pod_name + "_" + opt.model

print(f"Model Creates")

model.train()
model.to(device)
if torch.cuda.device_count() > 1:
    model = DataParallel(model, device_ids=[int(i) for i in opt.devices.split(',')])
    
optimizer = AdamW(model.parameters(), lr=opt.lr, correct_bias=True)
total_steps = int(epochs * len(train_dataset) / opt.batch_size)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_steps,num_training_steps=total_steps)
     
max_grad_norm = opt.max_grad_norm    
kwargs = {'batch_size': opt.batch_size}
count = 0
train_success = False
for e in range(epoch, epochs):
    tuned_batch_size = etcd_wraper.get(opt.pod_name, "tuned_batch_size")
    if tuned_batch_size is not None and tuned_batch_size != opt.batch_size:
        kwargs.update({"batch_size": tuned_batch_size}, )
        print(f"Epoch {e+1} kwargs are: {kwargs}")
        train_loader = DataLoader(
            train_dataset,
            batch_size = kwargs["batch_size"],
            shuffle = True,
            num_workers = opt.num_workers
        )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    model.train()
    recorder = utils.TrainRecorder(opt.pod_name, optimizer)
    # Training process
    for i, data in enumerate(train_loader):
        minibatch_time_start = time.time()

        input_ids = data['input_ids'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        start_positions = data['start_positions'].to(device)
        end_positions = data['end_positions'].to(device)
        ans_exists = data['ans_exists'].to(device)
        outputs = model(input_ids,  token_type_ids = token_type_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, ans_exists = ans_exists)
        loss = outputs["loss"].mean()
        optimizer.zero_grad()     
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()   
        scheduler.step()
        losses.update(loss.data.item(), input_ids.size(0))
        batch_time.update(time.time() - end)

        minibatch_time_end = time.time()
        if not recorder.after_minibatch(minibatch_time_start, minibatch_time_end):
            break
        if i % 100 == 0 and i > 0:
            print("| epoch {:3d} | {:5d}/{:5d} batches".format(e, i, len(train_loader)))

    model.eval()
    test_losses = AverageMeter()

    if i + 1 < len(train_loader):
        # In case of tuning batchsize, i.e. break at line 125
        # No need to take validation process, it costs too much!
        continue
    # Validation process
    f1 = 0
    for i, data in enumerate(valid_loader):
        with torch.no_grad():
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            start_positions = data['start_positions'].to(device)
            end_positions = data['end_positions'].to(device)
            ans_exists = data['ans_exists'].to(device)

            outputs = model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, ans_exists = ans_exists)
            pred_start_positions = torch.argmax(outputs["start_logits"], dim = 1)
            pred_end_positions = torch.argmax(outputs["end_logits"], dim = 1)
            pred_ans_exists = torch.argmax(outputs["ans_logits"], dim = 1)
            loss = outputs["loss"].mean()

            for qa_id, start_pos, end_pos, ans, index in zip(
                data["ids"],
                pred_start_positions,
                pred_end_positions,
                pred_ans_exists,
                range(len(data["ids"])),
            ):
                if start_pos >= end_pos or ans == 0:
                    pred_ans = None
                else:
                    pred_ans = input_ids[index][int(start_pos):int(end_pos)]
                    pred_ans = tokenizer.decode(pred_ans)
                qa = qas[qa_id]
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                if pred_ans is None and not ground_truths:
                    f1 += 1
                elif pred_ans is not None and ground_truths:
                    f1 += f1_score(pred_ans, ground_truths)

        test_losses.update(loss.data.item(), input_ids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    f1 = f1 / len(qas)
    if f1 >= opt.f1_sco:
        count += 1
    print("| end of epoch {:3d} | f1 score {:5.2f}".format(e+1, f1 * 100))
    print("-" * 89)
    if count >= opt.patience:
        train_success = True
        break
recorder.finish(e, f1, opt.f1_sco, train_success)
torch.cuda.empty_cache()