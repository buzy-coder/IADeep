import shutil, os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import string
import re
from collections import Counter



def load_checkpoint(*args):
    if len(args) == 2:
        filename = os.path.join(args[0], args[1])
    else:
        filename = args[0]
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state


def save_checkpoint(check_path, filename, state, is_best = 0):
    if not os.path.isdir(check_path):
        os.makedirs(check_path, 0o777)
    bestFilename = os.path.join(check_path, 'best_' + filename)
    filename = os.path.join(check_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True

def is_chinese_char(char):


    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2B81F) or  # #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truths):
    f1s_for_ground_truths = []
    for ground_truth in ground_truths:      
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        
        num_same = sum(common.values())
        if num_same == 0:
            f1s_for_ground_truths.append(0)
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1s_for_ground_truths.append(f1)

    return max(f1s_for_ground_truths)


def exact_match_score(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:      
        score = (normalize_answer(prediction) == normalize_answer(ground_truth))
        scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)