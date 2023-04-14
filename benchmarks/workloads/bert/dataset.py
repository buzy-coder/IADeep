import json
import torch.utils.data as data
import torch

def read_squad(path, split = 'train'):
   
    contexts  = []
    questions = []
    question_ids = []
    answers   = []

    with open(path) as f:
        data = json.load(f)

    for article_id in range(len(data['data'])):
        paragraphs = data['data'][article_id]['paragraphs']
        for paragraph in paragraphs:
            context   = paragraph['context']
            qas       = paragraph['qas']
            for qa in qas:
                question = qa['question']
                question_id = qa['id']
                if split == 'train':
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        question_ids.append(question_id)
                        answers.append(answer) 
                    if not qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        question_ids.append(question_id)
                        answers.append(None)
                else:
        
                    contexts.append(context)
                    questions.append(question)
                    question_ids.append(question_id)
                    if qa['answers']:
                        answer = qa['answers'][0]
                        answers.append(answer) 
                    else:
                        answers.append(None)

    return contexts, questions, answers, question_ids


def adjust_start_idx(contexts, answers):
    for context, answer in zip(contexts, answers):
        if answer is not None:
            ans_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(ans_text)

            if context[start_idx:end_idx] == ans_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == ans_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1     
            elif context[start_idx-2:end_idx-2] == ans_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2     

class SquadDataset(data.Dataset):
    def __init__(self, tokenizer, opt, split = 'train'):
        self.split = split
        if split == 'train':
            self.datapath = '/workspace/dataset/squad-2/SQuAD2/train-v2.0.json'
        elif split == "valid":
            self.datapath = "/workspace/dataset/squad-2/SQuAD2/dev-v2.0.json"
    
        self.tokenizer =  tokenizer

        contexts, questions, answers, self.ids = read_squad(self.datapath, split)

        adjust_start_idx(contexts, answers)
        # print(contexts[1],questions[1],answers[1])
        if opt.model == "xlnet":
            self.encodings = self.tokenizer(contexts, questions, truncation=True, padding=True, max_length = 1024)
        else:
            self.encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)

        self.cal_token_positions(answers)

        # print(self.encodings["start_positions"][1],self.encodings["end_positions"][1])
        # print(self.tokenizer.decode(self.encodings["input_ids"][1][self.encodings["start_positions"][1]:self.encodings["end_positions"][1]]))
        # print(len(self.encodings.input_ids))
        # print(len(contexts))
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["ids"] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

    def cal_token_positions(self, answers):
        start_positions = []
        end_positions = []
        ans_exists = []
        for i in range(len(answers)):
            if answers[i] is not None:
                start_position = self.encodings.char_to_token(i, answers[i]['answer_start'])
                end_position = self.encodings.char_to_token(i, answers[i]['answer_end'] - 1)
                ans_exists.append(1)
                # print(i, answers[i]['answer_start'], start_position)
                # print(i, answers[i]['answer_end'], end_position)
            else:
                start_position = None
                end_position = None
                ans_exists.append(0)
             
            # if start position is None, the answer passage has been truncated
            if start_position is not None:
                start_positions.append(start_position)
            else:
                start_positions.append(511)

            if end_position is not None:
                end_positions.append(end_position + 1)
            else:
                end_positions.append(511)
        
        self.encodings.update({'start_positions': start_positions, 'end_positions': end_positions, 'ans_exists': ans_exists})
