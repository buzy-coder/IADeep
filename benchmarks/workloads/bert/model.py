
from transformers import BertPreTrainedModel, BertModel, XLNetModel, XLNetPreTrainedModel
from torch import nn

class BertForSQuAD(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
        ans_exists=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
       
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        # print(sequence_output.shape, pooled_output.shape)
        ans_logit = self.classifier(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fuction = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fuction(start_logits, start_positions)
            end_loss = loss_fuction(end_logits, end_positions)
            # print(ans_logit.shape,ans_exists.shape,end_logits.shape,end_positions.shape)
            ansexist_loss = loss_fuction(ans_logit.view(-1, 2), ans_exists.view(-1))
            total_loss = (start_loss + end_loss + ansexist_loss) / 3

        return {"loss":total_loss,
            "start_logits":start_logits, 
            "end_logits":end_logits,
            "ans_logits":ans_logit}

      
class XLNetForSQuAD(XLNetPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
        ans_exists=None,
    ):

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
       
        pooled_output = outputs[0][:,0]
        # pooled_output = self.dropout(pooled_output)
        # print(sequence_output.shape, pooled_output.shape)
        ans_logit = self.classifier(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fuction = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fuction(start_logits, start_positions)
            end_loss = loss_fuction(end_logits, end_positions)
            # print(ans_logit.shape,ans_exists.shape,end_logits.shape,end_positions.shape)
            ansexist_loss = loss_fuction(ans_logit.view(-1, 2), ans_exists.view(-1))
            total_loss = (start_loss + end_loss + ansexist_loss) / 3

        return {"loss":total_loss,
            "start_logits":start_logits, 
            "end_logits":end_logits,
            "ans_logits":ans_logit}
