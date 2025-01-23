
from torch import nn

from transformers import (
    AutoModelForQuestionAnswering as AutoModelForQA,
    AutoModelForSequenceClassification as AutoModelSCL,
    BertModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions as EncoderOutput,
    QuestionAnsweringModelOutput,
)


def load_hf_model_bert(model_name: str) -> nn.Module:
    model = BertModel.from_pretrained(model_name)
    model.eval()
    return model.encoder


class BertQAWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForQA.from_pretrained(model_name)
        self.model.eval()
        self.encoder = self.model.bert.encoder
        self.qa_outputs = self.model.qa_outputs

    def forward(self, hidden_states, attention_mask=None):
        last_hidden_state = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
        )[0]
        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return QuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
        )


class BertWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        model = BertModel.from_pretrained(model_name)
        model.eval()
        self.encoder = model.encoder
        self.pooler = model.pooler

    def forward(self, *args, **kwargs):
        sequence_output = self.encoder(
            hidden_states=kwargs.get("inputs_embeds"),
            attention_mask=kwargs.get("attention_mask"),
        )[0]
        pooled_output = (
            self.pooler(sequence_output)
            if self.pooler is not None
            else None
        )
        return EncoderOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


def load_bert_qa_model(model_name: str) -> nn.Module:
    model = AutoModelForQA.from_pretrained(model_name)
    model.bert = BertWrapper(model_name)
    return model
    # return BertQAWrapper(model_name)


def load_bert_scl_model(model_name: str) -> nn.Module:
    model = AutoModelSCL.from_pretrained(model_name)
    model.bert = BertWrapper(model_name)
    return model
