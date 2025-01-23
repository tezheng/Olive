import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    BertModel,
)

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


def create_4d_mask(mask, input_shape):
    batch_sz, seq_len = input_shape
    expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.float()
    return inverted_mask.masked_fill(inverted_mask.bool(), -50.0)


@Registry.register_pre_process()
def bert_pre_process_fast(
    dataset, model_name, input_cols, label_col="label",
    seq_length=512, max_samples=4, **kwargs
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    embeddings = model.embeddings

    def generate_inputs(sample, indices):
        encoded_input = tokenizer(
            *[sample[input_col] for input_col in input_cols],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        inputs_embeds = embeddings(
            encoded_input.input_ids,
            token_type_ids=encoded_input.get("token_type_ids", None)
        )
        attention_mask = create_4d_mask(
            encoded_input.attention_mask,
            encoded_input.input_ids.shape,
        )
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            label_col: sample.get(label_col, indices)
        }

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))

    with torch.inference_mode():
        tokenized_datasets = dataset.select(range(max_samples)).map(
            generate_inputs,
            batched=True,
            with_indices=True,
            remove_columns=dataset.column_names,
        )
        tokenized_datasets.set_format("torch", output_all_columns=True)

    # return BaseDataset(tokenized_datasets, label_col)
    return BaseDataset(list(tokenized_datasets), label_col)


@Registry.register_pre_process()
def bert_pre_process_wo_dataset(
    dataset, model_name, input_cols, label_col="label",
    seq_length=512, max_samples=4, **kwargs
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    embeddings = model.embeddings

    def generate_inputs(sample):
        encoded_input = tokenizer(
            *[sample[input_col] for input_col in input_cols],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        inputs_embeds = embeddings(
            encoded_input.input_ids,
            token_type_ids=encoded_input.get("token_type_ids", None)
        )
        attention_mask = create_4d_mask(
            encoded_input.attention_mask,
            encoded_input.input_ids.shape,
        )
        return {
            "inputs_embeds": inputs_embeds[0,],
            "attention_mask": attention_mask[0,],
        }

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))

    with torch.inference_mode():
        data = [{
            **generate_inputs(sample),
            label_col: sample.get(label_col, torch.tensor(idx)),
        } for idx, sample in enumerate(dataset.select(range(max_samples)))]

    return BaseDataset(data, label_col)


@Registry.register_pre_process()
def bert_pre_process_slow(
    dataset, model_name, input_cols, label_col="label",
    seq_length=512, sample_count=4, **kwargs
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    target = model.encoder

    from inspect import signature
    params = [n for (n, _) in signature(target.forward).parameters.items()]

    inputs = {}

    def capture_inputs(_, args, kwargs) -> None:
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                inputs[params[i]] = arg[0,].clone()
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value[0,].clone()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.clamp(inputs["attention_mask"],
                                                   min=-50, max=0)

    def inference(examples, idx):
        inputs.clear()
        encoded_input = tokenizer(
            *[examples[input_col] for input_col in input_cols],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.inference_mode():
            model(**encoded_input)
        inputs[label_col] = torch.tensor(idx)
        return inputs.copy()

    target.register_forward_pre_hook(capture_inputs, with_kwargs=True)

    data = [inference(item, idx) for idx, item in
            enumerate(dataset.select(range(sample_count)))]

    return BaseDataset(data, label_col)


@Registry.register_post_process()
def bert_qa_post_process(outputs, **kwargs):
    logits = [outputs["start_logits"], outputs["end_logits"]]
    return torch.stack(logits, dim=1)


@Registry.register_post_process()
def bert_scl_post_process(outputs, **kwargs):
    return outputs.argmax(dim=-1)
