
from typing import Dict, Union

from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer


def eval_squad(
    outputs,
    targets,
    dataset_config: Dict[str, str],
    model_name: str,
    seq_length: int = 512,
) -> Dict[str, Union[float, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(
        path=dataset_config["data_name"],
        split=dataset_config["split"],
    )

    predictions = []
    references = []

    for pred, i in zip(outputs.preds, targets):
        sample = dataset[i.item()]
        encoded_input = tokenizer(
            sample["question"],
            sample["context"],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        start_logits, end_logits = pred.unbind(dim=0)
        start_index = start_logits.argmax(dim=-1)
        end_index = end_logits.argmax(dim=-1)
        offset_mapping = encoded_input["offset_mapping"]
        answer_start = offset_mapping[:, start_index, 0].squeeze()
        answer_end = offset_mapping[:, end_index, 1].squeeze()
        pred_answer = sample["context"][answer_start:answer_end]

        references.append({
            "id": sample["id"],
            "answers": {
                "answer_start": sample["answers"]["answer_start"],
                "text": sample["answers"]["text"],
            },
        })
        predictions.append({
            "id": sample["id"],
            "prediction_text": pred_answer,
        })

    results = load("squad").compute(
        predictions=predictions,
        references=references,
    )

    return {"f1": results["f1"], "exact_match": results["exact_match"]}
