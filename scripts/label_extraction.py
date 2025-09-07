#! /usr/bin/env python3
""" """
import hydra
import os
import textwrap
import warnings
from PIL import Image
from dotenv import load_dotenv
from multiprocess import set_start_method
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print, traceback
from rich.console import Console
from rich.markdown import Markdown

# huggingface
import torch
from datasets import load_dataset, DatasetDict, Dataset

# mlflow
import mlflow
from mlflow.data.huggingface_dataset import HuggingFaceDataset, HuggingFaceDatasetSource

# chexagent
from model_chexagent.chexagent import CheXagent

MLFLOW_TRACKING_URI = "http://localhost:5000"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pretty.install()
console = Console()


def log_params(args: DictConfig):
    params = OmegaConf.to_container(args, resolve=True)
    mlflow.log_params(params)
    return params


def generate_classification_report(
    y_true,
    y_pred,
    target_names=[f"No Pneumonia", "Pneumonia"],
    split: str = "validate",
    results_dir: str = "results",
):
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    from tabulate import tabulate

    os.makedirs(results_dir, exist_ok=True)

    # Confusion matrix
    console.print(Markdown(f"## Confusion Matrix for '{split}'"))
    cfm = confusion_matrix(y_true, y_pred)
    if cfm.shape[0] < 2:
        if y_true[0] == 0:
            cfm = [[cfm[0][0], 0], [0, 0]]
        else:
            cfm = [[0, 0], [0, cfm[0][0]]]
    tbl = tabulate(
        cfm, headers=target_names, showindex=target_names, tablefmt="rounded_grid"
    )
    print(tbl)
    cfm_df = pd.DataFrame(cfm, index=target_names, columns=target_names)
    cfm_df.to_csv(f"{results_dir}/{split}_confusion_matrix.csv")

    # Classification report
    cr_string = classification_report(
        y_true, y_pred, labels=target_names, target_names=target_names
    )
    cr = classification_report(
        y_true, y_pred, labels=target_names, target_names=target_names, output_dict=True
    )
    console.print(Markdown(f"## Classification Report for '{split}'"))
    print(cr_string)
    metrics = {
        f"{split}_f1": round(cr["macro avg"]["f1-score"], 4),
        f"{split}_sensitivity": round(cr["Pneumonia"]["recall"], 4),
        f"{split}_specificity": round(cr["No Pneumonia"]["recall"], 4),
        f"{split}_recall": round(cr["macro avg"]["recall"], 4),
        f"{split}_precision": round(cr["macro avg"]["precision"], 4),
        f"{split}_accuracy": round(cr.get("accuracy", 0.0), 4),
    }
    summary = pd.DataFrame(
        metrics,
        index=["chexagent"],
    )
    summary.to_csv(f"{results_dir}/{split}_classification_summary.csv")
    mlflow.log_metrics(metrics, step=0)


def log_results(ds: Dataset, args: DictConfig, log_test=False):
    data = []
    split_names = []

    if args.dry_run:
        split_names = ["validate"]
        data.append(ds)
    else:
        split_names = ["train", "validate"]
        train_ds = ds.filter(lambda x: x["split"] == "train")
        data.append(train_ds)
        val_ds = ds.filter(lambda x: x["split"] == "validate")
        data.append(val_ds)
        if log_test:
            split_names.append("test")
            test_ds = ds.filter(lambda x: x["split"] == "test")
            data.append(test_ds)

    for split, ds in zip(split_names, data):
        console.log(f"Logging '{split}' results...")
        y_true = ds[args.positive_class]
        y_true = [1 if _ == 1 else 0 for _ in y_true]
        y_pred = ds["chexagent"]
        generate_classification_report(
            y_true,
            y_pred,
            split=split,
        )
        os.makedirs("data", exist_ok=True)
        source = HuggingFaceDatasetSource(path=args.dataset, split=split)
        mlflow_ds = HuggingFaceDataset(
            ds=ds,
            source=source,
            name="".join([args.dataset.split("/")[-1], f":{split}"]),
        )
        mlflow.log_input(mlflow_ds, context="binary_prediction")

    data_dict = dict(zip(split_names, data))
    new_ds = DatasetDict(data_dict)
    return new_ds


@hydra.main(config_path="../configs", config_name="chexagent", version_base="1.1")
def main(args: DictConfig):
    console.clear()
    config_str = OmegaConf.to_yaml(args, resolve=True)
    console.print(Markdown("## Configuration\n\n"))
    config_str = textwrap.dedent(
        f"""
        ```yaml
{config_str}
        """
    ).strip()
    console.print(Markdown(config_str))

    console.log(f"Loading dataset: '{args.dataset}'")
    ds = load_dataset(
        args.dataset, split="validate" if args.dry_run else "train+validate+test"
    )
    ds = ds.remove_columns(
        [
            "multiclass_labels",
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Effusion",
            "Pleural Other",
            "Pneumothorax",
            "Support Devices",
        ]
    )

    # pipeline testing
    # if args.dry_run:
    #     ds = ds.shuffle(42).select(range(10))

    console.log(f"Dataset loaded. Loading chexagent model:")
    model = CheXagent(model_name=args.model_id, dtype=torch.float32)
    yes_no = {"yes": 1, "no": 0}

    def get_response_with_report(example: dict):
        image_file = os.path.join(args.data_dir, example["image_files"])
        if example["findings"] or example["impressions"]:
            text = " ".join([example["findings"], example["impressions"]])
        else:
            text = example["reports"].strip()

        prompt = textwrap.dedent(
            f"""
            Consider the following chest X-ray description:

            {text}

            Does this chest X-ray contain a {args.positive_class}?
            """
        )
        response = model.generate([image_file], prompt).lower().strip()
        y_pred = yes_no[response]
        example["chexagent"] = y_pred
        return example

    console.log("Creating CheXagent labels....")
    with_rank = args.get("with_rank", False)
    num_proc = torch.cuda.device_count() if with_rank else None

    # mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_name = f"{args.name}-{args.date}-{args.timestamp}"
    experiment = mlflow.set_experiment(args.name)
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        mlflow.autolog()
        log_params(args)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            labelled_ds = ds.map(
                get_response_with_report,
                batched=args.get("batched", False),
                batch_size=args.batch_size,
                with_rank=with_rank,
                num_proc=num_proc,
                desc="In progress",
            )
        new_ds = log_results(labelled_ds, args, log_test=not args.dry_run)
        mlflow.log_artifacts(".", artifact_path="outputs")

        push_to_hub = args.get("push_to_hub", False)
        if push_to_hub or not args.dry_run:
            repo = "nclgbd" if args.dry_run else "vllm-pneumonia-detection"
            dataset_name = "mimic-cxr-labelled-dataset"
            suffix = "-test" if args.dry_run else ""
            path = f"{repo}/{dataset_name}{suffix}"
            mlflow.log_param("dataset_hub_path", path)
            console.log(f"Pushing to hub: '{path}'...")
            commit = new_ds.push_to_hub(
                path,
                private=True,
            )
            print(commit)

        else:
            console.log(f"Not pushing to hub. Exiting...")


if __name__ == "__main__":
    main()
