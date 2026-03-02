#! /usr/bin/env python3
"""
export SCRIPT_NAME=label_extraction
python scripts/${SCRIPT_NAME}.py
"""
import hydra
import os
import textwrap
import warnings
from PIL import Image
from dotenv import load_dotenv
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

# rtk
from rtk.datasets import MIMIC_CLASS_NAMES
from rtk.metrics import generate_classification_report, METRICS_DIR
from rtk.utils import get_console, get_logger, intro

get_logger("mlflow").setLevel("ERROR")
MLFLOW_TRACKING_URI = "http://localhost:5000"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pretty.install()
console: Console = get_console()
logger = get_logger(
    "".join(["chexagent.scripts.", __file__.split("/")[-1].split(".")[0], f".main"])
)
# get_logger("mlflow").setLevel("DEBUG")


def log_params(args: DictConfig):
    params = OmegaConf.to_container(args, resolve=True)
    mlflow.log_params(params)
    return params


def log_results(data: DatasetDict, args: DictConfig):
    positive_class = args.positive_class
    for split, data in data.items():
        logger.info(f"Logging '{split}' results...")
        # os.makedirs("data", exist_ok=True)
        source = HuggingFaceDatasetSource(path=args.dataset, split=split)
        mlflow_ds = HuggingFaceDataset(
            ds=data,
            source=source,
            name="".join([args.dataset.split("/")[-1], f":{split}"]),
        )
        mlflow.log_input(mlflow_ds, context="binary_prediction")
        # for col in args.target_columns:
        y_true = list(data[positive_class])
        y_true = [1 if _ == 1 else 0 for _ in y_true]
        new_col = f"{positive_class}-{args.model_id.split('/')[-1]}"
        y_pred = list(data[new_col])
        summary_dict = generate_classification_report(
            y_true,
            y_pred,
            split=split,
            target_names=[f"No {positive_class}", f"{positive_class}"],
            log=True,
        )
        mlflow.log_metrics(summary_dict, model_id=args.model_id, step=0)

    # return new_data


@hydra.main(config_path="../configs", config_name="chexagent", version_base="1.1")
def main(args: DictConfig):
    intro(args, "CheXagent Label Extraction")
    load_dotenv(args.get("env_file", None))
    name: str = args.get("name", "label-extraction")
    prefix: str = args.get("prefix", "-test") if args.dry_run else ""
    positive_class: str = args.get("positive_class")
    # target_columns = args.get("target_columns", [args.positive_class])

    logger.info(f"Loading dataset: '{args.dataset}'")
    data: DatasetDict = load_dataset(args.dataset)
    if args.dry_run:
        data.pop("train")

    # columns_to_remove = list(
    #     filter(lambda x: x not in target_columns, MIMIC_CLASS_NAMES)
    # )
    # data = data.remove_columns(columns_to_remove)
    data["test"] = data["test"].filter(lambda x: x["gt"])

    # pipeline testing
    if args.dry_run:
        data["validate"] = data["validate"].filter(lambda x: x["gt"])
        data["test"] = (
            data["test"].shuffle(seed=42).select(range(len(data["validate"]) // 2))
        )

    console.print(data)
    logger.info(f"Dataset loaded. Loading '{args.model_id}' model")

    # Silence attention mask warning
    import transformers

    transformers.logging.set_verbosity_error()
    model = CheXagent(model_name=args.model_id)
    yes_no = {"yes": 1, "no": 0}
    model_name: str = args.model_id.split("/")[-1]

    def get_response_with_report(example: dict):
        image_file = os.path.join(args.data_dir, example["image_files"])

        # for col in target_columns:
        if example["findings"] or example["impressions"]:
            text = " ".join([example["findings"], example["impressions"]]).strip()
        else:
            text = example["reports"].strip()
        prompt = textwrap.dedent(
            f"""
            Consider the following chest X-ray description:

            {text}

            Does this chest X-ray contain a {positive_class}?
            """
        ).strip()
        response = model.generate([image_file], prompt).lower().strip()
        debug_info = {
            "prompt": prompt,
            "response": response,
            "dicom_id": example["dicom_id"],
        }
        logger.debug(debug_info)
        new_col = f"{positive_class}-{model_name}"
        try:
            y_pred = yes_no[response]
            example[new_col] = y_pred
        except KeyError as e:
            logger.error(f"Invalid response returned. See: {debug_info}")
            example[new_col] = 0

        return example

    with_rank = args.get("with_rank", False)
    num_proc = torch.cuda.device_count() if with_rank else None

    # mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = f"{name}{prefix}"
    run_name = f"{name}-{args.date}-{args.timestamp}"
    experiment = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=run_name, experiment_id=experiment.experiment_id
    ) as run:
        mlflow.autolog()
        log_params(args)
        with console.status(
            f"\t[bold]Getting binary labels for [green]'{positive_class}'[/green][/bold]\n",
            spinner="arrow3",
        ):
            with torch.inference_mode():
                labelled_ds = data.map(
                    get_response_with_report,
                    batched=args.get("batched", False),
                    batch_size=args.get("batch_size", None),
                    with_rank=with_rank,
                    num_proc=num_proc,
                    desc="In progress",
                )

        logger.info(f"Inference complete. Logging results to mlflow")
        log_results(labelled_ds, args)
        mlflow.log_artifacts(".", artifact_path="outputs")

        push_to_hub = args.get("push_to_hub", False)
        if push_to_hub or not args.dry_run:
            logger.info(
                "Label extraction results saved to mlflow. Pushing results to 🤗"
            )
            from huggingface_hub import HfApi

            api = HfApi()
            repo = args.get(
                "repo", "nclgbd" if args.dry_run else "vllm-pneumonia-detection"
            )
            dataset_name = f"mimic-cxr-labelled-dataset{prefix}"
            path = f"{repo}/{dataset_name}"
            # Check if repo exists
            if api.list_datasets(search=dataset_name, author=repo):
                cur_dataset = load_dataset(path)
                new_col = f"{positive_class}-{model_name}"
                if new_col in cur_dataset.column_names["validate"]:
                    logger.info(
                        f"Column '{new_col}' already exists in dataset. Overwriting..."
                    )
                    cur_dataset = cur_dataset.remove_columns([new_col, positive_class])
                for split_name, ds_split in cur_dataset.items():
                    cur_dataset[split_name] = ds_split.add_column(
                        name=positive_class,
                        column=labelled_ds[split_name][positive_class],
                    )
                    cur_dataset[split_name] = cur_dataset[split_name].add_column(
                        name=new_col,
                        column=labelled_ds[split_name][new_col],
                    )
                labelled_ds = cur_dataset

            logger.info(f"Pushing to hub: '{path}'...")
            commit = labelled_ds.push_to_hub(
                path,
                private=True,
                create_pr=True,
                commit_description=f"`mlflow.run_id`: '{run.info.run_id}'",
            )
            try:
                revision = commit.pr_revision
                api.upload_folder(
                    folder_path=".",
                    path_in_repo="outputs",
                    repo_id=path,
                    repo_type="dataset",
                    revision=revision,
                )
                tags = commit.__dict__
                mlflow.set_tags(tags)
                logger.info(f"Commit url: {commit}")
            except Exception as e:
                logger.error(e)
                api.delete_branch(repo_id=path, branch=revision, repo_type="dataset")

        else:
            logger.info(f"Not pushing to hub. Exiting...")


if __name__ == "__main__":
    main()
