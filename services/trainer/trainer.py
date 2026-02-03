"""
Run: python3 -m services.trainer.trainer
"""

import os
from datetime import datetime

import pandas as pd
from ast import literal_eval
import torch
import numpy as np
import random

from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
from torch.amp import GradScaler, autocast

import neptune
from tqdm.auto import tqdm
import configparser

from services.trainer.reinit_model_weights import ModelWithRandomizingSomeWeights
from services.trainer.datasets import TripletDataset, NTXentDataset, PairsDataset
from services.udpipe_model import UDPipeModel
from services.word_sense_detector import WordSenseDetector
from services.utils_results import prediction_accuracy
from services.poolings import PoolingStrategy
from services.prediction_strategies import PredictionStrategy
from services.trainer.utlis import report_gpu
from services.trainer.utlis import AverageMeter
from services.trainer.losses import TripletLoss, NTXentLoss, MNRLoss

from dotenv import load_dotenv

load_dotenv()

import warnings

# warnings.simplefilter('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.manual_seed(47)
random.seed(92)
np.random.seed(39)


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        self.backbone = backbone  # This is your AutoModel
        self.hidden_dim = backbone.config.hidden_size

        # SimCLR/NT-Xent standard: MLP with ReLU
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, projection_dim),
        )

    def forward(self, input_ids, attention_mask):
        # 1. Get Transformer outputs
        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        # 2. Extract Mean Pooling (or token-level pooling)
        # Using your existing mean pooling logic here
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

        # 3. Project to the contrastive space
        projected = self.projection_head(pooled)
        return projected


# TODO: add logging to the class Trainer
class Trainer:
    def __init__(self, config: configparser.ConfigParser):
        # TODO: i think that a lot of the following code should be move to separate file
        self.config = config

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        self._init_logger()

        if not os.path.isdir(
            self.config["MODEL_TUNING"]["path_to_save_fine_tuned_model"]
        ):
            os.mkdir(self.config["MODEL_TUNING"]["path_to_save_fine_tuned_model"])

        self._init_model()

        self._init_loss()
        self._init_datasets_and_loaders()

        self.train_avg_meter = AverageMeter("train_loss")
        self.max_wsd_acc = 0
        self.rounds_count = 0

        # TODO: i'd like to have better config parsing. It takes too much space
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.getfloat("MODEL_TUNING", "learning_rate"),
        )

        if self.apply_warmup:
            total_steps = len(self.train_loader) * self.config.getint(
                "MODEL_TUNING", "num_epochs"
            )
            warmup_ratio = self.config.getfloat(
                "MODEL_TUNING", "warmup_ratio", fallback=None
            )
            if warmup_ratio is None:
                warmup_ratio = self.config.getfloat(
                    "MODEL_TUNING", "warmup_ratio", fallback=0.0
                )
            warmup_steps = min(int(warmup_ratio * total_steps), total_steps)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optim,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

    def _init_logger(self):
        self.apply_warmup = self.config.getboolean("MODEL_TUNING", "apply_warmup")

        self.log_to_neptune = self.config.getboolean("MODEL_TUNING", "log_to_neptune")
        if self.log_to_neptune:
            # TODO: move to config
            neptune_project_name = self.config["MODEL_TUNING"]["neptune_project_name"]
            self.run = neptune.init_run(project=neptune_project_name)
            self.run["epochs"] = self.config.getint("MODEL_TUNING", "num_epochs")
            self.run["batch_size"] = self.config.getint("MODEL_TUNING", "batch_size")
            self.run["learning_rate"] = self.config.getfloat(
                "MODEL_TUNING", "learning_rate"
            )
            self.run["early_stopping"] = self.config.getint(
                "MODEL_TUNING", "early_stopping"
            )
            # self.run["dataset/diff_threshold"] = 0.3
            self.run_id = self.run["sys/id"].fetch().split("/")[-1][4:]
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _init_model(self):
        self.udpipe_model = UDPipeModel(
            self.config["MODEL_TUNING"]["path_to_udpipe_model"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["MODEL_TUNING"]["tokenizer_name"], trust_remote_code=True
        )

        base_model = AutoModel.from_pretrained(
            self.config["MODEL_TUNING"]["model_to_fine_tune"], output_hidden_states=True
        )

        self.model = base_model

        layers_to_unfreeze = self.config["MODEL_TUNING"].getint(
            "layers_to_unfreeze", fallback=0
        )
        if layers_to_unfreeze > 0:
            # freeze model weights except n last layers and pooler
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.encoder.layer[-layers_to_unfreeze:].parameters():
                param.requires_grad = True

            for param in self.model.pooler.parameters():
                param.requires_grad = True

        # self.model = None

        # if self.config.get('MODEL_TUNING', 'loss') == 'nt_xent_loss':
        #     self.model = ContrastiveModel(self.model)
        # else:
        #     self.model = self.model

        self.model = self.model.to(self.device)
        # if self.config.getboolean('MODEL_TUNING', 'enable_gpu_parallel'):
        #     self.model = nn.DataParallel(self.model)

        # if self.config.getboolean('MODEL_TUNING', 'random_model_weights_reinitialization'): # TODO: for now we don't apply it
        #     self.model = ModelWithRandomizingSomeWeights(
        #         model=self.model,
        #         reinit_n_layers=self.config.getint('MODEL_TUNING', 'number_of_layers_for_reinitialization')
        #     ).to(self.device)

    def _init_loss(self):
        loss_type = self.config.get("MODEL_TUNING", "loss")
        if loss_type == "triplet_loss":
            self.loss = TripletLoss(
                model=self.model,
                margin=0.2,
                p=2,
                pool_targets=self.config.getboolean("MODEL_TUNING", "pool_targets"),
                use_both_poolings=self.config.getboolean(
                    "MODEL_TUNING", "use_both_poolings"
                ),
            )
        elif loss_type == "nt_xent_loss":
            self.loss = NTXentLoss(model=self.model, temperature=0.05)
        elif loss_type == "mnr_loss":
            self.loss = MNRLoss(
                model=self.model,
                temperature=0.05,
                pool_targets=self.config.getboolean("MODEL_TUNING", "pool_targets"),
                use_both_poolings=self.config.getboolean(
                    "MODEL_TUNING", "use_both_poolings"
                ),
            )
        else:
            raise NotImplementedError(f"Loss {loss_type} is not implemented yet")

    def _init_datasets_and_loaders(self):
        self.wsd_eval_data = self._load_wsd_eval_dataset()
        self.train_data, self.eval_data = self._load_train_eval_datasets()

        loss_type = self.config.get("MODEL_TUNING", "loss")
        num_workers = self.config.getint("MODEL_TUNING", "num_workers", fallback=4)
        prefetch_factor = self.config.getint(
            "MODEL_TUNING", "prefetch_factor", fallback=2
        )
        drop_last = loss_type == "nt_xent_loss"

        collate_fn = None
        pool_targets = self.config.getboolean(
            "MODEL_TUNING", "pool_targets"
        ) or self.config.getboolean("MODEL_TUNING", "use_both_poolings")

        if loss_type == "triplet_loss":
            self.train_dataset = TripletDataset(
                anchor=self.train_data["anchor"].values,
                positive=self.train_data["positive"].values,
                negative=self.train_data["negative"].values,
                tokenizer=self.tokenizer,
                pool_targets=pool_targets,
                anchor_target_word_ids=self.train_data.get(
                    "anchor_target_word_ids", None
                ),
                positive_target_word_ids=self.train_data.get(
                    "positive_target_word_ids", None
                ),
                negative_target_word_ids=self.train_data.get(
                    "negative_target_word_ids", None
                ),
            )

            self.eval_dataset = TripletDataset(
                anchor=self.eval_data["anchor"].values,
                positive=self.eval_data["positive"].values,
                negative=self.eval_data["negative"].values,
                tokenizer=self.tokenizer,
                pool_targets=pool_targets,
                anchor_target_word_ids=self.eval_data.get("anchor_target_word_ids"),
                positive_target_word_ids=self.eval_data.get("positive_target_word_ids"),
                negative_target_word_ids=self.eval_data.get("negative_target_word_ids"),
            )

            if pool_targets:
                collate_fn = lambda batch: {
                    "anchor_ids": torch.stack([item["anchor_ids"] for item in batch]),
                    "anchor_mask": torch.stack([item["anchor_mask"] for item in batch]),
                    "positive_ids": torch.stack(
                        [item["positive_ids"] for item in batch]
                    ),
                    "positive_mask": torch.stack(
                        [item["positive_mask"] for item in batch]
                    ),
                    "negative_ids": torch.stack(
                        [item["negative_ids"] for item in batch]
                    ),
                    "negative_mask": torch.stack(
                        [item["negative_mask"] for item in batch]
                    ),
                    "anchor_target_word_ids": torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.tensor(item["anchor_target_word_ids"])
                            for item in batch
                        ],
                        batch_first=True,
                        padding_value=-1,
                    ),
                    # "positive_target_word_ids": torch.nn.utils.rnn.pad_sequence(
                    #     [
                    #         torch.tensor(item["positive_target_word_ids"]) if "positive_target_word_ids" in item else torch.tensor([])
                    #         for item in batch
                    #     ],
                    #     batch_first=True,
                    #     padding_value=-1,
                    # ),
                    # "negative_target_word_ids": torch.nn.utils.rnn.pad_sequence(
                    #     [
                    #         torch.tensor(item["negative_target_word_ids"]) if "negative_target_word_ids" in item else torch.tensor([])
                    #         for item in batch
                    #     ],
                    #     batch_first=True,
                    #     padding_value=-1,
                    # ),
                }
        elif loss_type == "mnr_loss":
            self.train_dataset = PairsDataset(
                anchor=self.train_data["anchor"].values,
                positive=self.train_data["positive"].values,
                tokenizer=self.tokenizer,
                pool_targets=pool_targets,
                anchor_target_word_ids=self.train_data.get("anchor_target_word_ids"),
            )

            self.eval_dataset = PairsDataset(
                anchor=self.eval_data["anchor"].values,
                positive=self.eval_data["positive"].values,
                tokenizer=self.tokenizer,
                pool_targets=pool_targets,
                anchor_target_word_ids=self.eval_data.get("anchor_target_word_ids"),
            )

            if pool_targets:
                collate_fn = lambda batch: {
                    "anchor_ids": torch.stack([item["anchor_ids"] for item in batch]),
                    "anchor_mask": torch.stack([item["anchor_mask"] for item in batch]),
                    "positive_ids": torch.stack(
                        [item["positive_ids"] for item in batch]
                    ),
                    "positive_mask": torch.stack(
                        [item["positive_mask"] for item in batch]
                    ),
                    "anchor_target_word_ids": torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.tensor(item["anchor_target_word_ids"])
                            for item in batch
                        ],
                        batch_first=True,
                        padding_value=-1,
                    ),
                }
        elif loss_type == "nt_xent_loss":
            self.train_dataset = NTXentDataset(
                view1=self.train_data["view1"].values,
                view2=self.train_data["view2"].values,
                tokenizer=self.tokenizer,
            )

            self.eval_dataset = NTXentDataset(
                view1=self.eval_data["view1"].values,
                view2=self.eval_data["view2"].values,
                tokenizer=self.tokenizer,
            )
        else:
            raise NotImplementedError(
                f"Dataset for loss {loss_type} is not implemented yet"
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.getint("MODEL_TUNING", "batch_size"),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,  # Add this
            prefetch_factor=prefetch_factor,  # Add this
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

        self.eval_loader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.getint("MODEL_TUNING", "batch_size"),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,  # Add this
            prefetch_factor=prefetch_factor,  # Add this
            collate_fn=collate_fn,
            drop_last=False,
        )

        if self.log_to_neptune:
            self.run["dataset/train"] = len(self.train_data)
            self.run["dataset/eval"] = len(self.eval_data)

    def _load_wsd_eval_dataset(self):
        wsd_eval_data = pd.read_csv(
            self.config["MODEL_TUNING"]["path_to_wsd_eval_dataset"]
        )
        wsd_eval_data["examples"] = wsd_eval_data["examples"].apply(literal_eval)
        wsd_eval_data["gloss"] = wsd_eval_data["gloss"].apply(
            lambda x: literal_eval(x) if x.startswith("[") else [x]
        )
        return wsd_eval_data

    def _load_train_eval_datasets(self):
        data = pd.read_csv(self.config["MODEL_TUNING"]["path_to_triplet_dataset"])
        data = data.sample(frac=1)
        train_data = data[: int(len(data) * 0.99)]
        eval_data = data[int(len(data) * 0.99) :]

        return train_data, eval_data

    def _calculate_wsd_accuracy(self, eval_data):
        word_sense_detector = WordSenseDetector(
            pretrained_model=self.model,
            udpipe_model=self.udpipe_model,
            evaluation_dataset=eval_data,
            tokenizer=self.tokenizer,
            pooling_strategy=PoolingStrategy.mean_pooling,
            prediction_strategy=PredictionStrategy.all_examples_to_one_embedding,
        )

        eval_data = word_sense_detector.run()
        return prediction_accuracy(eval_data)

    def _save_model(self, model, path_to_save_model):
        try:
            if isinstance(model, torch.nn.DataParallel):
                model.module.save_pretrained(path_to_save_model, from_pt=True)
            else:
                model.save_pretrained(path_to_save_model, from_pt=True)
        except Exception as e:
            print(f"model not saved, error = {e}")

    @torch.no_grad()
    def evaluate_epoch(self, epoch, batch_count):
        self.model.eval()

        eval_loss = 0
        eval_bar = tqdm(self.eval_loader, leave=True, desc="Triplet Eval")
        for eval_batch in eval_bar:
            with autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                eval_loss += self.loss(eval_batch)

        if self.log_to_neptune:
            self.run["eval/loss"].append(eval_loss / len(self.eval_loader))

        with autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            wsd_acc = self._calculate_wsd_accuracy(self.wsd_eval_data)

        report_gpu()

        if self.log_to_neptune:
            self.run["eval/wsd_acc"].append(wsd_acc)

        if wsd_acc > self.max_wsd_acc:
            self.max_wsd_acc = wsd_acc
            self.rounds_count = 0

            if batch_count > 0:
                # TODO: model won't be save if neptune is false
                self._save_model(
                    self.model,
                    f"{self.config['MODEL_TUNING']['path_to_save_fine_tuned_model']}/model_{self.run_id}_{epoch}",
                )

        elif wsd_acc <= self.max_wsd_acc:
            self.rounds_count += 1

        print(
            {
                "epoch": epoch,
                "batch_count": batch_count,
                "wsd_acc": wsd_acc,
                "max_wsd_acc": self.max_wsd_acc,
                "rounds_count": self.rounds_count,
            }
        )

        if self.rounds_count >= self.config.getint("MODEL_TUNING", "early_stopping"):
            print(
                f'Early stopping, model not improve WSD for {self.config.getint("MODEL_TUNING", "early_stopping")}'
            )
            return True

        return False

    def train_epoch(self, epoch):
        train_bar = tqdm(self.train_loader, leave=True, desc=f"Train epoch: {epoch}")

        for batch_count, batch in enumerate(train_bar):
            self.model.train()
            self.optim.zero_grad()

            with autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                loss = self.loss(batch).to(self.device)

            self.scaler.scale(loss).backward()
            max_grad_norm = self.config.getfloat(
                "MODEL_TUNING", "max_grad_norm", fallback=0.0
            )
            if max_grad_norm and max_grad_norm > 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.scaler.step(self.optim)
            self.scaler.update()

            # self.optim.zero_grad()
            # loss.backward()

            # self.optim.step()
            self.train_avg_meter.update(
                loss.item(), self.config.getint("MODEL_TUNING", "batch_size")
            )  # TODO: .detach().cpu()?

            if self.apply_warmup:
                self.scheduler.step()

            if self.log_to_neptune:
                acc_val, acc_avg = self.train_avg_meter()
                self.run["train/loss"].append(acc_avg)
                self.run["train/lr"].append(self.optim.param_groups[0]["lr"])

            # report_gpu() # TODO: it's interesting do we really need it

            if (
                batch_count > 0
                and batch_count
                % self.config.getint("MODEL_TUNING", "num_batch_to_eval")
                == 0
            ):
                if self.evaluate_epoch(epoch, batch_count):
                    return True  # reach early stopping rounds

    def train(self):
        try:
            # initial evaluation of the raw model
            for epoch in range(self.config.getint("MODEL_TUNING", "num_epochs")):
                early_stop = self.evaluate_epoch(epoch=epoch, batch_count=0)
                report_gpu()

                if not early_stop:
                    early_stop = self.train_epoch(epoch)

                if early_stop:
                    path_to_save_model = self.config["MODEL_TUNING"][
                        "path_to_save_fine_tuned_model"
                    ]
                    model_name = f"{path_to_save_model}/model_{self.run_id}_{epoch}_early_stopped"
                    self._save_model(self.model, model_name)
                    break

                path_to_save_model = self.config["MODEL_TUNING"][
                    "path_to_save_fine_tuned_model"
                ]
                model_name = f"{path_to_save_model}/model_{self.run_id}_{epoch}"
                self._save_model(self.model, model_name)
        finally:
            path_to_save_model = self.config["MODEL_TUNING"][
                "path_to_save_fine_tuned_model"
            ]
            model_name = f"{path_to_save_model}/model_{self.run_id}_final"
            self._save_model(self.model, model_name)

            if self.log_to_neptune:
                self.run.stop()


if __name__ == "__main__":
    import configparser

    config = configparser.ConfigParser()
    config.read("services/trainer/fine_tuning_config.ini")

    model_trainer = Trainer(config)
    model_trainer.train()
