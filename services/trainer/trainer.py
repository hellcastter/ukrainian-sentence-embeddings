"""
Run: python3 -m services.trainer.trainer
"""

import os
import random
from ast import literal_eval
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup

import wandb
from tqdm.auto import tqdm

from services.trainer.data_factory import DataFactory
from services.trainer.training_config import TrainingConfig

# from services.trainer.reinit_model_weights import ModelWithRandomizingSomeWeights
from services.udpipe_model import UDPipeModel
from services.word_sense_detector import WordSenseDetector
from services.utils_results import prediction_accuracy
from services.poolings import PoolingStrategy
from services.prediction_strategies import PredictionStrategy
from services.trainer.utlis import report_gpu, AverageMeter
from services.trainer.losses import TripletLoss, NTXentLoss, MNRLoss
from services.config import PATH_TO_SOURCE_UDPIPE
from eval.eval_wsd import evaluate_wsd

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
    def __init__(self, config: str):
        # TODO: i think that a lot of the following code should be move to separate file
        self.config = TrainingConfig.from_config(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        self.global_step = 0  # used for wandb logging steps
        self._init_logger()

        if not os.path.isdir(self.config.path_to_save_fine_tuned_model):
            os.mkdir(self.config.path_to_save_fine_tuned_model)

        self._init_model()

        self._init_loss()
        self._setup_data()

        self.train_avg_meter = AverageMeter("train_loss")
        self.max_wsd_acc = 0
        self.rounds_count = 0

        self._setup_optimizer()

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=0.01
        )

        # Warmup logic: needs the total number of training steps
        if self.config.warmup_ratio > 0:
            total_steps = len(self.train_loader) * self.config.num_epochs
            warmup_steps = int(total_steps * self.config.warmup_ratio)

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = None

    def _init_logger(self):
        if self.config.log_to_wandb:
            # gather params from the config section to pass to W&B
            params = {}
            for key in self.config.__dataclass_fields__.keys():
                params[key] = getattr(self.config, key)

            wandb_project_name = self.config.wandb_project_name
            wandb_entity = self.config.wandb_entity

            # initialize the run (expects WANDB_API_KEY env var or w&b login)
            init_kwargs = {"project": wandb_project_name, "config": params}
            if wandb_entity:
                init_kwargs["entity"] = wandb_entity

            # keep run name deterministic-ish
            run_name = self.config.wandb_run_name or datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            init_kwargs["name"] = run_name

            self.wandb_run = wandb.init(**{k: v for k, v in init_kwargs.items() if v})
            # record a few root-level fields as config (dataset sizes are filled later)
            self.wandb_run.config.update(
                {
                    "epochs": self.config.num_epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "early_stopping": self.config.early_stopping,
                },
                allow_val_change=True,
            )
            self.run_id = wandb.run.id
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run = None

    def _init_model(self):
        self.udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.config.model_to_fine_tune, output_hidden_states=True
        ).to(self.device)

        layers_to_unfreeze = self.config.layers_to_unfreeze

        if layers_to_unfreeze > 0:
            # freeze model weights except n last layers and pooler
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.encoder.layer[-layers_to_unfreeze:].parameters():
                param.requires_grad = True

            for param in self.model.pooler.parameters():
                param.requires_grad = True

    def _init_loss(self):
        loss_type = self.config.loss_type
        if "triplet_loss" in loss_type:
            self.loss = TripletLoss(
                model=self.model,
                margin=0.2 if loss_type == "triplet_loss_cosine" else 1.0,
                p=2,
                loss_type=loss_type,
                pool_targets=self.config.pool_targets,
                use_both_poolings=self.config.use_both_poolings,
            )
        elif loss_type == "nt_xent_loss":
            self.loss = NTXentLoss(model=self.model, temperature=0.05)
        elif loss_type == "mnr_loss":
            self.loss = MNRLoss(
                model=self.model,
                temperature=0.05,
                pool_targets=self.config.pool_targets,
                use_both_poolings=self.config.use_both_poolings,
            )
        else:
            raise NotImplementedError(f"Loss {loss_type} is not implemented yet")


    def _setup_data(self):
        # 1. Load raw dataframes
        df = pd.read_csv(self.config.train_data_path).sample(frac=1, random_state=42)

        # Simple 99/1 split as per your original code
        split_idx = int(len(df) * 0.99)
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]

        # 2. Get loaders from Factory
        # This keeps the Trainer class clean of dataset-specific logic
        self.train_loader, self.eval_loader = DataFactory.get_loaders(
            config=self.config,
            tokenizer=self.tokenizer,
            train_df=train_df,
            eval_df=eval_df,
        )

        if self.config.log_to_wandb:
            # update dataset sizes in wandb config
            self.wandb_run.config.update(
                {
                    "dataset/train": len(train_df),
                    "dataset/eval": len(eval_df),
                },
                allow_val_change=True,
            )

        # 3. Load WSD-specific evaluation data (The 'Ground Truth' for WSD tasks)
        self.wsd_eval_df = pd.read_csv(self.config.wsd_eval_path)
        # Applying your original parsing logic
        self.wsd_eval_df["examples"] = self.wsd_eval_df["examples"].apply(literal_eval)
        self.wsd_eval_df["gloss"] = self.wsd_eval_df["gloss"].apply(
            lambda x: literal_eval(x) if x.startswith("[") else [x]
        )

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

        mean_eval_loss = eval_loss / len(self.eval_loader)

        with autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            wsd_acc = self._calculate_wsd_accuracy(self.wsd_eval_df)

        report_gpu()

        if self.config.log_to_wandb:
            # log epoch-level eval metrics
            self.wandb_run.log(
                {
                    "eval/loss": mean_eval_loss,
                    "eval/wsd_acc": wsd_acc,
                    "epoch": epoch,
                },
                step=self.global_step,
            )

        if wsd_acc > self.max_wsd_acc:
            self.max_wsd_acc = wsd_acc
            self.rounds_count = 0

            if batch_count > 0:
                # save the model and upload to W&B as artifact if enabled
                model_dir = f"{self.config.path_to_save_fine_tuned_model}/model_{self.run_id}_{epoch}"
                self._save_model(
                    self.model,
                    model_dir,
                )

                # if self.config.log_to_wandb:
                #     # create an artifact for the saved model directory
                #     artifact = wandb.Artifact(
                #         name=f"model_{self.run_id}_{epoch}", type="model"
                #     )
                #     artifact.add_dir(model_dir)
                #     self.wandb_run.log_artifact(artifact)

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

        if self.rounds_count >= self.config.early_stopping:
            print(
                f"Early stopping, model not improve WSD for {self.config.early_stopping}"
            )
            return True

        return False

    def train_epoch(self, epoch):
        train_bar = tqdm(self.train_loader, leave=True, desc=f"Train epoch: {epoch}")

        for batch_count, batch in enumerate(train_bar):
            self.model.train()
            self.optimizer.zero_grad()

            with autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                loss = self.loss(batch).to(self.device)

            self.scaler.scale(loss).backward()
            max_grad_norm = self.config.max_grad_norm
            if max_grad_norm and max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.train_avg_meter.update(
                loss.item(), self.config.batch_size
            )  # TODO: .detach().cpu()?

            if self.config.apply_warmup:
                self.scheduler.step()

            if self.config.log_to_wandb:
                acc_val, acc_avg = self.train_avg_meter()
                self.wandb_run.log(
                    {
                        "train/loss": acc_avg,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=self.global_step,
                )

            # increment global step for W&B alignment
            self.global_step += 1

            if batch_count > 0 and batch_count % self.config.num_batch_to_eval == 0:
                if self.evaluate_epoch(epoch, batch_count):
                    return True  # reach early stopping rounds

    def train(self):
        try:
            # initial evaluation of the raw model
            for epoch in range(self.config.num_epochs):
                early_stop = self.evaluate_epoch(epoch=epoch, batch_count=0)
                report_gpu()

                if not early_stop:
                    early_stop = self.train_epoch(epoch)

                if early_stop:
                    path_to_save_model = self.config.path_to_save_fine_tuned_model
                    model_name = f"{path_to_save_model}/model_{self.run_id}_{epoch}_early_stopped"
                    self._save_model(self.model, model_name)

                    if self.config.log_to_wandb:
                        artifact = wandb.Artifact(
                            name=f"model_{self.run_id}_{epoch}_early_stopped",
                            type="model",
                        )
                        artifact.add_dir(model_name)
                        self.wandb_run.log_artifact(artifact)
                    break

                path_to_save_model = self.config.path_to_save_fine_tuned_model
                model_name = f"{path_to_save_model}/model_{self.run_id}_{epoch}"
                self._save_model(self.model, model_name)

                # if self.config.log_to_wandb:
                    # artifact = wandb.Artifact(
                    #     name=f"model_{self.run_id}_{epoch}", type="model"
                    # )
                    # artifact.add_dir(model_name)
                    # self.wandb_run.log_artifact(artifact)
        finally:
            path_to_save_model = self.config.path_to_save_fine_tuned_model
            model_name = f"{path_to_save_model}/model_{self.run_id}_final"
            self._save_model(self.model, model_name)
            
            # final evaluation of the model
            wsd_acc = evaluate_wsd(
                model_path=model_name,
                model_tokenizer_path=self.config.tokenizer_name,
                verbose=True,
            )
            
            # log final WSD accuracy to W&B if enabled
            if self.config.log_to_wandb:
                # artifact = wandb.Artifact(
                #     name=f"model_{self.run_id}_final", type="model"
                # )
                # artifact.add_dir(model_name)
                # self.wandb_run.log_artifact(artifact)

                self.wandb_run.log({"test/wsd_acc": wsd_acc}, step=self.global_step)
                wandb.finish()


if __name__ == "__main__":
    config_path = "services/trainer/fine_tuning_config.ini"

    model_trainer = Trainer(config_path)
    model_trainer.train()
