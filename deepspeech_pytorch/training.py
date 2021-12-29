import json

import hydra
from deepspeech_pytorch.checkpoint import GCSCheckpointHandler, FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, GCSCheckpointConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch


def train(cfg: DeepSpeechConfig):
    seed_everything(cfg.seed)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    if cfg.trainer.checkpoint_callback:
        if OmegaConf.get_type(cfg.checkpoint) is GCSCheckpointConfig:
            checkpoint_callback = GCSCheckpointHandler(
                cfg=cfg.checkpoint
            )
        else:
            checkpoint_callback = FileCheckpointHandler(
                cfg=cfg.checkpoint
            )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            print("resume check point === ", resume_from_checkpoint)
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint
    #print("**********HERE**************")
    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        is_distributed=cfg.trainer.gpus > 1
    )
#    print("**********HERE**************", cfg.trainer
    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )
#    print("**********HERE**************", cfg.trainer)
    trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        replace_sampler_ddp=False,
        callbacks=[checkpoint_callback] if cfg.trainer.checkpoint_callback else None,
    )
    trainer.fit(model, data_loader)
