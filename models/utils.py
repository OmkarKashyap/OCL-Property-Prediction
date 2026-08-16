import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import hydra
import torch
import yaml
from ignite.engine import Engine
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models.base_model import BaseModel

import argparse
import os
import torch.nn.functional as F
import torch.distributed as dist
import sys
from torchvision import transforms
import numpy as np
from models.dinosaur_model import DINOSAURpp, Visual_Encoder
import sys
sys.path.insert(0, '/home/sravanti/.cache/torch/hub/facebookresearch_dino_main')


@dataclass
class ForwardPass:
    model: BaseModel
    device: Union[torch.device, str]
    preprocess_fn: Optional[Callable] = None

    def __call__(self, batch: dict) -> Tuple[dict, dict]:
        for key in batch.keys():
            batch[key] = batch[key].to(self.device, non_blocking=True)
        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self.model(batch["image"])
        return batch, output


def single_forward_pass(
    model: BaseModel, dataloader: DataLoader, device: Union[torch.device, str]
) -> Tuple[dict, dict]:
    eval_step = ForwardPass(model, device)
    evaluator = Engine(lambda e, b: eval_step(b))
    evaluator.run(dataloader, 1, 1)
    batch, output = evaluator.state.output
    return batch, output


class TrainCheckpointHandler:
    def __init__(
        self, checkpoint_path: Union[str, Path], device: Union[torch.device, str]
    ):
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.checkpoint_train_path = checkpoint_path / "train_checkpoint.pt"
        self.model_path = checkpoint_path / "model.pt"
        self.train_yaml_path = checkpoint_path / "train_state.yaml"
        self.device = device

    def save_checkpoint(self, state_dicts: dict):
        """Saves a checkpoint.

        If the state contains the key "model", the model parameters are saved
        separately to model.pt, and they are not saved to the checkpoint file.
        """
        if "model" in state_dicts:
            logging.info(f"Saving model to {self.model_path}")
            torch.save(state_dicts["model"], self.model_path)
            del state_dicts["model"]  # do not include model (avoid duplicating)
        torch.save(state_dicts, self.checkpoint_train_path)

        # Save train state (duplicate info from main checkpoint)
        trainer_state = state_dicts["trainer"]
        with open(self.train_yaml_path, "w") as f:
            train_state = {
                "step": trainer_state["iteration"],
                "max_step": trainer_state["epoch_length"],
            }
            yaml.dump(train_state, f)

    def load_checkpoint(self, objects: dict):
        """Loads checkpoint into the provided dictionary."""

        # Load checkpoint without model
        state = torch.load(self.checkpoint_train_path, self.device)
        for varname in state:
            logging.debug(f"Loading checkpoint: variable name '{varname}'")
            objects[varname].load_state_dict(state[varname])

        # Load model
        if "model" in objects:
            logging.debug(f"Loading checkpoint: model")
            model_state_dict = torch.load(self.model_path, self.device)
            objects["model"].load_state_dict(model_state_dict)


def linear_warmup_exp_decay(
    warmup_steps: Optional[int] = None,
    exp_decay_rate: Optional[float] = None,
    exp_decay_steps: Optional[int] = None,
) -> Callable[[int], float]:
    assert (exp_decay_steps is None) == (exp_decay_rate is None)
    use_exp_decay = exp_decay_rate is not None
    if warmup_steps is not None:
        assert warmup_steps > 0

    def lr_lambda(step):
        multiplier = 1.0
        if warmup_steps is not None and step < warmup_steps:
            multiplier *= step / warmup_steps
        if use_exp_decay:
            multiplier *= exp_decay_rate ** (step / exp_decay_steps)
        return multiplier

    return lr_lambda

def restart_from_checkpoint( args, run_variables, **kwargs):
    checkpoint_path = args.checkpoint_path
    print('CKPT path', checkpoint_path )

    assert checkpoint_path is not None
    # assert os.path.exists(checkpoint_path)

    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint with msg {}".format(key, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint".format(key))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint".format(key))
        else:
            print("=> key '{}' not found in checkpoint".format(key))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def get_parser():
    #### Parser args
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--resize_to',  nargs='+', type=int, default=[336, 336]) 
    parser.add_argument('--encoder', type=str, default="dinov2-vitb-14", 
                        choices=["dinov2-vitb-14", "dino-vitb-16", "dino-vitb-8", "sup-vitb-16","dinov3-vitb-16"])
    parser.add_argument('--num_slots', type=int, default=7)
    parser.add_argument('--num_slots_sub', type=int, default=3)
    parser.add_argument('--slot_att_iter', type=int, default=3)
    parser.add_argument('--slot_dim', type=int, default=768)
    parser.add_argument('--query_opt', action="store_true", default = False)
    parser.add_argument('--ISA', action="store_true", default = False)
    # parser.add_argument('--use_checkpoint', action="store_true")
    parser.add_argument('--checkpoint_path', type=str, default='/data/omkar/object-centric-library/checkpoints/ftdino_eval/checkpoint_epoch_99.pt')
    # parser.add_argument('--validation_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    # parser.add_argument('--model_save_path', type=str, required=True)
    return parser


def load_model(
    config: DictConfig, checkpoint_path: Union[Path, str], model_args: dict = None
) -> BaseModel:
    """Instantiates model from config and loads it from checkpoint."""
    # if model_args is None:
    #     model_args = {}
    # model: BaseModel = hydra.utils.instantiate(config.model, **model_args)
    # model.to(config.device)
    # if isinstance(checkpoint_path, str):
    #     checkpoint_path = Path(checkpoint_path)
    # model_path = checkpoint_path / "model.pt"
    # model.load_state_dict(torch.load(model_path, config.device))

    model_name  = config.model.name
    
    if 'dinov2' in model_name:  #For dinov2 gt masks
        from transformers import AutoImageProcessor, AutoModel, AutoConfig
        vision_tower = AutoModel.from_pretrained('facebook/dinov2-base')
        vision_tower = vision_tower.cuda().eval()
        vision_tower.requires_grad_(False)

        return vision_tower

    if 'dinosaur' in model_name: 
        from transformers import AutoImageProcessor, AutoModel, AutoConfig

        args = get_parser().parse_args([])
        # init_distributed_mode(self.args)
        args.use_checkpoint = True
        args.patch_size = int(args.encoder.split("-")[-1])
        args.token_num = (args.resize_to[0] * args.resize_to[1]) // (args.patch_size ** 2)
        args.gpus = 1

        image_processor = transforms.Compose([transforms.Resize(336),
                                    transforms.CenterCrop(336),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
        
        vision_encoder = Visual_Encoder(args).cuda()

        # vision_encoder.load_state_dict('dinov2_ckpt')
        vision_tower = DINOSAURpp(args, None).cuda()
        vision_tower = torch.nn.parallel.DataParallel(vision_tower)
        to_restore = {"epoch": 99}
        restart_from_checkpoint(args, 
                                run_variables=to_restore, 
                                model=vision_tower)
        vision_tower = vision_tower.module
        vision_tower.vision_encoder = vision_encoder
        vision_tower.eval()
        vision_encoder.eval()
        vision_tower.requires_grad_(False)
        vision_encoder.requires_grad_(False)
        return vision_tower
    
    if 'ft-dinosaur-patch-avg' in model_name:
        import cv2
        from transformers import AutoImageProcessor, AutoModel, AutoConfig

        from ftdinosaur_inference.ftdinosaur_inference import build_dinosaur
        from ftdinosaur_inference.ftdinosaur_inference.utils import resize_patches_to_image, build_preprocessing, soft_masks_to_one_hot

        vision_tower = AutoModel.from_pretrained('facebook/dinov2-base')
        vision_tower = vision_tower.cuda().eval()
        vision_tower.requires_grad_(False)

        dino_model_name = "dinosaur_base_patch14_518_topk3.coco_dv2_ft_s7_300k+10k"
        ftdino_model = build_dinosaur.build(dino_model_name)
        ftdino_model = ftdino_model.to(torch.float32).cuda()
        ftdino_model.eval()
        ftdino_model.requires_grad_(False)

        return (vision_tower, ftdino_model)
    
    if 'ft-dinosaur' in model_name:
        from ftdinosaur_inference.ftdinosaur_inference import build_dinosaur
        from ftdinosaur_inference.ftdinosaur_inference.utils import resize_patches_to_image, build_preprocessing, soft_masks_to_one_hot

        dino_model_name = "dinosaur_base_patch14_518_topk3.coco_dv2_ft_s7_300k+10k"
        vision_tower = build_dinosaur.build(dino_model_name)
        vision_tower = vision_tower.to(torch.float32).cuda()
        vision_tower.eval()
        vision_tower.requires_grad_(False)

        return vision_tower

def infer_model_type(model_name: str) -> str:
    if model_name.startswith("baseline_vae"):
        return "distributed"
    if model_name in [
        "slot-attention",
        "monet",
        "genesis",
        "space",
        "monet-big-decoder",
        "slot-attention-big-decoder",
        "dinov2",
        "ft-dinosaur",
        "ft-dinosaur-patch-avg",
        "dinosaur",
    ]:
        return "object-centric"
    raise ValueError(f"Could not infer model type for model '{model_name}'")
