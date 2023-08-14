# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.multiprocessing as mp
import omegaconf
from omegaconf.omegaconf import OmegaConf, open_dict
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager



@hydra.main(config_path="conf", config_name="pretraining_config")
def main(cfg) -> None:
    # WIP: In the GPT script it is called at the top level of the script
    if cfg.name == 'megatron_gpt' or (cfg.name == 'megatron_bert' and cfg.model.data.dataloader_type != "LDDL"):
        mp.set_start_method("spawn", force=True)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            # WIP: this should be instantiated by hydra from the config
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.hysteresis,
            )

        # WIP: in bart's script there is no condition `not with_distributed_adam`
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    # WIP: GPT and BERT lack ModelSummary callback. According to docs https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelSummary.html
    # this only prints a message
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[ModelSummary(max_depth=3)])

    exp_manager(trainer, cfg.exp_manager)

    # WIP: in bert script there is no the first branch of this statement
    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path

    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision


    # WIP: following 2 lines are WAR for different structure of class structure and configs
    _target_ = cfg.model.pop('_target_')
    _cfg = omegaconf.DictConfig({'_target_': _target_, 'cfg': cfg.model})
    model = hydra.utils.instantiate(_cfg, trainer=trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
