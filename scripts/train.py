import os
import wandb
os.environ['WANDB_API_KEY']='KEY'
os.environ['WANDB_MODE']='offline'


from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback




log = logging.getLogger(__name__)

CONFIG_PATH = '/home/ln/code/bev_vit/config'
#CONFIG_PATH = Path.cwd() / 'config'                 #这个执行的路径和需求路径在执行时不同
CONFIG_NAME = 'config.yaml'

# print(CONFIG_PATH)

def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False) #创建日志文件

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)

    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)
    # print("########################训练000")
    #print("ckpt_path111111111111111111111111111111111111",ckpt_path)
    #print("ckpt_path111111111111111111111111111111111111",model_module)
    #print("ckpt_path111111111111111111111111111111111111",data_module)
    #print("ckpt***********8",ckpt_path)
    os.environ['WANDB_API_KEY'] = 'KEY'
    os.environ['WANDB_MODE'] = 'offline'
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)   #使用 trainer.fit() 函数，模型的实际训练开始。
    #print("ckpt***********8",ckpt_path)
    #print("ckpt_path222222222222222222222222222222222222", ckpt_path)                                                                        # trainer.fit(model) model.save_to(config.save_to)
    #rint("########################训练111") #从训练中提取最佳检查点的路径；您可以将其更新到任何检查点。
    # checkpoint_path = trainer.checkpoint_callback.best_model_path
    # 创建评估模型并加载检查点。
    # eval_model = nemo_nlp.models.TextClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

if __name__ == '__main__':
    main()