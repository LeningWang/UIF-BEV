import torch
import pytorch_lightning as pl
import numpy as np

from PIL import Image

class ModelModule(pl.LightningModule):  #  从这里开始看！！！！,这里即为模型的开端
    def __init__(self, backbone, loss_func, metrics, optimizer_args, scheduler_args=None, cfg=None):
        super().__init__()  #骨架；损失函数；  评价指标函数；    优化器参数； 调度程序参数

        self.save_hyperparameters(
            cfg,
            ignore=['backbone', 'loss_func', 'metrics', 'optimizer_args', 'scheduler_args'])

        self.backbone = backbone  # _target_: cross_view_transformer.model.backbones.efficientnet.EfficientNetExtractor
        self.loss_func = loss_func
        self.metrics = metrics  # 好像这个是评价！！！

        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def forward(self, batch):   #将batch输入到文件
        return self.backbone(batch)

    def shared_step(self, batch, prefix='', on_step=False, return_output=True):  # 应该是权重共享
        array1=batch['image']
        #data = np.matrix(array1.cpu())
        data = array1
        #np.savetxt("/home/ln/result1.txt",data);
        # 显示图像:
        #new_map.show()

        pred = self(batch)
        loss, loss_details = self.loss_func(pred, batch)  # pred和batch求损失函数  loss = loss_func(output, target)
        # print("lossssssssssssssssssss",loss)
        self.metrics.update(pred, batch)  # 使用这个，从而评估模型的正确与否

        if self.trainer is not None:
            self.log(f'{prefix}/loss', loss.detach(), on_step=on_step, on_epoch=True)
            self.log_dict({f'{prefix}/loss/{k}': v.detach() for k, v in loss_details.items()},
                          on_step=on_step, on_epoch=True)

        # Used for visualizations
        if return_output:
            return {'loss': loss, 'batch': batch, 'pred': pred}
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train', True,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val', False,
                                batch_idx % self.hparams.experiment.log_image_interval == 0)

    def on_validation_start(self) -> None:
        self._log_epoch_metrics('train')
        self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def validation_epoch_end(self, outputs):  # 训练集(train)验证集(validation)测试集(test)
        self._log_epoch_metrics('val')

    def _log_epoch_metrics(self, prefix: str):
        """
        lightning is a little odd - it goes

        on_train_start
        ... does all the training steps ...
        on_validation_start
        ... does all the validation steps ...
        on_validation_epoch_end
        on_train_epoch_end
        """
        metrics = self.metrics.compute()

        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, val in value.items():
                    self.log(f'{prefix}/metrics/{key}{subkey}', val)
            else:
                self.log(f'{prefix}/metrics/{key}', value)

        self.metrics.reset()

    def _enable_dataloader_shuffle(self, dataloaders):

        for v in dataloaders:
            v.sampler.shuffle = True
            # v.sampler.shuffle = False
            v.sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self, disable_scheduler=False):
        parameters = [x for x in self.backbone.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)

        if disable_scheduler or self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_args)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
