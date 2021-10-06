import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer, BaseTrainer_det
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from map_boxes import mean_average_precision_for_boxes
###
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0 :
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class Trainer_det(BaseTrainer_det):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, transform,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None,):
        super().__init__(model, criterion, metric_ftns, optimizer, config, data_loader, transform)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int((len(data_loader) / self.config["batch_size"]) // 8)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.loss_hist = Averager()
        self.score_threshold = self.config["score_threshold"]

    def _train_epoch(self, epoch, kfold, data_loader):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.to(self.device)
        self.model.train()
        self.loss_hist.reset()
        self.len_data_set = len(data_loader)
        for batch_idx, (images, targets, *_) in tqdm(enumerate(data_loader)):
            images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
            images = images.to(self.device).float()
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]
            target = {"bbox": boxes, "cls": labels}

            self.optimizer.zero_grad()
            loss, cls_loss, box_loss = self.model(images, target).values()
            loss_value = loss.detach().item()

            self.loss_hist.send(loss_value)

            loss.backward()
            self.optimizer.step()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) # 
            if batch_idx % self.log_step == 0:
                if self.config["save"] : self.logger.debug('Train Epoch, kfold: {} {} {} Loss: {:.6f}'.format(
                    epoch,
                    kfold, 
                    self._progress((batch_idx+1)*self.config["batch_size"]),
                    self.loss_hist.value))

            if batch_idx == self.len_epoch:
                break
        log = {"loss" : self.loss_hist.value}
        return log

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        # return log

    def _valid_epoch(self, valid_data_loader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        config = get_efficientdet_config(self.config["Net"]["args"]["det_name"])
        config.num_classes = 10
        config.image_size = (512,512)
        
        config.soft_nms = False
        config.max_det_per_image = 40
        
        checkpoint = torch.load(self.is_save_pth_filename, map_location='cpu')
        net = EfficientDet(config, pretrained_backbone=False)

        net.class_net = HeadNet(config, num_outputs=config.num_classes)

        net = DetBenchPredict(net)
        net.load_state_dict(checkpoint)
        net.eval()
        net.to(self.device)
        
        with torch.no_grad():
            new_pred = []
            gt = []
            for images, targets, _, filename in tqdm(valid_data_loader):
                # gpu 계산을 위해 image.to(device)       
                images = torch.stack(images) # bs, ch, w, h 
                images = images.to(self.device).float()
                output = net(images)
                outputs = []
                for out in output:
                    outputs.append({'boxes': out.detach().cpu().numpy()[:,:4], 
                                    'scores': out.detach().cpu().numpy()[:,4], 
                                    'labels': out.detach().cpu().numpy()[:,-1]})  
                for i, output in enumerate(outputs):
                    for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                        if score > self.score_threshold:
                            new_pred.append([filename[i], int(label), score, box[0]*2, box[2]*2, box[1]*2, box[3]*2])
                for i, target in enumerate(targets):
                    bbox = target["boxes"][i]
                    gt.append([filename[i], int(target["labels"][i]), bbox[1].item(), bbox[3].item(), bbox[0].item(), bbox[2].item()])
            mean_ap, _ = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)
        return {"mAP" : mean_ap}

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if self.config["save"] :
            if save_best:
                filename = str(self.checkpoint_dir / 'model_best-epoch{}.pth'.format(epoch))
                torch.save(self.model.state_dict(), best_path)
                self.logger.info("Saving current best: model_best.pth ...")
            else:
                filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
                torch.save(self.model.state_dict(), filename)
                self.logger.info("Saving checkpoint: {} ...".format(filename))
            self.is_save_pth_filename = filename
    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        # if hasattr(self.len_data_set, 'n_samples'):
        #     current = batch_idx * self.len_data_set
        #     total = self.len_data_set.n_samples
        # else:
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
        