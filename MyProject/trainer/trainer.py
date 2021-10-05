import copy
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
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

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

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

class Trainer_dett(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, transform,
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
        for batch_idx, data, target, *_ in tqdm(enumerate(self.data_loader)):
            images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
            images = images.to(self.device).float()
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]
            target = {"bbox": boxes, "cls": labels}

            self.optimizer.zero_grad()
            loss, cls_loss, box_loss = self.model(images, target).values()
            loss_value = loss.detach().item()

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss_value)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     log.update(**{'val_'+k : v for k, v in val_log.items()})

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

class Trainer_det(object):
    """
    Trainer class
    """
    
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, transform,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        # super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler

        cfg_trainer = config['trainer']
        self.model = model
        self.optimizer = optimizer
        self.data_set = data_loader
        self.transform = transform
        self.score_threshold = cfg_trainer["score_threshold"]
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        # self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def train(self):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        loss_hist = Averager()
    
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=None) # Kfold
        # stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None) # Kfold
        kfold_index = 0
        def collate_fn(batch):
            return tuple(zip(*batch))
        self.data_set.set_transform(self.transform.transformations['train'])
        self.data_loader = torch.utils.data.DataLoader(
                            self.data_set,
                            batch_size=32, 
                            num_workers=4,
                            shuffle=True,
                            collate_fn=collate_fn)
        target_count = 0
        for epoch in range(self.epochs):
            loss_hist.reset()

            for images, targets, *_ in tqdm(self.data_loader):
            # for images, targets, image_ids in tqdm(self.data_loader):
                images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]
                target = {"bbox": boxes, "cls": labels}

                # calculate loss
                loss, cls_loss, box_loss = self.model(images, target).values()
                loss_value = loss.detach().item()
                
                loss_hist.send(loss_value)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # cls_loss.backward()
                # box_loss.backward()
                # grad clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 35)
                
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
            torch.save(self.model.state_dict(), f'epoch_{epoch+1}.pth')

        # for epoch in range(self.epochs):
        #     kfold_index = 0
        #     self.model.train()
        #     self.model.to(self.device)
        #     for train_index, validation_index in kfold.split(self.data_set):
        #     # for train_index, validation_index in stratified_kfold.split(self.data_set, labels):
        #         kfold_index+=1
        #         print(f'####### epochs :: {epoch}th, KFold :: {kfold_index}th')
                
        #         train_dataset = torch.utils.data.dataset.Subset(self.data_set, train_index)
                
        #         copied_dataset = copy.deepcopy(self.data_set)
        #         valid_dataset = torch.utils.data.dataset.Subset(copied_dataset, validation_index)
                
        #         train_dataset.dataset.set_transform(self.transform.transformations['train'])
        #         valid_dataset.dataset.set_transform(self.transform.transformations['val'])        
        #         self.data_loader = torch.utils.data.DataLoader(
        #                     train_dataset,
        #                     batch_size= 32, 
        #                     num_workers=4,
        #                     shuffle=True,
        #                     collate_fn=collate_fn)

        #         self.valid_data_loader = torch.utils.data.DataLoader(
        #                     valid_dataset,
        #                     batch_size= 1,
        #                     num_workers=1,
        #                     shuffle=False,
        #                     collate_fn=collate_fn) 

        #         target_count = 0
        #         loss_hist.reset()
        #         for images, targets, *_ in tqdm(self.data_loader):
        #             # print(images.shape
        #             images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
        #             images = images.to(self.device).float()
        #             boxes = [target['boxes'].to(self.device).float() for target in targets]
        #             labels = [target['labels'].to(self.device).float() for target in targets]
        #             target = {"bbox": boxes, "cls": labels}

        #             # calculate loss
        #             loss, cls_loss, box_loss = self.model(images, target).values()
        #             loss_value = loss.detach().item()
                    
        #             loss_hist.send(loss_value)
                    
        #             # backward
        #             self.optimizer.zero_grad()
        #             loss.backward()
        #             # grad clip
        #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 35)
                    
        #             self.optimizer.step()
        #         print(f"!!!!!!! Epoch #{epoch+1}, loss: {loss_hist.value}")

        #     torch.save(self.model.state_dict(), f'epoch_{epoch+1}.pth')
            # self._valid_epoch(epoch)

    def _valid_epoch(self, epoch):
        # self.valid_metrics.reset()
        config = get_efficientdet_config('tf_efficientdet_d1')
        config.num_classes = 10
        config.image_size = (512,512)
        
        config.soft_nms = False
        config.max_det_per_image = 40
        
        checkpoint = torch.load(f"epoch_{epoch+1}.pth", map_location='cpu')
        net = EfficientDet(config, pretrained_backbone=False)

        net.class_net = HeadNet(config, num_outputs=config.num_classes)

        net = DetBenchPredict(net)
        net.load_state_dict(checkpoint)
        net.eval()
        net.to(self.device)
        # self.model.eval()
        with torch.no_grad():
            new_pred = []
            gt = []
            for images, targets, _, filename in tqdm(self.valid_data_loader):
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
            print(f"!!!!!!! mAP : {mean_ap}")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)