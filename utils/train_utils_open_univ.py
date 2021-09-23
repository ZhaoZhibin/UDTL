#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
from torch import optim
import models
import datasets
from utils.counter import AccuracyCounter
import torch.nn.functional as F
from utils.lib import *

#Adapted from https://github.com/YU1ut/openset-DA and https://github.com/thuml/Universal-Domain-Adaptation

class train_utils_open_univ(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        if isinstance(args.transfer_task[0], str):
           #print(args.transfer_task)
           args.transfer_task = eval("".join(args.transfer_task))
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets[
            'target_val'], self.num_classes = Dataset(args.data_dir, args.transfer_task,
                                                      args.inconsistent, args.normlizetype).data_split(
            transfer_learning=True)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        # Define the model
        self.max_iter = len(self.dataloaders['source_train']) * args.max_epoch
        self.model = getattr(models, args.model_name)(args.pretrained)
        self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                              nn.ReLU(inplace=True), nn.Dropout())
        if args.inconsistent == 'OSBP':
            if args.bottleneck:
                self.classifier_layer = getattr(models, 'classifier_OSBP')(in_feature=args.bottleneck_num,
                                                                           output_num=self.num_classes + 1,
                                                                           max_iter=self.max_iter,
                                                                           trade_off_adversarial=args.trade_off_adversarial,
                                                                           lam_adversarial=args.lam_adversarial
                                                                           )
            else:
                self.classifier_layer = getattr(models, 'classifier_OSBP')(in_feature=self.model.output_num(),
                                                                           output_num=self.num_classes + 1,
                                                                           max_iter=self.max_iter,
                                                                           trade_off_adversarial=args.trade_off_adversarial,
                                                                           lam_adversarial=args.lam_adversarial
                                                                           )
        else:
            if args.bottleneck:
                self.classifier_layer = nn.Linear(args.bottleneck_num, self.num_classes)
                self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=args.bottleneck_num,
                                                                        hidden_size=args.hidden_size,
                                                                        max_iter=self.max_iter,
                                                                        trade_off_adversarial=args.trade_off_adversarial,
                                                                        lam_adversarial=args.lam_adversarial
                                                                        )
                self.AdversarialNet_auxiliary = getattr(models, 'AdversarialNet_auxiliary')(in_feature=args.bottleneck_num,
                                                                                            hidden_size=args.hidden_size)
            else:
                self.classifier_layer = nn.Linear(self.model.output_num(), self.num_classes)
                self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                        hidden_size=args.hidden_size,
                                                                        max_iter=self.max_iter,
                                                                        trade_off_adversarial=args.trade_off_adversarial,
                                                                        lam_adversarial=args.lam_adversarial
                                                                        )
                self.AdversarialNet_auxiliary = getattr(models, 'AdversarialNet_auxiliary')(
                    in_feature=self.model.output_num(),
                    hidden_size=args.hidden_size)
        if args.bottleneck:
            self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        else:
            self.model_all = nn.Sequential(self.model, self.classifier_layer)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.inconsistent == 'UAN':
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
                self.AdversarialNet_auxiliary = torch.nn.DataParallel(self.AdversarialNet)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # Define the learning parameters
        if args.inconsistent == "OSBP":
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
        else:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet_auxiliary.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet_auxiliary.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")


        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")


        self.start_epoch = 0


        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.inconsistent == 'UAN':
            self.AdversarialNet.to(self.device)
            self.AdversarialNet_auxiliary.to(self.device)
        self.classifier_layer.to(self.device)

        if args.inconsistent == "OSBP":
            self.inconsistent_loss = nn.BCELoss()

        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_hscore = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])
            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                if phase == 'target_val':
                    counters = [AccuracyCounter() for x in range(self.num_classes + 1)]

                # Set model to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.inconsistent=="UAN":
                        self.AdversarialNet.train()
                        self.AdversarialNet_auxiliary.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.inconsistent=="UAN":
                        self.AdversarialNet.eval()
                        self.AdversarialNet_auxiliary.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train':
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, _ = iter_target.next()
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        features = self.model(inputs)
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)
                        if phase != 'source_train':
                            logits = outputs
                            if not (phase == 'target_val' and args.inconsistent == "UAN"):
                                loss = self.criterion(logits, labels)
                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))
                            classifier_loss = self.criterion(logits, labels)

                            if args.inconsistent == 'OSBP':
                                output_t = self.classifier_layer(
                                    features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)), adaption=True)

                                output_t_prob_unk = F.softmax(output_t, dim=1)[:, -1]
                                # print(output_t_prob_unk)
                                inconsistent_loss = self.inconsistent_loss(output_t_prob_unk,
                                                                                                   torch.tensor([
                                                                                                                    args.th] * args.batch_size).to(
                                                                                                       self.device))  # th为阈值
                            else:
                                domain_prob_source = self.AdversarialNet_auxiliary.forward(
                                    features.narrow(0, 0, labels.size(0)).detach())
                                domain_prob_target = self.AdversarialNet_auxiliary.forward(
                                    features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)).detach())

                                source_share_weight = get_source_share_weight(
                                    domain_prob_source, outputs.narrow(0, 0, labels.size(0)), domain_temperature=1.0,
                                    class_temperature=10.0)
                                source_share_weight = normalize_weight(source_share_weight)
                                target_share_weight = get_target_share_weight(
                                    domain_prob_target,
                                    outputs.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)),
                                    domain_temperature=1.0,
                                    class_temperature=1.0)

                                target_share_weight = normalize_weight(target_share_weight)
                                adv_loss = torch.zeros(1, 1).to(self.device)
                                adv_loss_auxiliary = torch.zeros(1, 1).to(self.device)

                                tmp = source_share_weight * nn.BCELoss(reduction='none')(
                                    domain_prob_source,
                                    torch.ones_like(domain_prob_source))
                                adv_loss += torch.mean(tmp, dim=0, keepdim=True)
                                tmp = target_share_weight * nn.BCELoss(reduction='none')(
                                    domain_prob_target,
                                    torch.zeros_like(domain_prob_target))
                                adv_loss += torch.mean(tmp, dim=0, keepdim=True)

                                adv_loss_auxiliary += nn.BCELoss()(domain_prob_source,
                                                                   torch.ones_like(
                                                                       domain_prob_source))
                                adv_loss_auxiliary += nn.BCELoss()(domain_prob_target,
                                                                   torch.zeros_like(
                                                                       domain_prob_target))
                                inconsistent_loss = adv_loss + adv_loss_auxiliary
                            loss = classifier_loss + inconsistent_loss

                        if phase == 'target_val' and args.inconsistent == "OSBP":
                            loss_temp = loss.item() * labels.size(0)
                            epoch_loss += loss_temp
                            epoch_length += labels.size(0)
                            for (each_predict, each_label) in zip(logits, labels.cpu()):
                                    counters[each_label].Ntotal += 1.0

                                    each_pred_id = np.argmax(each_predict.cpu())
                                    if each_pred_id == each_label:
                                        counters[each_label].Ncorrect += 1.0
                        elif phase == 'target_val' and args.inconsistent == "UAN":
                            for (each_predict, each_label,each_target_share_weight) in zip(logits, labels.cpu(), target_share_weight):
                                    if each_label < self.num_classes:
                                        counters[each_label].Ntotal += 1.0
                                        each_pred_id = np.argmax(each_predict.cpu())
                                        #print(each_target_share_weight)
                                        if not (each_target_share_weight[0] < args.th) and each_pred_id == each_label:
                                            counters[each_label].Ncorrect += 1.0
                                    else:
                                        counters[-1].Ntotal += 1.0
                                        if each_target_share_weight[0] > args.th:
                                            counters[-1].Ncorrect += 1.0
                        else:
                            pred = logits.argmax(dim=1)
                            correct = torch.eq(pred, labels).float().sum().item()
                            loss_temp = loss.item() * labels.size(0)
                            epoch_loss += loss_temp
                            epoch_acc += correct
                            epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(labels), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1
            if phase == 'target_val':
                correct = [x.Ncorrect for x in counters]
                amount = [x.Ntotal for x in counters]
                common_acc = np.sum(correct[0:-1]) / np.sum(amount[0:-1])
                outlier_acc = correct[-1] / amount[-1]
                acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
                acc_class = torch.ones(1, 1) * np.mean(acc_tests)
                acc_class = acc_class[0][0]
                acc_all = np.sum(correct[0:]) / np.sum(amount[0:])
                hscore = 2 * common_acc * outlier_acc / (common_acc + outlier_acc)
                if args.inconsistent == "OSBP":
                    epoch_loss = epoch_loss / epoch_length
                    logging.info(
                        'Epoch: {} {}-Loss: {:.4f} {}-common_acc: {:.4f} outlier_acc: {:.4f} acc_class: {:.4f} acc_all: {:.4f} hscore: {:.4f}, Cost {:.1f} sec'.format(
                            epoch, phase, epoch_loss, phase, common_acc, outlier_acc, acc_class, acc_all, hscore, time.time() - epoch_start
                        ))
                else:
                    logging.info(
                        'Epoch: {} {}-common_acc: {:.4f} outlier_acc: {:.4f} acc_class: {:.4f} acc_all: {:.4f} hscore: {:.4f}, Cost {:.1f} sec'.format(
                            epoch, phase, common_acc, outlier_acc, acc_class, acc_all, hscore, time.time() - epoch_start
                        ))
                # save the checkpoint for other learning
                model_state_dic = self.model_all.state_dict()
                # save the best model according to the val accuracy

                if hscore > best_hscore:
                    best_hscore = hscore
                    logging.info(
                "save best model_hscore epoch {}, common_acc: {:.4f} outlier_acc: {:.4f} acc_class: {:.4f} acc_all: {:.4f} best_hscore: {:.4f},".format(
                            epoch, common_acc, outlier_acc, acc_class, acc_all, best_hscore))
                    torch.save(model_state_dic, os.path.join(self.save_dir,
                                                             '{}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}.pth'.format(
                                                                 epoch, common_acc, outlier_acc,acc_class, acc_all,best_hscore)))
                if epoch > args.max_epoch - 2:
                    logging.info(
                "save last model epoch {}, common_acc: {:.4f} outlier_acc: {:.4f} acc_class: {:.4f} acc_all: {:.4f} hscore: {:.4f}".format(
                            epoch, common_acc, outlier_acc, acc_class, acc_all, hscore))
                    torch.save(model_state_dic, os.path.join(self.save_dir,
                                                             '{}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}.pth'.format(
                                                                 epoch, common_acc, outlier_acc,acc_class, acc_all,hscore)))
                # Print the train and val information via each epoch
            else:
                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()














