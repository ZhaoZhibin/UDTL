#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import math
import torch
from torch import nn
from torch import optim

import models
import datasets
from loss.DAN import DAN
from loss.JAN import JAN
from loss.CORAL import CORAL
from utils.entropy_CDA import Entropy
from utils.entropy_CDA import calc_coeff
from utils.entropy_CDA import grl_hook
class train_utils(object):
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
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)


        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        # Define the model
        self.model = getattr(models, args.model_name)(args.pretrained)
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)

        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders['source_train'])*(args.max_epoch-args.middle_epoch)
            if args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                if args.bottleneck:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=args.bottleneck_num*Dataset.num_classes,
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
                else:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num()*Dataset.num_classes,
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
            else:
                if args.bottleneck_num:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=args.bottleneck_num,
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )
                else:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                            hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial
                                                                            )


        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.domain_adversarial:
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # Define the learning parameters
        if args.domain_adversarial:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
        else:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]


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
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)


        # Define the distance loss
        if args.distance_metric:
            if args.distance_loss == 'MK-MMD':
                self.distance_loss = DAN
            elif args.distance_loss == "JMMD":
                ## add additional network for some methods
                self.softmax_layer = nn.Softmax(dim=1)
                self.softmax_layer = self.softmax_layer.to(self.device)
                self.distance_loss = JAN
            elif args.distance_loss == "CORAL":
                self.distance_loss = CORAL
            else:
                raise Exception("loss not implement")
        else:
            self.distance_loss = None

        # Define the adversarial loss
        if args.domain_adversarial:
            if args.adversarial_loss == 'DA':
                self.adversarial_loss = nn.BCELoss()
            elif args.adversarial_loss == "CDA" or args.adversarial_loss == "CDA+E":
                ## add additional network for some methods
                self.softmax_layer_ad = nn.Softmax(dim=1)
                self.softmax_layer_ad = self.softmax_layer_ad.to(self.device)
                self.adversarial_loss = nn.BCELoss()
            else:
                raise Exception("loss not implement")
        else:
            self.adversarial_loss = None

        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        iter_num = 0
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
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

                # Set model to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.domain_adversarial:
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.domain_adversarial:
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train' or epoch < args.middle_epoch:
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
                        if phase != 'source_train' or epoch < args.middle_epoch:
                            logits = outputs
                            loss = self.criterion(logits, labels)
                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))
                            classifier_loss = self.criterion(logits, labels)
                            # Calculate the distance metric
                            if self.distance_loss is not None:
                                if args.distance_loss == 'MK-MMD':
                                    distance_loss = self.distance_loss(features.narrow(0, 0, labels.size(0)),
                                                                       features.narrow(0, labels.size(0), inputs.size(0)-labels.size(0)))
                                elif args.distance_loss == 'JMMD':
                                    softmax_out = self.softmax_layer(outputs)
                                    distance_loss = self.distance_loss([features.narrow(0, 0, labels.size(0)),
                                                                        softmax_out.narrow(0, 0, labels.size(0))],
                                                                       [features.narrow(0, labels.size(0),
                                                                                        inputs.size(0)-labels.size(0)),
                                                                        softmax_out.narrow(0, labels.size(0),
                                                                                           inputs.size(0)-labels.size(0))],
                                                                       )
                                elif args.distance_loss == 'CORAL':
                                    distance_loss = self.distance_loss(outputs.narrow(0, 0, labels.size(0)),
                                                                       outputs.narrow(0, labels.size(0), inputs.size(0)-labels.size(0)))
                                else:
                                    raise Exception("loss not implement")

                            else:
                                distance_loss = 0


                            # Calculate the domain adversarial
                            if self.adversarial_loss is not None:
                                if args.adversarial_loss == 'DA':
                                    domain_label_source = torch.ones(labels.size(0)).float()
                                    domain_label_target = torch.zeros(inputs.size(0)-labels.size(0)).float()
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    adversarial_out = self.AdversarialNet(features)
                                    adversarial_loss = self.adversarial_loss(adversarial_out, adversarial_label)
                                elif args.adversarial_loss == 'CDA':
                                    softmax_out = self.softmax_layer_ad(outputs).detach()
                                    op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
                                    adversarial_out = self.AdversarialNet(op_out.view(-1, softmax_out.size(1) * features.size(1)))

                                    domain_label_source = torch.ones(labels.size(0)).float()
                                    domain_label_target = torch.zeros(inputs.size(0)-labels.size(0)).float()
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(self.device)
                                    adversarial_loss = self.adversarial_loss(adversarial_out, adversarial_label)
                                elif args.adversarial_loss == "CDA+E":
                                    softmax_out = self.softmax_layer_ad(outputs)
                                    coeff = calc_coeff(iter_num, self.max_iter)
                                    entropy = Entropy(softmax_out)
                                    entropy.register_hook(grl_hook(coeff))
                                    entropy = 1.0 + torch.exp(-entropy)
                                    entropy_source = entropy.narrow(0, 0, labels.size(0))
                                    entropy_target = entropy.narrow(0, labels.size(0), inputs.size(0) - labels.size(0))

                                    softmax_out = softmax_out.detach()
                                    op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
                                    adversarial_out = self.AdversarialNet(
                                        op_out.view(-1, softmax_out.size(1) * features.size(1)))
                                    domain_label_source = torch.ones(labels.size(0)).float().to(
                                        self.device)
                                    domain_label_target = torch.zeros(inputs.size(0) - labels.size(0)).float().to(
                                        self.device)
                                    adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(
                                        self.device)
                                    weight = torch.cat((entropy_source / torch.sum(entropy_source).detach().item(),
                                                        entropy_target / torch.sum(entropy_target).detach().item()), dim=0)

                                    adversarial_loss = torch.sum(weight.view(-1, 1) * self.adversarial_loss(adversarial_out, adversarial_label)) / torch.sum(weight).detach().item()
                                    iter_num += 1

                                else:
                                    raise Exception("loss not implement")
                            else:
                                adversarial_loss = 0

                            # Calculate the trade off parameter lam
                            if args.trade_off_distance == 'Cons':
                                lam_distance = args.lam_distance
                            elif args.trade_off_distance == 'Step':
                                lam_distance = 2 / (1 + math.exp(-10 * ((epoch-args.middle_epoch) /
                                                                        (args.max_epoch-args.middle_epoch)))) - 1
                            else:
                                raise Exception("trade_off_distance not implement")

                            # if args.trade_off_adversarial == 'Cons':
                            #     lam_adversarial = args.lam_adversarial
                            # elif args.trade_off_adversarial == 'Step':
                            #     lam_adversarial = 2 / (1 + math.exp(-10 * ((epoch-args.middle_epoch) /
                            #                                             (args.max_epoch-args.middle_epoch)))) - 1
                            # else:
                            #     raise Exception("loss not implement")

                            # loss = classifier_loss + lam_distance * distance_loss + lam_adversarial * adversarial_loss
                            loss = classifier_loss + lam_distance * distance_loss + adversarial_loss


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

                # Print the train and val information via each epoch

                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))
                # save the model
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()
                    # save the best model according to the val accuracy
                    if (epoch_acc > best_acc or epoch > args.max_epoch-2) and (epoch > args.middle_epoch-1):
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()














