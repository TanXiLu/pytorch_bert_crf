# coding=utf-8
import os
import torch
import pandas as pd
from utils import set_logger, build_optimizer_and_scheduler
from model import metric


class Trainer:
    def __init__(self, model, device, config, train_loader, dev_loader, test_loader=None):
        self.model = model
        # self.metric = metric
        self.device = device

        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.id2tag = self.config["id2tag"]
        self.labels = self.config["labels"]

        #
        self.global_step = 1
        self.epochs = self.config["epochs"]
        self.log_interval = self.config["log_interval"]
        self.valid_step = self.config["valid_step"]
        self.total_step = len(self.train_loader) * self.epochs
        #
        self.save_model_name = self.config["save_model_name"]
        self.save_model_dir = self.config["save_model_dir"]
        #
        self.best_valid_micro_f1 = 0.0
        #
        if not os.path.exists(config['log_dir']):
            os.makedirs(config['log_dir'])
        self.logger = set_logger(os.path.join(config['log_dir'], config['log_save_name']))
        #
        self.optimizer, self.lr_scheduler = build_optimizer_and_scheduler(config, model, self.total_step)

        resume = self.config["resume"]
        if resume is not None:
            self._resume_checkpoint(resume_path=resume)

    def _train_loop(self, epoch):
        self.model.train()
        running_loss = 0.
        for batch, batch_data in enumerate(self.train_loader):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(self.device)

            loss, logits = self.model(batch_data)
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()

            running_loss += loss.item()
            if self.global_step % self.log_interval == 0:
                cur_loss = running_loss / self.log_interval
                self.logger.info(
                    "[Train] epoch：{} step:{}/{} loss：{:.6f}".format(epoch, self.global_step, self.total_step, cur_loss)
                )
                running_loss = 0.0

            # 验证
            if self.global_step % self.valid_step == 0:
                dev_loss, accuracy, micro_f1, macro_f1 = self._valid_loop(epoch)
                if macro_f1 > self.best_valid_micro_f1:
                    checkpoint = {
                        "epoch": epoch,
                        "loss": dev_loss,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()
                    }
                    self.best_valid_micro_f1 = macro_f1
                    self._save_checkpoint(checkpoint)

            self.global_step += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _valid_loop(self, epoch):
        self.model.eval()
        total_loss = 0.0
        counts = 0
        results = []
        callback_texts = []
        callback_entities = []
        with torch.no_grad():
            for batch, dev_batch_data in enumerate(self.dev_loader):
                for k, v in dev_batch_data.items():
                    if isinstance(v, torch.Tensor):
                        dev_batch_data[k] = v.to(self.device)

                dev_loss, dev_logits = self.model(dev_batch_data)
                counts += 1
                total_loss += dev_loss.item()
                if self.config["use_crf"] is True:
                    batch_output = dev_logits
                else:
                    batch_output = torch.argmax(dev_logits, dim=2).cpu().numpy().tolist()

                results.extend(batch_output)
                callback_texts.extend(dev_batch_data['texts'])
                callback_entities.extend(dev_batch_data['entity_info'])
        # 解码同时计算 p, r, f1
        p, r, f1 = metric(results, callback_texts, callback_entities, self.id2tag, self.labels)
        self.logger.info(
            "[Valid] epoch：{} loss：{:.6f} precision：{:.4f} recall：{:.4f} f1-score：{:.4f} bestF1-score：{:.4f}".format(
                epoch, total_loss / counts, p, r, f1, self.best_valid_micro_f1))
        return total_loss/counts, p, r, f1

    def _save_checkpoint(self, state):
        """
        Saving checkpoints
        """
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        checkpoint_path = os.path.join(self.save_model_dir, self.save_model_name)
        torch.save(state, checkpoint_path)
        self.logger.info(f"Saving current best model: {self.save_model_name} ...")

    def _resume_checkpoint(self, resume_path):
        """
        resume from saved checkpoints
        """
        checkpoint = torch.load(resume_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(checkpoint['epoch']))

    def train(self):
        for epoch in range(self.epochs):
            self._train_loop(epoch)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        counts = 0
        results = []
        callback_texts = []
        callback_entities = []
        with torch.no_grad():
            for batch, dev_batch_data in enumerate(self.dev_loader):
                for k, v in dev_batch_data.items():
                    if isinstance(v, torch.Tensor):
                        dev_batch_data[k] = v.to(self.device)

                dev_loss, dev_logits = self.model(dev_batch_data)
                counts += 1
                total_loss += dev_loss.item()
                if self.config["use_crf"] is True:
                    batch_output = dev_logits
                else:
                    batch_output = torch.argmax(dev_logits, dim=2).cpu().numpy().tolist()

                results.extend(batch_output)
                callback_texts.extend(dev_batch_data['texts'])
                callback_entities.extend(dev_batch_data['entity_info'])

        # 解码同时计算 p, r, f1
        report, p, r, f1, decode_results = metric(results, callback_texts, callback_entities,
                                                  self.id2tag, self.labels, report_results=True)
        self.logger.info(
            "[Test] loss：{:.6f} precision：{:.4f} recall：{:.4f} f1-score：{:.4f}".format(
                total_loss / counts, p, r, f1))
        # 将预测数据写入文件
        result = pd.DataFrame({"文本": callback_texts, "实体": callback_entities, "预测实体": decode_results})
        result['实体'] = result['实体'].apply(lambda x: {key: value for key, value in x.items() if len(value) != 0})
        result['预测实体'] = result['预测实体'].apply(lambda x: {key: value for key, value in x.items() if len(value) != 0})
        if not os.path.exists(self.config['save_result_dir']):
            os.makedirs(self.config['save_result_dir'])
        result.to_excel(os.path.join(self.config['save_result_dir'], 'result.xlsx'))
        return report
