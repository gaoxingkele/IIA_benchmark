def train(self):
    self.run_name = self.logger.name
    path = os.path.join(self.args.checkpoint_path, self.run_name)
    make_dir_if_not_exists(path, verbose=True)
    self.optimizer = self._select_optimizer()
    self.criterion = self._select_criterion()
    self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
    self._init_lr_scheduler()
    self.model.to(self.device)
    opt_steps = 0
    cur_epoch = 0
    while cur_epoch < self.args.max_epoch:
        self.model.train()
        for batch_x in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad(set_to_none=True)
            timeseries = batch_x.timeseries.float().to(self.device)
            input_mask = batch_x.input_mask.long().to(self.device)
            if not self.args.set_input_mask:
                input_mask = torch.ones_like(input_mask)
            with torch.autocast(device_type='cuda', dtype=dtype_map(self.args.torch_dtype), enabled=self.args.use_amp):
                outputs = self.model(x_enc=timeseries, input_mask=input_mask, mask=None)
            recon_loss = self.criterion(outputs.reconstruction, timeseries)
            observed_mask = input_mask * (1 - outputs.pretrain_mask)
            n_channels = outputs.reconstruction.shape[1]
            observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
            observed_mask = self.target_validity(observed_mask, batch_x)
            masked_loss = observed_mask * recon_loss
            loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-07)
            self.logger.log({'step_train_loss': loss.item(), 'learning_rate': self.optimizer.param_groups[0]['lr']})
            if self.args.debug and opt_steps >= 1:
                self.debug_model_outputs(loss, outputs, batch_x)
            self.scaler.scale(loss).backward()
            self.model.finish_backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            self.audit_gradients_and_loss(loss, batch_x)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            opt_steps = opt_steps + 1
            if opt_steps % self.args.log_interval == 0:
                self.evaluate_and_log()
            if opt_steps % self.args.checkpoint_interval == 0:
                self.logger.alert(title='Saving model', text=f'Saving model after {opt_steps} steps', level=AlertLevel.INFO)
                self.save_model(self.model, path, opt_steps, self.optimizer, self.scaler)
                self.evaluate_model_external(path, opt_steps, self.device)
            self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
        cur_epoch = cur_epoch + 1
    return self.model
