class DistribuedMongomRunner(Runner):
    """Custom Runner for demonstrating a NeuroImaging Pipeline"""

    def __init__(self, n_classes: int, parallel: bool, grad_checkpoint: bool):
        """Init."""
        super().__init__()
        self.n_classes = n_classes
        self.parallel = parallel
        self.grad_checkpoint = grad_checkpoint

    def get_engine(self):
        """Gets engine for multi or single gpu case"""
        if self.parallel:
            engine = DataParallelAMPEngine()

        else:
            engine = AMPEngine()

        return engine

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "macro_dice"]
        }

    def handle_batch(self, batch):
        """
        Custom train/ val step that includes batch unpacking, training, and
        DICE metrics
        """
        # model train/valid step
        #try:
        #    x, y = batch["images"], batch["nii_labels"].long()
        #except Exception as e:
        #    print("batch ", len(batch), batch[0].shape, batch[1].shape, batch[-1].shape)
        #    raise(e)
        x, y = batch
        print(x.shape, y.shape)


        with self.engine.autocast():
            if self.grad_checkpoint:
                segments = 4
                y_hat = checkpoint_sequential(self.model.module.model, segments, x)
            else:
                y_hat = self.model(x)

            ce_loss = F.cross_entropy(y_hat, y)

        one_hot_targets = (
            torch.nn.functional.one_hot(y, self.n_classes)
            .permute(0, 4, 1, 2, 3)
        )

        loss = ce_loss

        if self.is_train_loader:
            self.engine.backward_loss(loss, self.model, self.optimizer)
            self.engine.optimizer_step(loss, self.model, self.optimizer)
            #scheduler.step()
            self.optimizer.zero_grad()

        macro_dice = dice(F.softmax(y_hat), one_hot_targets, mode="macro")

        self.batch_metrics.update({"loss": loss, "macro_dice": macro_dice})

        for key in ["loss", "macro_dice"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )

    def on_loader_end(self, runner):
        """
        Calls runner methods when a dataloader finishes running and updates
        metrics
        """
        for key in ["loss", "macro_dice"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def predict_batch(self, batch):
        """
        Predicts a batch for an inference dataloader and returns the
        predictions as well as the corresponding slice indices
        """
        # model inference step
        input = batch["images"]
        input = self.engine.sync_device(input)

        if self.parallel:
            for layer in self.model.module.model:
                input = layer(input)

        else:
            for layer in self.model.model:
                input = layer(input)

        y_hat = input

        return (
            y_hat,
            batch["coords"],
        )