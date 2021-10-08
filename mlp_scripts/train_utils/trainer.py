"""base trainer: https://github.com/kdha0727/easyrun-pytorch/blob/main/easyrun.py (Dongha Kim)"""

# Frameworks and Internal Modules

import os
import pathlib
import glob

import contextlib
import time
import functools

import torch
import torch.nn.functional
import torch.utils.data


@functools.lru_cache(maxsize=None)  # Remember prior inputs
def get_loader_information(loader):
    batch_size = getattr(loader, 'batch_size', 1)
    loader_length = len(loader)
    dataset_length = len(getattr(loader, 'dataset', loader))
    return batch_size, loader_length, dataset_length


class CheckpointMixin(object):

    _closed = True

    #
    # De-constructor: executed in buffer-cleaning in python exit
    #

    def __del__(self):
        self._close()

    #
    # Context manager magic methods
    #

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_info, exc_class, exc_traceback):
        try:
            self._close()
        except Exception as exc:
            if (exc_info or exc_class or exc_traceback) is not None:
                pass  # executed in exception handling - just let python raise that exception
            else:
                raise exc

    def _require_context(self):
        if self._closed:
            raise ValueError('Already closed: %r' % self)

    @contextlib.contextmanager
    def _with_context(self):
        prev = True
        try:
            prev = self._open()
            yield
        finally:
            self._close(prev)

    def _open(self, *args, **kwargs):
        raise NotImplementedError

    def _close(self, *args, **kwargs):
        raise NotImplementedError


class MovableMixin(object):

    __to_parse = (None, None, False, None)

    #
    # Device-moving Methods
    #

    def to(self, *args, **kwargs):  # overwrite this in subclass, for further features
        self._to_set(*args, **kwargs)
        return self

    # Internal Device-moving Methods

    def _to_set(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)  # noqa
        device = device or self.__to_parse[0]
        dtype = dtype or self.__to_parse[1]
        non_blocking = non_blocking or self.__to_parse[2]
        convert_to_format = convert_to_format or self.__to_parse[3]
        self.__to_parse = (device, dtype, non_blocking, convert_to_format)

    def _to_apply_module(self, v):
        device, dtype, non_blocking, convert_to_format = self.__to_parse
        return v.to(device, dtype, non_blocking, memory_format=convert_to_format)

    def _to_apply_tensor(self, v):
        device, dtype, _, convert_to_format = self.__to_parse
        return v.to(device, dtype, memory_format=convert_to_format)

    def _to_apply_multi_tensor(self, *v):
        return tuple(map(self._to_apply_tensor, v))


class TimerLoggerMixin(object):

    train_iter: ...
    total_epoch: int
    verbose: bool
    use_timer: bool
    progress: bool
    save_and_load: bool
    _best_loss: float

    _time_start = None
    _time_stop = None

    # Internal Timing Functions

    def _timer_start(self):
        if self.use_timer:
            self._time_start = time.time()

    def _timer_stop(self):
        if self.use_timer:
            self._time_stop = time.time()

    # Internal Logging Methods

    def _log_start(self):
        if self.verbose:
            log = f"\n<Start Learning> "
            if self.total_epoch is not None:
                log += f"\t\t\t\tTotal {self.total_epoch} epochs"
            self.log_function(log)

    def _log_step(self, epoch: int):
        if self.verbose:
            if self.progress:
                self.log_function(f'\nEpoch {epoch}')
            else:
                self.log_function(f'Epoch {epoch}', end=' ')

    def _log_train_doing(self, loss, iteration, whole=None):
        if self.verbose and self.progress:
            if isinstance(whole, int):
                batch_size = 1
                loader_length = dataset_length = whole
            else:
                batch_size, loader_length, dataset_length = get_loader_information(whole or self.train_iter)
            self.log_function(
                f'\r[Train]\t '
                f'Progress: {iteration * batch_size}/{dataset_length} '
                f'({100. * iteration / loader_length:05.2f}%), \tLoss: {loss:.6f}',
                end=' '
            )

    def _log_train_done(self, loss, whole=None):
        if self.verbose:
            if isinstance(whole, int):
                dataset_length = whole
            else:
                _, _, dataset_length = get_loader_information(whole or self.train_iter)
            if self.progress:
                log = f'\r[Train]\t Progress: {dataset_length}/{dataset_length} (100.00%), \t'
            else:
                log = f'[Train]\t '
            log += f'Average Loss: {loss:.6f}'
            if self.progress:
                self.log_function(log)
            else:
                self.log_function(log, end='\t ')

    def _log_eval(self, loss, test=False):
        if self.verbose:
            log = '\n[Test]\t ' if test else '[Eval]\t '
            log += f'Average loss: {loss:.6f}, '
            if self.use_timer:
                log += "\tTime Elapsed: "
                duration = time.time() - self._time_start
                if duration > 60:
                    log += f"{int(duration // 60):02}m {duration % 60:05.2f}s"
                else:
                    log += f"{duration:05.2f}s"
            self.log_function(log)

    def _log_stop(self):
        if self.verbose:
            log = "\n<Stop Learning> "
            if self.save_and_load:
                log += f"\tLeast loss: {self._best_loss:.4f}"
            if self.use_timer:
                log += "\tDuration: "
                duration = self._time_stop - self._time_start
                if duration > 60:
                    log += f"{int(duration // 60):02}m {duration % 60:05.2f}s"
                else:
                    log += f"{duration:05.2f}s"
            self.log_function(log)

    # Log function: overwrite this to use custom logging hook
    log_function = staticmethod(print)


class TrainerMixin(CheckpointMixin, MovableMixin, TimerLoggerMixin):

    @staticmethod
    def _check_criterion(criterion):
        if not callable(criterion):  # allow string: convert string to callable function or module
            assert isinstance(criterion, str), \
                "Invalid criterion type: %s" % criterion.__class__.__name__
            assert (hasattr(torch.nn, criterion) or hasattr(torch.nn.functional, criterion)), \
                "Invalid criterion string: %s" % criterion
            criterion = getattr(torch.nn.functional, criterion, getattr(torch.nn, criterion)())
        return criterion

    @staticmethod
    def _check_data(dataset_or_loader):
        if dataset_or_loader is None:
            return
        assert isinstance(dataset_or_loader, (torch.utils.data.Dataset, torch.utils.data.DataLoader)), \
            "Invalid test_iter type: %s" % dataset_or_loader.__class__.__name__
        return dataset_or_loader

    _open: ...
    _close: ...


#
# One-time Trainer Class
#

class RegressionTrainer(TrainerMixin):
    """

    One-time Pytorch Trainer Utility

    Parameters:

        model : torch.nn.Module
            model object, or list of model objects to use.

        criterion : callable, torch.nn.Module, or str
            loss function or model object.
            You can also provide string name.

        optimizer : torch.optim.Optimizer
            optimizer.

        lr_scheduler : (optional) torch.optim.lr_scheduler._LRScheduler
            learning rate scheduler.

        epoch : int
            total epochs.

        train_iter : Dataset or Dataloader
            train data loader, or train dataset.

        val_iter : Dataset or Dataloader
            validation data loader, or validation dataset.

        test_iter : Dataset or Dataloader
            test data loader, or test dataset.

        snapshot_dir: str
            provide if you want to use parameter saving and loading.
            in this path name, model's weight parameter at best(least) loss will be temporarily saved.

        verbose: bool, default=True
            verbosity. with turning it on, you can view learning logs.
            default value is True.

        timer: bool, default=True
            provide with verbosity, if you want to use time-checking.

        log_interval: int, default=20
            provide with verbosity, if you want to change log interval.

    Methods:

        to(device)
            apply to(device) in model, criterion, and all tensors.

        train()
            run training one time with train dataset.
            returns train_loss.

        evaluate()
            run validating one time with validation dataset.
            returns val_loss).

        step()
            run training, followed by validating one time.
            returns (train_loss, val_loss).

        run() or fit() or () [call]
            repeat training and validating for all epochs.
            returns train result list, which contains each epoch`s
            (train_loss, val_loss).

        test()
            run testing one time with test dataset.
            returns (test_loss, test_accuracy).

        state_dict()
            returns state dictionary of trainer class.

        load_state_dict()
            loads state dictionary of trainer class.

    Attributes:

        _current_epoch: int
            current epoch (iteration).

        _best_loss: int
            best loss value.

    Properties:

        best_loss: int
            best loss value.

    """

    #
    # Constructor
    #

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            lr_scheduler=None,
            epoch=None,
            train_iter=None,
            val_iter=None,
            test_iter=None,
            snapshot_dir=None,
            verbose=True,
            timer=True,
            progress=True,
            log_interval=20,
    ):

        assert isinstance(log_interval, int) and log_interval > 0, \
            "Log Interval is expected to be positive int, got %s" % log_interval

        self.model = model
        self.criterion = self._check_criterion(criterion)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epoch = epoch

        self.train_iter = self._check_data(train_iter)
        self.val_iter = self._check_data(val_iter)
        self.test_iter = self._check_data(test_iter)

        self.snapshot_dir = pathlib.Path(snapshot_dir).resolve() if snapshot_dir is not None else None
        self.verbose = verbose
        self.use_timer = timer
        self.progress = progress
        self.log_interval = log_interval
        self.save_and_load = snapshot_dir is not None

        # State Variables
        self._current_epoch = 0
        self._best_loss = float('inf')
        self._processing_fn = None
        self._current_run_result = None

    #
    # Running Methods
    #

    def _train(self, data=None):

        self._require_context()

        if data is not None:
            data = self._check_data(data)
            log_kwargs = dict(whole=data)
        else:
            data = self.train_iter
            log_kwargs = dict()
        assert data is not None, "You must provide dataset for evaluating method."
        alpha = 0.3
        loss_f = None

        self.model.train()

        for iteration, (x, y) in enumerate(data, 1):
            x, y = self._to_apply_multi_tensor(x, y)
            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if loss_f is None:
                loss_f = loss.item()
            else:
                loss_f = (1 - alpha) * loss_f + alpha * loss.item()
            if iteration % self.log_interval == 0:
                self._log_train_doing(loss_f, iteration, **log_kwargs)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self._log_train_done(loss_f, **log_kwargs)
        return loss_f

    @torch.no_grad()
    def _evaluate(self, data=None, *, test=False):

        data = self._check_data(data) or (self.test_iter if test else self.val_iter)
        assert data is not None, "You must provide dataset for evaluating method."
        total_len, batch_size = 0, 1
        total_loss = 0.
        if isinstance(data, torch.utils.data.DataLoader):
            try:
                total_len = len(data.dataset)  # type: ignore
            except (TypeError, ValueError):
                pass
        count = total_len == 0

        self.model.eval()

        for x, y in data:
            x, y = self._to_apply_multi_tensor(x, y)
            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            total_loss += loss.item() * y.size(0)
            if count:
                total_len += 1

        try:
            avg_loss = total_loss / total_len
        except ZeroDivisionError:
            raise TypeError("Empty Dataset")

        self._log_eval(avg_loss, test)
        return avg_loss

    def train(self, data=None):

        result = self._train(data)
        self._current_epoch += 1
        return result

    def evaluate(self, data=None):

        return self._evaluate(data, test=False)

    def test(self, data=None):

        return self._evaluate(data, test=True)

    def step(self, train_data=None, val_data=None):

        self._require_context()

        self._log_step(self._current_epoch + 1)

        train_loss = self._train(train_data)
        self._save()

        if val_data or self.val_iter:
            val_data = val_data if val_data is not None else None
            test_loss = self._evaluate(val_data, test=False)

            # Save the model having the smallest validation loss
            if test_loss < self._best_loss and self.save_and_load:
                self._best_loss = test_loss
                self._save(self.snapshot_dir / f'best_checkpoint_epoch_{str(self._current_epoch + 1).zfill(3)}.pt')
                for path in sorted(glob.glob(str(self.snapshot_dir / 'best_checkpoint_epoch_*.pt')))[:-3]:
                    os.remove(path)

        else:
            test_loss = None

        self._current_epoch += 1

        return train_loss, test_loss

    def run(self, train_data=None, val_data=None, split_result=False):

        assert self.total_epoch is not None, "You should set total epoch to use run() method."

        with self._with_context():

            self._current_run_result = result = []
            self._current_epoch = 0
            self._log_start()
            self._timer_start()

            try:
                while self._current_epoch < self.total_epoch:
                    result.append(self.step(train_data, val_data))

            finally:
                self._timer_stop()
                self._log_stop()
                self._current_run_result = None

                if self.save_and_load and self._current_epoch and (val_data or self.val_iter):
                    self._load()

            if split_result:
                return [t for t, v in result], [v for t, v in result]
            else:
                return result

    __call__ = fit = run

    #
    # State dictionary handler: used in saving and loading parameters
    #

    def state_dict(self):

        from collections import OrderedDict
        state_dict = OrderedDict()
        state_dict['epoch'] = self._current_epoch
        state_dict['best_loss'] = self._best_loss
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        if isinstance(self.criterion, torch.nn.Module):
            state_dict['criterion'] = self.criterion.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):

        self._current_epoch = state_dict['epoch']
        self._best_loss = state_dict['best_loss']
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.load_state_dict(state_dict['criterion'])

    #
    # Device-moving Methods
    #

    def to(self, *args, **kwargs):  # overwrite this in subclass, for further features

        super().to(*args, **kwargs)
        self._to_apply_module(self.model)
        if isinstance(self.criterion, torch.nn.Module):
            self._to_apply_module(self.criterion)
        return self

    # Internal Parameter Methods

    def _load(self, fn=None):
        self._require_context()
        if self.save_and_load:
            self.load_state_dict(torch.load(str(fn or self._processing_fn)))

    def _save(self, fn=None):
        self._require_context()
        if self.save_and_load:
            torch.save(self.state_dict(), str(fn or self._processing_fn))

    # Internal Context Methods

    def _open(self):
        prev = self._closed
        if prev:
            if self.save_and_load:
                self._processing_fn = str(self.snapshot_dir / f'_processing_{id(self)}.pt')
                self.snapshot_dir.mkdir(exist_ok=True)
        self._closed = False
        return prev

    def _close(self, prev: bool = True):
        if self._closed:
            return
        if prev:
            if self.save_and_load:
                try:
                    if self._processing_fn is not None:
                        os.remove(self._processing_fn)
                except FileNotFoundError:
                    pass
                self._processing_fn = None
        self._closed = prev

    # Properties

    @property
    def best_loss(self):
        return self._best_loss

    # Log function: overwrite this to use custom logging hook (e.g. StringIO)

    log_function = staticmethod(print)
