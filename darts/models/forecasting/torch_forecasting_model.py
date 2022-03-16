"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""

import os
import threading
import itertools
import warnings
import queue
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union

import multiprocessing as python_multiprocessing
import torch
import torch.multiprocessing as multiprocessing
from torch._utils import ExceptionWrapper
from torch._six import string_classes

from torch.utils.data import IterDataPipe, IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler, Dataset
from torch.utils.data import _utils

import torch.utils.data.graph_settings

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[T]], Any]


# These functions used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate: _collate_fn_t = _utils.collate.default_collate
default_convert = _utils.collate.default_convert

get_worker_info = _utils.worker.get_worker_info


class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.
    Used as sampler for :class:`~torch.utils.data.IterableDataset`.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self):
        super(_InfiniteConstantSampler, self).__init__(None)

    def __iter__(self):
        while True:
            yield None



class DataLoader(Generic[T_co]):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        generator (torch.Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            `base_seed` for workers. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of samples loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers samples prefetched across all workers. (default: ``2``)
        persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.

    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.
    """
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    prefetch_factor: int
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if num_workers == 0 and prefetch_factor != 2:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing.')
        assert prefetch_factor > 0

        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        
        #=#=#=#=# INTERNAL #=#=#=#=#
        print("Location: torch_forecasting_model.py -> DataLoader(...)")
        print("Dataset is saved and looks like this:")
        print(self.dataset)

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and IterableDataset ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if isinstance(dataset, IterDataPipe):
                torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)
            elif shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))

            if sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            self._dataset_kind = _DatasetKind.Map



        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                else:
                    sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, string_classes):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            ('multiprocessing_context option '
                             'should specify a valid start method in {!r}, but got '
                             'multiprocessing_context={!r}').format(valid_start_methods, multiprocessing_context))
                    # error: Argument 1 to "get_context" has incompatible type "Union[str, bytes]"; expected "str"  [arg-type]
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)  # type: ignore[arg-type]

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError(('multiprocessing_context option should be a valid context '
                                     'object or a string specifying the start method, but got '
                                     'multiprocessing_context={}').format(multiprocessing_context))
            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> '_BaseDataLoaderIter':
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one naively
            # does multi-processing data loading, since the samples will be duplicated.
            # However, no real use case should be actually using that behavior, so
            # it should count as a user error. We should generally trust user
            # code to do the proper thing (e.g., configure each replica differently
            # in `__iter__`), and give us the correct `__len__` if they choose to
            # implement it (this will still throw if the dataset does not implement
            # a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on the
            # `DataLoader`, save the returned value in `self._len_called`, and warn
            # if the iterator ends up yielding more than this number of samples.

            # Cannot statically verify that dataset is Sized
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
            if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)

    def check_worker_number_rationality(self):
        # This function check whether the dataloader's worker number is rational based on
        # current system's resource. Current rule is that if the number of workers this
        # Dataloader will create is bigger than the number of logical cpus that is allowed to
        # use, than we will pop up a warning to let user pay attention.
        #
        # eg. If current system has 2 physical CPUs with 16 cores each. And each core support 2
        #     threads, then the total logical cpus here is 2 * 16 * 2 = 64. Let's say current
        #     DataLoader process can use half of them which is 32, then the rational max number of
        #     worker that initiated from this process is 32.
        #     Now, let's say the created DataLoader has num_works = 40, which is bigger than 32.
        #     So the warning message is triggered to notify the user to lower the worker number if
        #     necessary.
        #
        #
        # [Note] Please note that this function repects `cpuset` only when os.sched_getaffinity is
        #        available (available in most of Linux system, but not OSX and Windows).
        #        When os.sched_getaffinity is not available, os.cpu_count() is called instead, but
        #        it doesn't repect cpuset.
        #        We don't take threading into account since each worker process is single threaded
        #        at this time.
        #
        #        We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
        #        other than `torch.set_num_threads` to 1 in the worker process, if the passing
        #        in functions use 3rd party modules that rely on those threading flags to determine
        #        how many thread to create (eg. numpy, etc), then it is caller's responsibility to
        #        set those flags correctly.
        def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):

            suggested_max_worker_msg = ((
                "Our suggested max number of worker in current system is {}{}, which is smaller "
                "than what this DataLoader is going to create.").format(
                    num_worker_suggest,
                    ("" if cpuset_checked else " (`cpuset` is not taken into account)"))
            ) if num_worker_suggest is not None else (
                "DataLoader is not able to compute a suggested max number of worker in current system.")

            warn_msg = (
                "This DataLoader will create {} worker processes in total. {} "
                "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary.").format(
                    num_worker_created,
                    suggested_max_worker_msg)
            return warn_msg

        if not self.num_workers or self.num_workers == 0:
            return

        # try to compute a suggested max number of worker based on system's resource
        max_num_worker_suggest = None
        cpuset_checked = False
        if hasattr(os, 'sched_getaffinity'):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
                cpuset_checked = True
            except Exception:
                pass
        if max_num_worker_suggest is None:
            # os.cpu_count() could return Optional[int]
            # get cpu count first and check None in order to satify mypy check
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count

        if max_num_worker_suggest is None:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))
            return

        if self.num_workers > max_num_worker_suggest:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))



class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader) -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(self.__class__.__name__)

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                self._reset()
            data = self._next_data()
            self._num_yielded += 1
            if self._dataset_kind == _DatasetKind.Iterable and \
                    self._IterableDataset_len_called is not None and \
                    self._num_yielded > self._IterableDataset_len_called:
                warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                            "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                  self._num_yielded)
                if self._num_workers > 0:
                    warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                 "IterableDataset replica at each worker. Please see "
                                 "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
                warnings.warn(warn_msg)
            return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may alreay be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children.
    #
    #      Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed (which, at least in
    #      CPython, is done via an `atexit` handler defined in
    #      `multiprocessing/util.py`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/util.py#L320-L362
    #      registered when an object requiring this mechanism is first
    #      created, e.g., `mp.Queue`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/context.py#L100-L103
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/queues.py#L29
    #      )
    #
    #      So in `__del__`, we check if `_utils.python_exit_status` is set or
    #      `None` (freed), and perform no-op if so.
    #
    #      However, simply letting library clean-up codes run can also be bad,
    #      because such codes (i.e., `multiprocessing.util._exit_function()`)
    #      include join putting threads for `mp.Queue`, which can be blocking.
    #      Hence, the main process putting threads are called with
    #      `cancel_join_thread` at creation.  See later section
    #      [ 3b. A process won't hang when putting into a queue; ]
    #      for more details.
    #
    #      Here are two example cases where library clean-up codes can run
    #      before `__del__` is called:
    #
    #        1. If we hold onto a reference to the iterator, it more often
    #           than not tries to do `multiprocessing` library cleaning before
    #           clearing the alive referenced objects (https://github.com/pytorch/pytorch/issues/48666)
    #           and thus prevents our cleaning-up code to run first.
    #
    #        2. A similar issue araises when a `DataLoader` is used in a subprocess.
    #           When a process ends, it shuts the all its daemonic children
    #           down with a SIGTERM (instead of joining them without a timeout).
    #           Simiarly for threads, but by a different mechanism. This fact,
    #           together with a few implementation details of multiprocessing, forces
    #           us to make workers daemonic. All of our problems arise when a
    #           DataLoader is used in a subprocess, and are caused by multiprocessing
    #           code which looks more or less like this:
    #
    #               try:
    #                   your_function_using_a_dataloader()
    #               finally:
    #                   multiprocessing.util._exit_function()
    #
    #           The joining/termination mentioned above happens inside
    #           `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #           throws, the stack trace stored in the exception will prevent the
    #           frame which uses `DataLoaderIter` to be freed. If the frame has any
    #           reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #           its  `__del__`, which starts the shutdown procedure, will not be
    #           called. That, in turn, means that workers aren't notified. Attempting
    #           to join in `_exit_function` will then result in a hang.
    #
    #           For context, `_exit_function` is also registered as an `atexit` call.
    #           So it is unclear to me (@ssnl) why this is needed in a finally block.
    #           The code dates back to 2008 and there is no comment on the original
    #           PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #           the finally block and the `atexit` registration) that explains this.
    #
    #
    #      Finally, another choice is to just shutdown workers with logic in 1
    #      above whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can still
    #           happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we set a timeout whenever we try to get data
    #           from `data_queue`, and check the workers' status on each timeout
    #           and error.
    #           See `_DataLoaderiter._get_batch()` and
    #           `_DataLoaderiter._try_get_data()` for details.
    #
    #           Additionally, for child exit on non-Windows platforms, we also
    #           register a SIGCHLD handler (which is supported on Windows) on
    #           the main process, which checks if any of the workers fail in the
    #           (Python) handler. This is more efficient and faster in detecting
    #           worker failures, compared to only using the above mechanism.
    #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `_utils.worker.ManagerWatchdog` class
    #               checks the status of the main process.
    #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #               check `pin_memory_thread` status periodically until `.get()`
    #               returns or see that `pin_memory_thread` died.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread is
    #           daemonic and usually automatically joined when the process
    #           *exits*.
    #
    #           In case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever.  The usual
    #           solution for this in Python is calling  `q.cancel_join_thread`,
    #           which prevents automatically joining it when finalizing
    #           (exiting).
    #
    #           Nonetheless, `cancel_join_thread` must only be called when the
    #           queue is **not** going to be read from or write into by another
    #           process, because it may hold onto a lock or leave corrupted data
    #           in the queue, leading other readers/writers to hang.
    #
    #           Hence,
    #             + For worker processes, we only do so (for their output
    #               queues, i.e., `worker_result_queue`) before exiting.
    #             + For `pin_memory_thread`, its output queue `data_queue` is a
    #               `queue.Queue` that does blocking `put` if the queue is full.
    #               So there is no above problem, but as a result, in
    #               `_pin_memory_loop`, we do need to  wrap the `put` in a loop
    #               that breaks not only upon success, but also when the main
    #               process stops reading, i.e., is shutting down.
    #             + For loader process, we `cancel_join_thread()` for all
    #               `_index_queues` because the whole purpose of workers and
    #               `pin_memory_thread` is to serve the loader process.  If
    #               loader process is already exiting, we don't really care if
    #               the queues are corrupted.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `workers_done_event`:
    #   A `multiprocessing.Event` shared among the main process and all worker
    #   processes. This is used to signal the workers that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # `pin_memory_thread_done_event`:
    #   A `threading.Event` for a similar purpose to that of
    #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
    #   that separate events are needed is that `pin_memory_thread` reads from
    #   the output queue of the workers. But the workers, upon seeing that
    #   `workers_done_event` is set, only wants to see the final `None`, and is
    #   not required to flush all data in the output queue (e.g., it may call
    #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
    #   happens to exhaust coincidentally, which is out of the control of the
    #   main process). Thus, since we will exit `pin_memory_thread` before the
    #   workers (see below), two separete events are used.
    #
    # NOTE: In short, the protocol is that the main process will set these
    #       `done_event`s and then the corresponding processes/threads a `None`,
    #       and that they may exit at any time after receiving the `None`.
    #
    # NOTE: Using `None` as the final signal is valid, since normal data will
    #       always be a 2-tuple with the 1st element being the index of the data
    #       transferred (different from dataset index/key), and the 2nd being
    #       either the dataset key or the data sample (depending on which part
    #       of the data model the queue is at).
    #
    # [ worker processes ]
    #   While loader process is alive:
    #     Get from `index_queue`.
    #       If get anything else,
    #          Check `workers_done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data:
    #                If is fetching from an `IterableDataset` and the iterator
    #                    is exhausted, send an `_IterableDatasetStopIteration`
    #                    object to signal iteration end. The main process, upon
    #                    receiving such an object, will send `None` to this
    #                    worker and not use the corresponding `index_queue`
    #                    anymore.
    #       If timed out,
    #          No matter `workers_done_event` is set (still need to see `None`)
    #          or not, must continue to next iteration.
    #   (outside loop)
    #   If `workers_done_event` is set,  (this can be False with `IterableDataset`)
    #     `data_queue.cancel_join_thread()`.  (Everything is ending here:
    #                                          main process won't read from it;
    #                                          other workers will also call
    #                                          `cancel_join_thread`.)
    #
    # [ pin_memory_thread ]
    #   # No need to check main thread. If this thread is alive, the main loader
    #   # thread must be alive, because this thread is set as daemonic.
    #   While `pin_memory_thread_done_event` is not set:
    #     Get from `index_queue`.
    #       If timed out, continue to get in the next iteration.
    #       Otherwise, process data.
    #       While `pin_memory_thread_done_event` is not set:
    #         Put processed data to `data_queue` (a `queue.Queue` with blocking put)
    #         If timed out, continue to put in the next iteration.
    #         Otherwise, break, i.e., continuing to the out loop.
    #
    #   NOTE: we don't check the status of the main thread because
    #           1. if the process is killed by fatal signal, `pin_memory_thread`
    #              ends.
    #           2. in other cases, either the cleaning-up in __del__ or the
    #              automatic exit of daemonic thread will take care of it.
    #              This won't busy-wait either because `.get(timeout)` does not
    #              busy-wait.
    #
    # [ main process ]
    #   In the DataLoader Iter's `__del__`
    #     b. Exit `pin_memory_thread`
    #          i.   Set `pin_memory_thread_done_event`.
    #          ii   Put `None` in `worker_result_queue`.
    #          iii. Join the `pin_memory_thread`.
    #          iv.  `worker_result_queue.cancel_join_thread()`.
    #
    #     c. Exit the workers.
    #          i.   Set `workers_done_event`.
    #          ii.  Put `None` in each worker's `index_queue`.
    #          iii. Join the workers.
    #          iv.  Call `.cancel_join_thread()` on each worker's `index_queue`.
    #
    #        NOTE: (c) is better placed after (b) because it may leave corrupted
    #              data in `worker_result_queue`, which `pin_memory_thread`
    #              reads from, in which case the `pin_memory_thread` can only
    #              happen at timeing out, which is slow. Nonetheless, same thing
    #              happens if a worker is killed by signal at unfortunate times,
    #              but in other cases, we are better off having a non-corrupted
    #              `worker_result_queue` for `pin_memory_thread`.
    #
    #   NOTE: If `pin_memory=False`, there is no `pin_memory_thread` and (b)
    #         can be omitted
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already shutting
    #     down.

    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      torch.cuda.current_device(),
                      self._pin_memory_thread_done_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit
            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore[misc]
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`,
        # the worker will be reset to available in the next epoch.
        self._workers_status = [True for i in range(self._num_workers)]
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration())
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _utils.worker._ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            if isinstance(e, queue.Empty):
                return (False, None)
            import tempfile
            import errno
            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise

# NOTE [ DataLoader on Linux and open files limit ]
#
# On Linux when DataLoader is used with multiprocessing we pass the data between
# the root process and the workers through SHM files. We remove those files from
# the filesystem as soon as they are created and keep them alive by
# passing around their file descriptors through AF_UNIX sockets. (See
# docs/source/multiprocessing.rst and 'Multiprocessing Technical Notes` in
# the wiki (https://github.com/pytorch/pytorch/wiki).)
#
# This sometimes leads us to exceeding the open files limit. When that happens,
# and the offending file descriptor is coming over a socket, the `socket` Python
# package silently strips the file descriptor from the message, setting only the
# `MSG_CTRUNC` flag (which might be a bit misleading since the manpage says that
# it _indicates that some control data were discarded due to lack of space in
# the buffer for ancillary data_). This might reflect the C implementation of
# AF_UNIX sockets.
#
# This behaviour can be reproduced with the script and instructions at the
# bottom of this note.
#
# When that happens, the standard Python `multiprocessing` (and not
# `torch.multiprocessing`) raises a `RuntimeError: received 0 items of ancdata`
#
# Sometimes, instead of the FD being stripped, you may get an `OSError:
# Too many open files`, both in the script below and in DataLoader. However,
# this is rare and seems to be nondeterministic.
#
#
#   #!/usr/bin/env python3
#   import sys
#   import socket
#   import os
#   import array
#   import shutil
#   import socket
#
#
#   if len(sys.argv) != 4:
#       print("Usage: ", sys.argv[0], " tmp_dirname iteration (send|recv)")
#       sys.exit(1)
#
#   if __name__ == '__main__':
#       dirname = sys.argv[1]
#       sock_path = dirname + "/sock"
#       iterations = int(sys.argv[2])
#       def dummy_path(i):
#           return dirname + "/" + str(i) + ".dummy"
#
#
#       if sys.argv[3] == 'send':
#           while not os.path.exists(sock_path):
#               pass
#           client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
#           client.connect(sock_path)
#           for i in range(iterations):
#               fd = os.open(dummy_path(i), os.O_WRONLY | os.O_CREAT)
#               ancdata = array.array('i', [fd])
#               msg = bytes([i % 256])
#               print("Sending fd ", fd, " (iteration #", i, ")")
#               client.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancdata)])
#
#
#       else:
#           assert sys.argv[3] == 'recv'
#
#           if os.path.exists(dirname):
#               raise Exception("Directory exists")
#
#           os.mkdir(dirname)
#
#           print("Opening socket...")
#           server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
#           server.bind(sock_path)
#
#           print("Listening...")
#           for i in range(iterations):
#               a = array.array('i')
#               msg, ancdata, flags, addr = server.recvmsg(1, socket.CMSG_SPACE(a.itemsize))
#               assert(len(ancdata) == 1)
#               cmsg_level, cmsg_type, cmsg_data = ancdata[0]
#               a.frombytes(cmsg_data)
#               print("Received fd ", a[0], " (iteration #", i, ")")
#
#           shutil.rmtree(dirname)
#
# Steps to reproduce:
#
# 1. Run two shells and set lower file descriptor limit in the receiving one:
# (shell1) ulimit -n 1020
# (shell2) ulimit -n 1022
#
# 2. Run the script above with the `recv` option in the first shell
# (shell1) ./test_socket.py sock_tmp 1017 recv
#
# 3. Run the script with the `send` option in the second shell:
# (shell2) ./test_socket.py sock_tmp 1017 send

    def _get_data(self):
        # Fetches data from `self._data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _try_put_index(self):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False

        assert self._workers_done_event.is_set() == shutdown

    def _shutdown_workers(self):
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Normal exit when last reference is gone / iterator is depleted.
                # See (1) and the second half of the note.

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    # We should be able to join here, but in case anything went
                    # wrong, we set a timeout and if the workers fail to join,
                    # they are killed in the `finally` block.
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()

    # staticmethod is used to remove reference to `_MultiProcessingDataLoaderIter`
    @staticmethod
    def _clean_up_worker(w):
        try:
            w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        finally:
            if w.is_alive():
                w.terminate()

    def __del__(self):
        self._shutdown_workers()

"""
This file contains several abstract classes:

    * TorchForecastingModel is the super-class of all torch (deep learning) darts forecasting models.

    * PastCovariatesTorchModel(TorchForecastingModel) for torch models consuming only past-observed covariates.
    * FutureCovariatesTorchModel(TorchForecastingModel) for torch models consuming only future values of
      future covariates.
    * DualCovariatesTorchModel(TorchForecastingModel) for torch models consuming past and future values of some single
      future covariates.
    * MixedCovariatesTorchModel(TorchForecastingModel) for torch models consuming both past-observed
      as well as past and future values of some future covariates.
    * SplitCovariatesTorchModel(TorchForecastingModel) for torch models consuming past-observed as well as future
      values of some future covariates.

    * TorchParametricProbabilisticForecastingModel(TorchForecastingModel) is the super-class of all probabilistic torch
      forecasting models.
"""

import datetime
import inspect
import os
import shutil
from abc import ABC, abstractmethod
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch import Tensor
#from torch.utils.data import DataLoader
#from darts.utils.dataloader import DataLoader

from darts.logging import (
    get_logger,
    raise_deprecation_warning,
    raise_if,
    raise_if_not,
    raise_log,
    suppress_lightning_warnings,
)
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.timeseries import TimeSeries
from darts.utils.data.encoders import SequentialEncoder
from darts.utils.data.inference_dataset import (
    DualCovariatesInferenceDataset,
    FutureCovariatesInferenceDataset,
    InferenceDataset,
    MixedCovariatesInferenceDataset,
    PastCovariatesInferenceDataset,
    SplitCovariatesInferenceDataset,
)
from darts.utils.data.sequential_dataset import (
    DualCovariatesSequentialDataset,
    FutureCovariatesSequentialDataset,
    MixedCovariatesSequentialDataset,
    PastCovariatesSequentialDataset,
    SplitCovariatesSequentialDataset,
)
from darts.utils.data.training_dataset import (
    DualCovariatesTrainingDataset,
    FutureCovariatesTrainingDataset,
    MixedCovariatesTrainingDataset,
    PastCovariatesTrainingDataset,
    SplitCovariatesTrainingDataset,
    TrainingDataset,
)
from darts.utils.likelihood_models import Likelihood
from darts.utils.torch import random_method

DEFAULT_DARTS_FOLDER = "darts_logs"
CHECKPOINTS_FOLDER = "checkpoints"
RUNS_FOLDER = "runs"
INIT_MODEL_NAME = "_model.pth.tar"

logger = get_logger(__name__)


def _get_checkpoint_folder(work_dir, model_name):
    return os.path.join(work_dir, model_name, CHECKPOINTS_FOLDER)


def _get_logs_folder(work_dir, model_name):
    return os.path.join(work_dir, model_name)


def _get_runs_folder(work_dir, model_name):
    return os.path.join(work_dir, model_name)


def _get_checkpoint_fname(work_dir, model_name, best=False):
    checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
    path = os.path.join(checkpoint_dir, "best-*" if best else "last-*")

    checklist = glob(path)
    if len(checklist) == 0:
        raise_log(
            FileNotFoundError(
                "There is no file matching prefix {} in {}".format(
                    "best-*" if best else "last-*", checkpoint_dir
                )
            ),
            logger,
        )

    file_name = max(checklist, key=os.path.getctime)
    return os.path.basename(file_name)


class TorchForecastingModel(GlobalForecastingModel, ABC):
    @random_method
    def __init__(
        self,
        batch_size: int = 32,
        n_epochs: int = 100,
        model_name: str = None,
        work_dir: str = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER),
        log_tensorboard: bool = False,
        nr_epochs_val_period: int = 1,
        torch_device_str: Optional[str] = None,
        force_reset: bool = False,
        save_checkpoints: bool = False,
        add_encoders: Optional[Dict] = None,
        random_state: Optional[int] = None,
        pl_trainer_kwargs: Optional[Dict] = None,
        show_warnings: bool = False,
    ):

        """Pytorch Lightning (PL)-based Forecasting Model.

        This class is meant to be inherited to create a new PL-based forecasting model.
        It governs the interactions between:
            - Darts forecasting models (module) :class:`PLTorchForecastingModel`
            - Darts integrated PL Lightning Trainer :class:`pytorch_lightning.Trainer` or custom PL Trainers
            - Dataset loaders :class:`TrainingDataset` and :class:`InferenceDataset` or custom Dataset Loaders.

        When subclassing this class, please make sure to set the self.model attribute
        in the __init__ function and then call super().__init__ while passing the kwargs.

        Parameters
        ----------
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH:MM:SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09:53:32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
        torch_device_str
            Optionally, a string indicating the torch device to use. By default, ``torch_device_str`` is ``None``
            which will run on CPU. Set it to ``"cuda"`` to use all available GPUs or ``"cuda:i"`` to only use
            GPU ``i`` (``i`` must be an integer). For example "cuda:0" will use the first GPU only.

            .. deprecated:: v0.17.0
                ``torch_device_str`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, specify this with keys ``"accelerator", "gpus", "auto_select_gpus"`` in your
                ``pl_trainer_kwargs`` dict. Some examples for setting the devices inside the ``pl_trainer_kwargs``
                dict:

                - ``{"accelerator": "cpu"}`` for CPU,
                - ``{"accelerator": "gpu", "gpus": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
                - ``{"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True}`` to use all available GPUS.

                For more info, see here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
                https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices
        force_reset
            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded). Default: ``False``.
        save_checkpoints
            Whether or not to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save_model()` and loaded using :func:`load_model()`. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.utils.data.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['absolute'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.
            Default: ``None``.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the
            supported kwargs. Default: ``None``.
            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_

            .. highlight:: python
            .. code-block:: python

                from pytorch_lightning.callbacks.early_stopping import EarlyStopping

                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )

                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..

            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.
        """
        super().__init__()
        suppress_lightning_warnings(suppress_all=not show_warnings)

        # We will fill these dynamically, upon first call of fit_from_dataset():
        self.model: Optional[PLForecastingModule] = None
        self.train_sample: Optional[Tuple] = None
        self.output_dim: Optional[int] = None

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # by default models do not use encoders
        self.add_encoders = add_encoders
        self.encoders: Optional[SequentialEncoder] = None

        # get model name and work dir
        if model_name is None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
            model_name = current_time + "_torch_model_run_" + str(os.getpid())

        self.model_name = model_name
        self.work_dir = work_dir

        # setup model save dirs
        self.save_checkpoints = save_checkpoints
        checkpoints_folder = _get_checkpoint_folder(self.work_dir, self.model_name)
        log_folder = _get_logs_folder(self.work_dir, self.model_name)
        checkpoint_exists = (
            os.path.exists(checkpoints_folder)
            and len(glob(os.path.join(checkpoints_folder, "*"))) > 0
        )

        # setup model save dirs
        if checkpoint_exists and save_checkpoints:
            raise_if_not(
                force_reset,
                f"Some model data already exists for `model_name` '{self.model_name}'. Either load model to continue "
                f"training or use `force_reset=True` to initialize anyway to start training from scratch and remove "
                f"all the model data",
                logger,
            )
            self.reset_model()
        elif save_checkpoints:
            self._create_save_dirs()
        else:
            pass

        # TODO: remove below in the next version ======>
        accelerator, gpus, auto_select_gpus = self._extract_torch_devices(
            torch_device_str
        )
        # TODO: until here <======

        # save best epoch on val_loss and last epoch under 'darts_logs/model_name/checkpoints/'
        if save_checkpoints:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=checkpoints_folder,
                save_last=True,
                monitor="val_loss",
                filename="best-{epoch}-{val_loss:.2f}",
            )
            checkpoint_callback.CHECKPOINT_NAME_LAST = "last-{epoch}"
        else:
            checkpoint_callback = None

        # save tensorboard under 'darts_logs/model_name/logs/'
        model_logger = (
            pl_loggers.TensorBoardLogger(save_dir=log_folder, name="", version="logs")
            if log_tensorboard
            else False
        )

        # setup trainer parameters from model creation parameters
        self.trainer_params = {
            "accelerator": accelerator,
            "gpus": gpus,
            "auto_select_gpus": auto_select_gpus,
            "logger": model_logger,
            "max_epochs": n_epochs,
            "check_val_every_n_epoch": nr_epochs_val_period,
            "enable_checkpointing": save_checkpoints,
            "callbacks": [cb for cb in [checkpoint_callback] if cb is not None],
        }

        # update trainer parameters with user defined `pl_trainer_kwargs`
        if pl_trainer_kwargs is not None:
            pl_trainer_kwargs_copy = {
                key: val for key, val in pl_trainer_kwargs.items()
            }
            self.n_epochs = pl_trainer_kwargs_copy.get("max_epochs", self.n_epochs)
            self.trainer_params["callbacks"] += pl_trainer_kwargs_copy.pop(
                "callbacks", []
            )
            self.trainer_params = dict(self.trainer_params, **pl_trainer_kwargs_copy)

        # pytorch lightning trainer will be created at training time
        self.trainer: Optional[pl.Trainer] = None
        self.load_ckpt_path: Optional[str] = None

        # pl_module_params must be set in __init__ method of TorchForecastingModel subclass
        self.pl_module_params: Optional[Dict] = None

    @staticmethod
    def _extract_torch_devices(
        torch_device_str,
    ) -> Tuple[str, Optional[Union[list, int]], bool]:
        """This method handles the deprecated `torch_device_str` and should be removed in a future Darts version.

        Returns
        -------
        Tuple
            (accelerator, gpus, auto_select_gpus)
        """

        if torch_device_str is None:
            return "cpu", None, False

        device_warning = (
            "`torch_device_str` is deprecated and will be removed in a coming Darts version. For full support "
            "of all torch devices, use PyTorch-Lightnings trainer flags and pass them inside "
            "`pl_trainer_kwargs`. Flags of interest are {`accelerator`, `gpus`, `auto_select_gpus`, `devices`}. "
            "For more information, visit "
            "https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags"
        )
        raise_deprecation_warning(device_warning, logger)
        # check torch device
        raise_if_not(
            any(
                [
                    device_str in torch_device_str
                    for device_str in ["cuda", "cpu", "auto"]
                ]
            ),
            f"unknown torch_device_str `{torch_device_str}`. String must contain one of `('cuda', 'cpu', 'auto') "
            + device_warning,
            logger,
        )
        device_split = torch_device_str.split(":")

        gpus = None
        auto_select_gpus = False
        accelerator = "gpu" if device_split[0] == "cuda" else device_split[0]

        if len(device_split) == 2 and accelerator == "gpu":
            gpus = device_split[1]
            gpus = [int(gpus)]
        elif len(device_split) == 1:
            if accelerator == "gpu":
                gpus = -1
                auto_select_gpus = True
        else:
            raise_if(
                True,
                f"unknown torch_device_str `{torch_device_str}`. " + device_warning,
                logger,
            )
        return accelerator, gpus, auto_select_gpus

    @classmethod
    def _validate_model_params(cls, **kwargs):
        """validate that parameters used at model creation are part of :class:`TorchForecastingModel`,
        :class:`PLForecastingModule` or cls __init__ methods.
        """
        valid_kwargs = (
            set(inspect.signature(TorchForecastingModel.__init__).parameters.keys())
            | set(inspect.signature(PLForecastingModule.__init__).parameters.keys())
            | set(inspect.signature(cls.__init__).parameters.keys())
        )

        invalid_kwargs = [kwarg for kwarg in kwargs if kwarg not in valid_kwargs]

        raise_if(
            len(invalid_kwargs) > 0,
            f"Invalid model creation parameters. Model `{cls.__name__}` has no args/kwargs `{invalid_kwargs}`",
            logger=logger,
        )

    @classmethod
    def _extract_torch_model_params(cls, **kwargs):
        """extract params from model creation to set up TorchForecastingModels"""
        cls._validate_model_params(**kwargs)
        get_params = list(
            inspect.signature(TorchForecastingModel.__init__).parameters.keys()
        )
        get_params.remove("self")
        return {kwarg: kwargs.get(kwarg) for kwarg in get_params if kwarg in kwargs}

    @staticmethod
    def _extract_pl_module_params(**kwargs):
        """Extract params from model creation to set up PLForecastingModule (the actual torch.nn.Module)"""
        get_params = list(
            inspect.signature(PLForecastingModule.__init__).parameters.keys()
        )
        get_params.remove("self")
        return {kwarg: kwargs.get(kwarg) for kwarg in get_params if kwarg in kwargs}

    def _create_save_dirs(self):
        """Create work dir and model dir"""
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)
        if not os.path.exists(_get_runs_folder(self.work_dir, self.model_name)):
            os.mkdir(_get_runs_folder(self.work_dir, self.model_name))

    def _remove_save_dirs(self):
        shutil.rmtree(
            _get_runs_folder(self.work_dir, self.model_name), ignore_errors=True
        )

    def reset_model(self):
        """Resets the model object and removes all stored data - model, checkpoints, loggers and training history."""
        self._remove_save_dirs()
        self._create_save_dirs()

        self.model = None
        self.trainer = None
        self.train_sample = None

    def _init_model(self, trainer: Optional[pl.Trainer] = None) -> None:
        """Initializes model and trainer based on examples of input/output tensors (to get the sizes right):"""

        raise_if(
            self.pl_module_params is None,
            "`pl_module_params` must be extracted in __init__ method of `TorchForecastingModel` subclass after "
            "calling `super.__init__(...)`. Do this with `self._extract_pl_module_params(**self.model_params).`",
        )

        # the tensors have shape (chunk_length, nr_dimensions)
        self.model = self._create_model(self.train_sample)

        precision = None
        dtype = self.train_sample[0].dtype
        if np.issubdtype(dtype, np.float32):
            logger.info("Time series values are 32-bits; casting model to float32.")
            precision = 32
        elif np.issubdtype(dtype, np.float64):
            logger.info("Time series values are 64-bits; casting model to float64.")
            precision = 64

        precision_user = (
            self.trainer_params.get("precision", None)
            if trainer is None
            else trainer.precision
        )
        raise_if(
            precision_user is not None and precision_user != precision,
            f"User-defined trainer_kwarg `precision={precision_user}`-bit does not match dtype: `{dtype}` of the "
            f"underlying TimeSeries. Set `precision` to `{precision}` or cast your data to `{precision_user}-"
            f"bit` with `TimeSeries.astype(np.float{precision_user})`.",
            logger,
        )

        self.trainer_params["precision"] = precision

        # we need to save the initialized TorchForecastingModel as PyTorch-Lightning only saves module checkpoints
        if self.save_checkpoints:
            self.save_model(
                os.path.join(
                    _get_runs_folder(self.work_dir, self.model_name), INIT_MODEL_NAME
                )
            )

    def _setup_trainer(
        self, trainer: Optional[pl.Trainer], verbose: bool, epochs: int = 0
    ) -> None:
        """Sets up the PyTorch-Lightning trainer for training or prediction."""

        self.trainer_params["enable_model_summary"] = (
            verbose if self.model.epochs_trained == 0 else False
        )
        self.trainer_params["enable_progress_bar"] = verbose

        self.trainer = (
            self._init_trainer(trainer_params=self.trainer_params, max_epochs=epochs)
            if trainer is None
            else trainer
        )

    @staticmethod
    def _init_trainer(
        trainer_params: Dict, max_epochs: Optional[int] = None
    ) -> pl.Trainer:
        """Initializes the PyTorch-Lightning trainer for training or prediction from `trainer_params`."""
        trainer_params_copy = {param: val for param, val in trainer_params.items()}
        if max_epochs is not None:
            trainer_params_copy["max_epochs"] = max_epochs

        return pl.Trainer(**trainer_params_copy)

    @abstractmethod
    def _create_model(self, train_sample: Tuple[Tensor]) -> torch.nn.Module:
        """
        This method has to be implemented by all children. It is in charge of instantiating the actual torch model,
        based on examples input/output tensors (i.e. implement a model with the right input/output sizes).
        """
        pass

    @abstractmethod
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> TrainingDataset:
        """
        Each model must specify the default training dataset to use.
        """
        pass

    @abstractmethod
    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> InferenceDataset:
        """
        Each model must specify the default training dataset to use.
        """
        pass

    @abstractmethod
    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        """
        Verify that the provided train dataset is of the correct type
        """
        pass

    @abstractmethod
    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        """
        Verify that the provided inference dataset is of the correct type
        """
        pass

    @abstractmethod
    def _verify_predict_sample(self, predict_sample: Tuple):
        """
        verify that the (first) sample contained in the inference dataset matches the model type and the
        data the model has been trained on.
        """
        pass

    @abstractmethod
    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        """
        Verify that any non-None covariates comply with the model type.
        """
        pass

    @random_method
    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        max_samples_per_ts: Optional[int] = None,
        num_loader_workers: int = 0,
    ):
        """Fit/train the model on one or multiple series.

        This method wraps around :func:`fit_from_dataset()`, constructing a default training
        dataset for this model. If you need more control on how the series are sliced for training, consider
        calling :func:`fit_from_dataset()` with a custom :class:`darts.utils.data.TrainingDataset`.

        Training is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        This function can be called several times to do some extra training. If ``epochs`` is specified, the model
        will be trained for some (extra) ``epochs`` epochs.

        Below, all possible parameters are documented, but not all models support all parameters. For instance,
        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.
        Darts will complain if you try fitting a model with the wrong covariates argument.

        When handling covariates, Darts will try to use the time axes of the target and the covariates
        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes
        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.

        Parameters
        ----------
        series
            A series or sequence of series serving as target (i.e. what the model will be trained to forecast)
        past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates
        future_covariates
            Optionally, a series or sequence of series specifying future-known covariates
        val_series
            Optionally, one or a sequence of validation target series, which will be used to compute the validation
            loss throughout training and keep track of the best performing models.
        val_past_covariates
            Optionally, the past covariates corresponding to the validation series (must match ``covariates``)
        val_future_covariates
            Optionally, the future covariates corresponding to the validation series (must match ``covariates``)
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform training. Using a custom ``trainer`` will
            override Darts' default trainer.
        verbose
            Optionally, whether to print progress.

            .. deprecated:: v0.17.0
                ``verbose`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, control verbosity with PyTorch Lightning Trainer parameters ``enable_progress_bar``,
                ``progress_bar_refresh_rate`` and ``enable_model_summary`` in the ``pl_trainer_kwargs`` dict
                at model creation. See for example here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#enable-progress-bar
        epochs
            If specified, will train the model for ``epochs`` (additional) epochs, irrespective of what ``n_epochs``
            was provided to the model constructor.
        max_samples_per_ts
            Optionally, a maximum number of samples to use per time series. Models are trained in a supervised fashion
            by constructing slices of (input, output) examples. On long time series, this can result in unnecessarily
            large number of training samples. This parameter upper-bounds the number of training samples per time
            series (taking only the most recent samples in each series). Leaving to None does not apply any
            upper bound.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            both for the training and validation loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.

        Returns
        -------
        self
            Fitted model.
        """
        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # TODO: also check the validation covariates
        self._verify_past_future_covariates(
            past_covariates=past_covariates, future_covariates=future_covariates
        )

        def wrap_fn(
            ts: Union[TimeSeries, Sequence[TimeSeries]]
        ) -> Sequence[TimeSeries]:
            return [ts] if isinstance(ts, TimeSeries) else ts

        series = wrap_fn(series)
        print("#=#=#=#=# INTERNAL #=#=#=#=#")
        print("Location: torch_forecasting_model.py -> fit(...)")
        print("Dataset is loaded into wrap_fn() and looks like this: ")
        print(series)
        print("#=#=#=#=# INTERNAL #=#=#=#=#")
        past_covariates = wrap_fn(past_covariates)
        future_covariates = wrap_fn(future_covariates)
        val_series = wrap_fn(val_series)
        val_past_covariates = wrap_fn(val_past_covariates)
        val_future_covariates = wrap_fn(val_future_covariates)

        # Check that dimensions of train and val set match; on first series only
        if val_series is not None:
            match = (
                series[0].width == val_series[0].width
                and (past_covariates[0].width if past_covariates is not None else None)
                == (
                    val_past_covariates[0].width
                    if val_past_covariates is not None
                    else None
                )
                and (
                    future_covariates[0].width
                    if future_covariates is not None
                    else None
                )
                == (
                    val_future_covariates[0].width
                    if val_future_covariates is not None
                    else None
                )
            )
            raise_if_not(
                match,
                "The dimensions of the series in the training set "
                "and the validation set do not match.",
            )

        self.encoders = self.initialize_encoders()

        if self.encoders.encoding_available:
            past_covariates, future_covariates = self.encoders.encode_train(
                target=series,
                past_covariate=past_covariates,
                future_covariate=future_covariates,
            )
        train_dataset = self._build_train_dataset(
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
        )

        if val_series is not None:
            if self.encoders.encoding_available:
                val_past_covariates, val_future_covariates = self.encoders.encode_train(
                    target=val_series,
                    past_covariate=val_past_covariates,
                    future_covariate=val_future_covariates,
                )

            val_dataset = self._build_train_dataset(
                target=val_series,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
                max_samples_per_ts=max_samples_per_ts,
            )
        else:
            val_dataset = None

        logger.info(f"Train dataset contains {len(train_dataset)} samples.")

        return self.fit_from_dataset(
            train_dataset, val_dataset, trainer, verbose, epochs, num_loader_workers
        )

    @random_method
    def fit_from_dataset(
        self,
        train_dataset: TrainingDataset,
        val_dataset: Optional[TrainingDataset] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        num_loader_workers: int = 0,
    ):
        """
        Train the model with a specific :class:`darts.utils.data.TrainingDataset` instance.
        These datasets implement a PyTorch ``Dataset``, and specify how the target and covariates are sliced
        for training. If you are not sure which training dataset to use, consider calling :func:`fit()` instead,
        which will create a default training dataset appropriate for this model.

        Training is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        This function can be called several times to do some extra training. If ``epochs`` is specified, the model
        will be trained for some (extra) ``epochs`` epochs.

        Parameters
        ----------
        train_dataset
            A training dataset with a type matching this model (e.g. :class:`PastCovariatesTrainingDataset` for
            :class:`PastCovariatesTorchModel`).
        val_dataset
            A training dataset with a type matching this model (e.g. :class:`PastCovariatesTrainingDataset` for
            :class:`PastCovariatesTorchModel`s), representing the validation set (to track the validation loss).
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction. Using a custom `trainer` will
            override Darts' default trainer.
        verbose
            Optionally, whether to print progress.

            .. deprecated:: v0.17.0
                ``verbose`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, control verbosity with PyTorch Lightning Trainer parameters ``enable_progress_bar``,
                ``progress_bar_refresh_rate`` and ``enable_model_summary`` in the ``pl_trainer_kwargs`` dict
                at model creation. See for example here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#enable-progress-bar
        epochs
            If specified, will train the model for ``epochs`` (additional) epochs, irrespective of what ``n_epochs``
            was provided to the model constructor.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            both for the training and validation loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.

        Returns
        -------
        self
            Fitted model.
        """

        self._verify_train_dataset_type(train_dataset)
        raise_if(
            len(train_dataset) == 0,
            "The provided training time series dataset is too short for obtaining even one training point.",
            logger,
        )
        raise_if(
            val_dataset is not None and len(val_dataset) == 0,
            "The provided validation time series dataset is too short for obtaining even one training point.",
            logger,
        )

        train_sample = train_dataset[0]
        if self.model is None:
            # Build model, based on the dimensions of the first series in the train set.
            self.train_sample, self.output_dim = train_sample, train_sample[-1].shape[1]
            self._init_model(trainer)
        else:
            # Check existing model has input/output dims matching what's provided in the training set.
            raise_if_not(
                len(train_sample) == len(self.train_sample),
                "The size of the training set samples (tuples) does not match what the model has been "
                "previously trained on. Trained on tuples of length {}, received tuples of length {}.".format(
                    len(self.train_sample), len(train_sample)
                ),
            )
            same_dims = tuple(
                s.shape[1] if s is not None else None for s in train_sample
            ) == tuple(s.shape[1] if s is not None else None for s in self.train_sample)
            raise_if_not(
                same_dims,
                "The dimensionality of the series in the training set do not match the dimensionality"
                " of the series the model has previously been trained on. "
                "Model input/output dimensions = {}, provided input/ouptput dimensions = {}".format(
                    tuple(
                        s.shape[1] if s is not None else None for s in self.train_sample
                    ),
                    tuple(s.shape[1] if s is not None else None for s in train_sample),
                ),
            )

        # Setting drop_last to False makes the model see each sample at least once, and guarantee the presence of at
        # least one batch no matter the chosen batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_fn,
        )

        # Prepare validation data
        val_loader = (
            None
            if val_dataset is None
            else DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_loader_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self._batch_collate_fn,
            )
        )

        # if user wants to train the model for more epochs, ignore the n_epochs parameter
        train_num_epochs = epochs if epochs > 0 else self.n_epochs

        if verbose is not None:
            raise_deprecation_warning(
                "kwarg `verbose` is deprecated and will be removed in a future Darts version. "
                "Instead, control verbosity with PyTorch Lightning Trainer parameters `enable_progress_bar`, "
                "`progress_bar_refresh_rate` and `enable_model_summary` in the `pl_trainer_kwargs` dict "
                "at model creation.",
                logger,
            )
        verbose = True if verbose is None else verbose

        # setup trainer
        self._setup_trainer(trainer, verbose, train_num_epochs)

        # TODO: multiple training without loading from checkpoint is not trivial (I believe PyTorch-Lightning is still
        #  working on that, see https://github.com/PyTorchLightning/pytorch-lightning/issues/9636)
        if self.epochs_trained > 0 and not self.load_ckpt_path:
            logger.warn(
                "Attempting to retrain the model without resuming from a checkpoint. This is currently "
                "discouraged. Consider setting `save_checkpoints` to `True` and specifying `model_name` at model "
                f"creation. Then call `model = {self.__class__.__name__}.load_from_checkpoint(model_name, "
                "best=False)`. Finally, train the model with `model.fit(..., epochs=new_epochs)` where "
                "`new_epochs` is the sum of (epochs already trained + some additional epochs)."
            )

        # Train model
        self._train(train_loader, val_loader)
        return self

    def _train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader]
    ) -> None:
        """
        Performs the actual training

        Parameters
        ----------
        train_loader
            the training data loader feeding the training data and targets
        val_loader
            optionally, a validation set loader
        """

        # if model was loaded from checkpoint (when `load_ckpt_path is not None`) and model.fit() is called,
        # we resume training
        ckpt_path = self.load_ckpt_path
        self.load_ckpt_path = None

        self.trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

    @random_method
    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        trainer: Optional[pl.Trainer] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = None,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Predict the ``n`` time step following the end of the training series, or of the specified ``series``.

        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        Below, all possible parameters are documented, but not all models support all parameters. For instance,
        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.
        Darts will complain if you try calling :func:`predict()` on a model with the wrong covariates argument.

        Darts will also complain if the provided covariates do not have a sufficient time span.
        In general, not all models require the same covariates' time spans:

        * | Models relying on past covariates require the last ``input_chunk_length`` of the ``past_covariates``
          | points to be known at prediction time. For horizon values ``n > output_chunk_length``, these models
          | require at least the next ``n - output_chunk_length`` future values to be known as well.
        * | Models relying on future covariates require the next ``n`` values to be known.
          | In addition (for :class:`DualCovariatesTorchModel` and :class:`MixedCovariatesTorchModel`), they also
          | require the "historic" values of these future covariates (over the past ``input_chunk_length``).

        When handling covariates, Darts will try to use the time axes of the target and the covariates
        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes
        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        series
            Optionally, a series or sequence of series, representing the history of the target series whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        past_covariates
            Optionally, the past-observed covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        future_covariates
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction. Using a custom ``trainer``
            will override Darts' default trainer.
        batch_size
            Size of batches during prediction. Defaults to the models' training ``batch_size`` value.
        verbose
            Optionally, whether to print progress.

            .. deprecated:: v0.17.0
                ``verbose`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, control verbosity with PyTorch Lightning Trainer parameters ``enable_progress_bar``,
                ``progress_bar_refresh_rate`` and ``enable_model_summary`` in the ``pl_trainer_kwargs`` dict
                at model creation. See for example here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#enable-progress-bar
        n_jobs
            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.
        roll_size
            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set ``output_chunk_length`` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            for the inference/prediction dataset loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            One or several time series containing the forecasts of ``series``, or the forecast of the training series
            if ``series`` is not specified and the model has been trained on a single series.
        """
        super().predict(n, series, past_covariates, future_covariates)

        if series is None:
            raise_if(
                self.training_series is None,
                "Input series has to be provided after fitting on multiple series.",
            )
            series = self.training_series

        if past_covariates is None and self.past_covariate_series is not None:
            past_covariates = self.past_covariate_series
        if future_covariates is None and self.future_covariate_series is not None:
            future_covariates = self.future_covariate_series

        called_with_single_series = False
        if isinstance(series, TimeSeries):
            called_with_single_series = True
            series = [series]

        past_covariates = (
            [past_covariates]
            if isinstance(past_covariates, TimeSeries)
            else past_covariates
        )
        future_covariates = (
            [future_covariates]
            if isinstance(future_covariates, TimeSeries)
            else future_covariates
        )

        # encoders are set when calling fit(), but not when calling fit_from_dataset()
        if self.encoders is not None and self.encoders.encoding_available:
            past_covariates, future_covariates = self.encoders.encode_inference(
                n=n,
                target=series,
                past_covariate=past_covariates,
                future_covariate=future_covariates,
            )

        dataset = self._build_inference_dataset(
            target=series,
            n=n,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        predictions = self.predict_from_dataset(
            n,
            dataset,
            trainer=trainer,
            verbose=verbose,
            batch_size=batch_size,
            n_jobs=n_jobs,
            roll_size=roll_size,
            num_samples=num_samples,
        )

        return predictions[0] if called_with_single_series else predictions

    @random_method
    def predict_from_dataset(
        self,
        n: int,
        input_series_dataset: InferenceDataset,
        trainer: Optional[pl.Trainer] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = None,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
    ) -> Sequence[TimeSeries]:

        """
        This method allows for predicting with a specific :class:`darts.utils.data.InferenceDataset` instance.
        These datasets implement a PyTorch ``Dataset``, and specify how the target and covariates are sliced
        for inference. In most cases, you'll rather want to call :func:`predict()` instead, which will create an
        appropriate :class:`InferenceDataset` for you.

        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions
        input_series_dataset
            Optionally, a series or sequence of series, representing the history of the target series' whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        trainer
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction.  Using a custom ``trainer``
            will override Darts' default trainer.
        batch_size
            Size of batches during prediction. Defaults to the models ``batch_size`` value.
        verbose
            Optionally, whether to print progress.

            .. deprecated:: v0.17.0
                ``verbose`` has been deprecated in v0.17.0 and will be removed in a future version.
                Instead, control verbosity with PyTorch Lightning Trainer parameters ``enable_progress_bar``,
                ``progress_bar_refresh_rate`` and ``enable_model_summary`` in the ``pl_trainer_kwargs`` dict
                at model creation. See for example here:
                https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#enable-progress-bar
        n_jobs
            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.
        roll_size
            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set ``output_chunk_length`` by default.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        num_loader_workers
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            for the inference/prediction dataset loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.

        Returns
        -------
        Sequence[TimeSeries]
            Returns one or more forecasts for time series.
        """
        self._verify_inference_dataset_type(input_series_dataset)

        # check that covariates and dimensions are matching what we had during training
        self._verify_predict_sample(input_series_dataset[0])

        if roll_size is None:
            roll_size = self.output_chunk_length
        else:
            raise_if_not(
                0 < roll_size <= self.output_chunk_length,
                "`roll_size` must be an integer between 1 and `self.output_chunk_length`.",
            )

        # check that `num_samples` is a positive integer
        raise_if_not(num_samples > 0, "`num_samples` must be a positive integer.")

        # iterate through batches to produce predictions
        batch_size = batch_size or self.batch_size

        # set prediction parameters
        self.model.set_predict_parameters(
            n=n,
            num_samples=num_samples,
            roll_size=roll_size,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )

        pred_loader = DataLoader(
            input_series_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_fn,
        )

        if verbose is not None:
            raise_deprecation_warning(
                "kwarg `verbose` is deprecated and will be removed in a future Darts version. "
                "Instead, control verbosity with PyTorch Lightning Trainer parameters `enable_progress_bar`, "
                "`progress_bar_refresh_rate` and `enable_model_summary` in the `pl_trainer_kwargs` dict "
                "at model creation.",
                logger,
            )
        verbose = True if verbose is None else verbose

        # setup trainer. will only be re-instantiated if both `trainer` and `self.trainer` are `None`
        trainer = trainer if trainer is not None else self.trainer
        self._setup_trainer(trainer=trainer, verbose=verbose, epochs=self.n_epochs)

        # prediction output comes as nested list: list of predicted `TimeSeries` for each batch.
        predictions = self.trainer.predict(self.model, pred_loader)
        # flatten and return
        return [ts for batch in predictions for ts in batch]

    @property
    @abstractmethod
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        """Abstract property that returns model specific encoder settings that are used to initialize the encoders.

        Must return Tuple (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates)
        """
        pass

    def initialize_encoders(self) -> SequentialEncoder:
        """instantiates the SequentialEncoder object based on self._model_encoder_settings and parameter
        ``add_encoders`` used at model creation"""
        (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
        ) = self._model_encoder_settings

        return SequentialEncoder(
            add_encoders=self.add_encoders,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            takes_past_covariates=takes_past_covariates,
            takes_future_covariates=takes_future_covariates,
        )

    @property
    def first_prediction_index(self) -> int:
        """
        Returns the index of the first predicted within the output of self.model.
        """
        return 0

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series;
        overriding the default value of 3 of ForecastingModel
        """
        return self.input_chunk_length + self.output_chunk_length

    @staticmethod
    def _batch_collate_fn(batch: List[Tuple]) -> Tuple:
        """
        Returns a batch Tuple from a list of samples
        """
        aggregated = []
        first_sample = batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i]
            if isinstance(elem, np.ndarray):
                aggregated.append(
                    torch.from_numpy(np.stack([sample[i] for sample in batch], axis=0))
                )
            elif elem is None:
                aggregated.append(None)
            elif isinstance(elem, TimeSeries):
                aggregated.append([sample[i] for sample in batch])
        return tuple(aggregated)

    def save_model(self, path: str) -> None:
        """Saves the model under a given path. The path should end with '.pth.tar'

        Parameters
        ----------
        path
            Path under which to save the model at its current state.
        """

        raise_if_not(
            path.endswith(".pth.tar"),
            "The given path should end with '.pth.tar'.",
            logger,
        )

        with open(path, "wb") as f_out:
            torch.save(self, f_out)

    @staticmethod
    def load_model(path: str) -> "TorchForecastingModel":
        """loads a model from a given file path. The file name should end with '.pth.tar'

        Example for loading a :class:`RNNModel`:

            .. highlight:: python
            .. code-block:: python

                from darts.models import RNNModel

                model_loaded = RNNModel.load_model("my_model.pth.tar")
            ..

        Parameters
        ----------
        path
            Path under which to save the model at its current state. The path should end with '.pth.tar'
        """

        raise_if_not(
            path.endswith(".pth.tar"),
            "The given path should end with '.pth.tar'.",
            logger,
        )

        with open(path, "rb") as fin:
            model = torch.load(fin)
        return model

    @staticmethod
    def load_from_checkpoint(
        model_name: str, work_dir: str = None, file_name: str = None, best: bool = True
    ) -> "TorchForecastingModel":
        """
        Load the model from automatically saved checkpoints under '{work_dir}/darts_logs/{model_name}/checkpoints/'.
        This method is used for models that were created with ``save_checkpoints=True``.

        If you manually saved your model, consider using :meth:`load_model() <TorchForeCastingModel.load_model()>`.

        Example for loading a :class:`RNNModel` from checkpoint (``model_name`` is the ``model_name`` used at model
        creation):

            .. highlight:: python
            .. code-block:: python

                from darts.models import RNNModel

                model_loaded = RNNModel.load_from_checkpoint(model_name, best=True)
            ..

        If ``file_name`` is given, returns the model saved under
        '{work_dir}/darts_logs/{model_name}/checkpoints/{file_name}'.

        If ``file_name`` is not given, will try to restore the best checkpoint (if ``best`` is ``True``) or the most
        recent checkpoint (if ``best`` is ``False`` from '{work_dir}/darts_logs/{model_name}/checkpoints/'.

        Parameters
        ----------
        model_name
            The name of the model (used to retrieve the checkpoints folder's name).
        work_dir
            Working directory (containing the checkpoints folder). Defaults to current working directory.
        file_name
            The name of the checkpoint file. If not specified, use the most recent one.
        best
            If set, will retrieve the best model (according to validation loss) instead of the most recent one. Only
            is ignored when ``file_name`` is given.

        Returns
        -------
        TorchForecastingModel
            The corresponding trained :class:`TorchForecastingModel`.
        """

        if work_dir is None:
            work_dir = os.path.join(os.getcwd(), DEFAULT_DARTS_FOLDER)

        checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
        model_dir = _get_runs_folder(work_dir, model_name)

        # load base TorchForecastingModel saved at model creation
        base_model_path = os.path.join(model_dir, INIT_MODEL_NAME)
        raise_if_not(
            os.path.exists(base_model_path),
            f"Could not find base model save file `{INIT_MODEL_NAME}` in {model_dir}.",
            logger,
        )

        model = TorchForecastingModel.load_model(base_model_path)

        # load pytorch lightning module from checkpoint
        # if file_name is None, find most recent file in savepath that is a checkpoint
        if file_name is None:
            file_name = _get_checkpoint_fname(work_dir, model_name, best=best)

        file_path = os.path.join(checkpoint_dir, file_name)
        logger.info(f"loading {file_name}")

        model.model = model.model.__class__.load_from_checkpoint(file_path)
        model.load_ckpt_path = file_path
        return model

    @property
    def model_created(self) -> bool:
        return self.model is not None

    @property
    def epochs_trained(self) -> int:
        return self.model.epochs_trained if self.model_created else 0

    @property
    def likelihood(self) -> Likelihood:
        return (
            self.model.likelihood
            if self.model_created
            else self.pl_module_params.get("likelihood", None)
        )

    @property
    def input_chunk_length(self) -> int:
        return (
            self.model.input_chunk_length
            if self.model_created
            else self.pl_module_params["input_chunk_length"]
        )

    @property
    def output_chunk_length(self) -> int:
        return (
            self.model.output_chunk_length
            if self.model_created
            else self.pl_module_params["output_chunk_length"]
        )

    def _is_probabilistic(self) -> bool:
        return (
            self.model._is_probabilistic()
            if self.model_created
            else self.likelihood is not None
        )


def _raise_if_wrong_type(obj, exp_type, msg="expected type {}, got: {}"):
    raise_if_not(isinstance(obj, exp_type), msg.format(exp_type, type(obj)))


"""
Below we define the 5 torch model types:
    * PastCovariatesTorchModel
    * FutureCovariatesTorchModel
    * DualCovariatesTorchModel
    * MixedCovariatesTorchModel
    * SplitCovariatesTorchModel
"""
# TODO: there's a lot of repetition below... is there a cleaner way to do this in Python- Using eg generics or something


def _basic_compare_sample(train_sample: Tuple, predict_sample: Tuple):
    """
    For all models relying on one type of covariates only (Past, Future, Dual), we can rely on the fact
    that training/inference datasets have target and a covariate in first and second position to do the checks.
    """
    tgt_train, cov_train = train_sample[:2]
    tgt_pred, cov_pred = predict_sample[:2]
    raise_if_not(
        tgt_train.shape[-1] == tgt_pred.shape[-1],
        "The provided target has a dimension (width) that does not match the dimension "
        "of the target this model has been trained on.",
    )
    raise_if(
        cov_train is not None and cov_pred is None,
        "This model has been trained with covariates; some covariates of matching dimensionality are needed "
        "for prediction.",
    )
    raise_if(
        cov_train is None and cov_pred is not None,
        "This model has been trained without covariates. No covariates should be provided for prediction.",
    )
    raise_if(
        cov_train is not None
        and cov_pred is not None
        and cov_train.shape[-1] != cov_pred.shape[-1],
        "The provided covariates must have dimensionality matching that of the covariates used for training "
        "the model.",
    )


def _mixed_compare_sample(train_sample: Tuple, predict_sample: Tuple):
    """
    For models relying on MixedCovariates.

    Parameters:
    ----------
    train_sample
        (past_target, past_covariates, historic_future_covariates, future_covariates, future_target)
    predict_sample
        (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates, ts_target)
    """
    # datasets; we skip future_target for train and predict, and skip future_past_covariates for predict datasets
    ds_names = [
        "past_target",
        "past_covariates",
        "historic_future_covariates",
        "future_covariates",
    ]

    train_has_ds = [ds is not None for ds in train_sample[:-1]]
    predict_has_ds = [ds is not None for ds in predict_sample[:4]]

    train_datasets = train_sample[:-1]
    predict_datasets = predict_sample[:4]

    tgt_train, tgt_pred = train_datasets[0], predict_datasets[0]
    raise_if_not(
        tgt_train.shape[-1] == tgt_pred.shape[-1],
        "The provided target has a dimension (width) that does not match the dimension "
        "of the target this model has been trained on.",
    )

    for idx, (ds_in_train, ds_in_predict, ds_name) in enumerate(
        zip(train_has_ds, predict_has_ds, ds_names)
    ):
        raise_if(
            ds_in_train and not ds_in_predict and ds_in_train,
            f"This model has been trained with {ds_name}; some {ds_name} of matching dimensionality are needed "
            f"for prediction.",
        )
        raise_if(
            ds_in_train and not ds_in_predict and ds_in_predict,
            f"This model has been trained without {ds_name}; No {ds_name} should be provided for prediction.",
        )
        raise_if(
            ds_in_train
            and ds_in_predict
            and train_datasets[idx].shape[-1] != predict_datasets[idx].shape[-1],
            f"The provided {ds_name} must have dimensionality that of the {ds_name} used for training the model.",
        )


class PastCovariatesTorchModel(TorchForecastingModel, ABC):

    uses_future_covariates = False

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> PastCovariatesTrainingDataset:

        
        print("#=#=#=#=# INTERNAL #=#=#=#=#")
        print("Location: torch_forecasting_model.py -> _build_train_dataset(...)")
        print("Dataset is passed and looks like this: ")
        print(target)
        print("#=#=#=#=# INTERNAL #=#=#=#=#")
        raise_if_not(
            future_covariates is None,
            "Specified future_covariates for a PastCovariatesModel (only past_covariates are expected).",
        )

        return PastCovariatesSequentialDataset(
            target_series=target,
            covariates=past_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> PastCovariatesInferenceDataset:

        raise_if_not(
            future_covariates is None,
            "Specified future_covariates for a PastCovariatesModel (only past_covariates are expected).",
        )

        return PastCovariatesInferenceDataset(
            target_series=target,
            covariates=past_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, PastCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, PastCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        raise_if_not(
            future_covariates is None,
            "Some future_covariates have been provided to a PastCovariates model. These models "
            "support only past_covariates.",
        )

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = False
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
        )


class FutureCovariatesTorchModel(TorchForecastingModel, ABC):

    uses_past_covariates = False

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> FutureCovariatesTrainingDataset:
        raise_if_not(
            past_covariates is None,
            "Specified past_covariates for a FutureCovariatesModel (only future_covariates are expected).",
        )

        return FutureCovariatesSequentialDataset(
            target_series=target,
            covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> FutureCovariatesInferenceDataset:
        raise_if_not(
            past_covariates is None,
            "Specified past_covariates for a FutureCovariatesModel (only future_covariates are expected).",
        )

        return FutureCovariatesInferenceDataset(
            target_series=target,
            covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, FutureCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, FutureCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        raise_if_not(
            past_covariates is None,
            "Some past_covariates have been provided to a PastCovariates model. These models "
            "support only future_covariates.",
        )

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = False
        takes_future_covariates = True
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
        )


class DualCovariatesTorchModel(TorchForecastingModel, ABC):

    uses_past_covariates = False

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> DualCovariatesTrainingDataset:

        return DualCovariatesSequentialDataset(
            target_series=target,
            covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> DualCovariatesInferenceDataset:

        return DualCovariatesInferenceDataset(
            target_series=target,
            covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, DualCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, DualCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        _basic_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        raise_if_not(
            past_covariates is None,
            "Some past_covariates have been provided to a PastCovariates model. These models "
            "support only future_covariates.",
        )

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = False
        takes_future_covariates = True
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
        )


class MixedCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> MixedCovariatesTrainingDataset:

        return MixedCovariatesSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> MixedCovariatesInferenceDataset:

        return MixedCovariatesInferenceDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, MixedCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, MixedCovariatesInferenceDataset)

    def _verify_predict_sample(self, predict_sample: Tuple):
        _mixed_compare_sample(self.train_sample, predict_sample)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        # both covariates are supported; do nothing
        pass

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = True
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
        )


class SplitCovariatesTorchModel(TorchForecastingModel, ABC):
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> SplitCovariatesTrainingDataset:

        return SplitCovariatesSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> SplitCovariatesInferenceDataset:

        return SplitCovariatesInferenceDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        _raise_if_wrong_type(train_dataset, SplitCovariatesTrainingDataset)

    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        _raise_if_wrong_type(inference_dataset, SplitCovariatesInferenceDataset)

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        # both covariates are supported; do nothing
        pass

    def _verify_predict_sample(self, predict_sample: Tuple):
        # TODO: we have to check both past and future covariates
        raise NotImplementedError()

    @property
    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        takes_past_covariates = True
        takes_future_covariates = True
        return (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
        )
