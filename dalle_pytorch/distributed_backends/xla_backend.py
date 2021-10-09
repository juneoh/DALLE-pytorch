"""Defines XLABackend for PyTorch XLA."""
import torch

from .distributed_backend import DistributedBackend


class XLABackend(DistributedBackend):
    """Distributed backend using PyTorch XLA."""

    BACKEND_MODULE_NAME = 'torch_xla.core.xla_model'
    BACKEND_NAME = 'XLA'

    def wrap_arg_parser(self, parser):
        if self.has_backend():
            parser.add_argument(
                '--tpu_cores', type=int, default=0, choices=[0, 1, 8],
                help='The number of TPU cores to use.')
        return parser

    def check_batch_size(self, batch_size):
        pass

    def _initialize(self):
        pass

    def _get_world_size(self):
        return self.backend_module.xrt_world_size()

    def _get_rank(self):
        return self.backend_module.get_ordinal()

    def _get_local_rank(self):
        return self.backend_module.get_local_ordinal()

    def _local_barrier(self):
        pass

    def _distribute(
            self,
            _args=None,
            model=None,
            optimizer=None,
            _model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            **_kwargs,
    ):
        device = self.backend_module.xla_device()

        model = model.to(device)

        import torch_xla.distributed.parallel_loader as pl
        training_data = pl.MpDeviceLoader(training_data, device)

        return (model, optimizer, training_data, lr_scheduler)

    def _average_all(self, tensor):
        return self.backend_module.all_reduce(
            self.backend_module.REDUCE_SUM,
            tensor,
            scale=1/self.backend_module.xrt_world_size()
        )
