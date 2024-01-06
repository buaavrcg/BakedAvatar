import sys
import math
import torch
import random
import numpy as np
import torch.nn.init as init
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from .misc_util import EasyDict


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def add_dict_to(total_dict, dict_to_add):
    for k, v in dict_to_add.items():
        if k in total_dict:
            total_dict[k] += v
        else:
            total_dict[k] = v


def log_value_dict(tb_logger, tag, value_dict, it):
    for name, value in value_dict.items():
        tb_logger.add_scalar(f'{tag}/{name}', value, it)


def assert_shape(tensor, shape):
    assert tensor.ndim == len(shape), \
        f"Wrong number of dimensions: got {tensor.ndim}, expected {len(shape)}"
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            assert torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}'
        elif isinstance(size, torch.Tensor):
            assert torch.equal(size, torch.as_tensor(ref_size)), \
                f'Wrong size for dimension {idx}: expected {ref_size}'
        else:
            assert size == ref_size, f'Wrong size for dimension {idx}: got {size}, expected {ref_size}'


def weights_init(init_type):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -1.0, 1.0)
            elif init_type == 'default':
                pass
            else:
                assert 0, f"Unsupported initialization: {init_type}"

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True, out=sys.stdout):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    if isinstance(outputs, dict):
        outputs = list(outputs.values())
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [
            e for e in entries
            if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)
        ]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print(file=out)
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)),
              file=out)
    print(file=out)
    return outputs


def chunked_forward(fn, max_batch_size: int):
    """
    Make a chunked version of the function that operates on minibatches.
    Args:
        fn: A callable of torch.nn.Module to perform forward pass.
        max_batch_size: The maximum batch size to use for each forward pass.
    Returns:
        warpped_fn: A callable that performs forward pass in chunks.
    """
    def wrapped_fn(*inputs: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        chunk_inputs = [torch.split(x, max_batch_size, dim=0) for x in inputs]

        for chunk_input in zip(*chunk_inputs):
            chunk_output = fn(*chunk_input)
            if not isinstance(chunk_output, (tuple, list)):
                chunk_output = (chunk_output, )
            outputs.append(chunk_output)

        return [torch.cat(x, dim=0) for x in zip(*outputs)]

    return wrapped_fn


class InfiniteSampler(Sampler):
    def __init__(self,
                 data_source: Dataset,
                 rank=0,
                 num_replicas=1,
                 shuffle=True,
                 seed=0,
                 window_size=0.5):
        super().__init__(data_source)
        assert len(data_source) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        self.dataset = data_source
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        try:
            while True:
                i = idx % order.size
                if idx % self.num_replicas == self.rank:
                    yield order[i]
                if window >= 2:
                    j = (i - rnd.randint(window)) % order.size
                    order[i], order[j] = order[j], order[i]
                idx += 1
        except GeneratorExit:
            pass


def fake_quant(x: torch.Tensor, scale=128, zero_point=0, num_bits=8, signed=True):
    """Fake quantization while keep float gradient."""
    x_quant = (x.detach() * scale + zero_point).round().int()
    if num_bits is not None:
        if signed:
            qmin = -(2**(num_bits - 1))
            qmax = 2**(num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**num_bits - 1
        x_quant = torch.clamp(x_quant, qmin, qmax)
    x_dequant = (x_quant - zero_point).float() / scale
    x = x - x.detach() + x_dequant  # stop gradient
    return x
