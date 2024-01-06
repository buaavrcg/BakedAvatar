import torch

class FrequencyEmbedder():
    """Embed input coordinates to network input vector."""

    def __init__(self,
                 num_freqs,
                 max_freq_log2=None,
                 input_dims=3,
                 include_input=True,
                 log_sampling=True,
                 periodic_fns=[torch.sin, torch.cos]) -> None:

        self.input_dims = input_dims
        self.embed_fns = []
        if include_input:
            self.embed_fns += [lambda x: x]

        max_freq_log2 = max_freq_log2 or num_freqs - 1
        if log_sampling:
            freq_bands = 2**torch.linspace(0, max_freq_log2, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2**0, 2**max_freq_log2, steps=num_freqs)

        for freq in freq_bands:
            for p_fn in periodic_fns:
                fn = lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)
                self.embed_fns.append(fn)

    @property
    def dim_out(self):
        return len(self.embed_fns) * self.input_dims

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)