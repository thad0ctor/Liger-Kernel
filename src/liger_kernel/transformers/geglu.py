import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import logging

from liger_kernel.ops import LigerGELUMulFunction

logger = logging.get_logger(__name__)


class LigerGEGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # TODO: support exact GELU
        # Right now Gemma 1, 1.1 and 2 models are all using `gelu_pytorch_tanh`
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/gemma/modeling_gemma.py#L175
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/activations.py#L46
        # So we can safely assume we use tanh approximation form all the time

    def forward(self, x):
        return self.down_proj(LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


class LigerGEGLUMLPForGemma4(LigerGEGLUMLP):
    """GEGLU MLP drop-in for HF's ``Gemma4TextMLP``.

    HF's ``Gemma4TextMLP`` is instantiated as ``Gemma4TextMLP(config, layer_idx)``
    and sizes ``intermediate_size`` per-layer: KV-shared layers (layers at
    ``i >= num_hidden_layers - num_kv_shared_layers``) use ``2*intermediate_size``
    when ``use_double_wide_mlp=True`` (E2B-it); all other layers and all 31B
    layers use ``intermediate_size`` unchanged. Forward is inherited from
    ``LigerGEGLUMLP``.

    Internal monkey-patch helper only — not part of the public API surface.
    """

    def __init__(self, config, layer_idx=None):
        # Do not chain through LigerGEGLUMLP.__init__: it would create linears
        # at the base intermediate_size, missing the double-wide KV-shared path.
        nn.Module.__init__(self)

        # Mirrors transformers.models.gemma4.modular_gemma4.Gemma4TextMLP sizing.
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
        is_kv_shared_layer = layer_idx is not None and layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class LigerGEGLUMLPForGemma4Vision(nn.Module):
    """GEGLU MLP drop-in for HF's ``Gemma4VisionMLP``.

    Gemma4VisionMLP is built on ``Gemma4ClippableLinear`` wrappers whose actual
    ``nn.Linear`` lives at ``self.linear``. To preserve state_dict keys
    (``gate_proj.linear.weight`` etc.) we instantiate the same
    ``Gemma4ClippableLinear`` objects rather than plain ``nn.Linear``.

    Fast path (``use_clipped_linears=False``) bypasses the clip wrappers and
    fuses gate * up + GELU via Liger's ``LigerGELUMulFunction`` (matches
    ``gelu_pytorch_tanh`` as used elsewhere in Gemma). When
    ``use_clipped_linears=True`` the class-level kernel swap would silently
    drop the clip semantics, so we fall back to HF's reference formula via
    the wrapped ``Gemma4ClippableLinear`` objects and emit a one-time warning.

    Internal monkey-patch helper only — not part of the public API surface.
    """

    def __init__(self, config, layer_idx=None):
        # Deliberately do NOT chain through ``LigerGEGLUMLP.__init__``: that
        # parent creates plain ``nn.Linear`` modules, which would break the
        # ``gate_proj.linear.weight`` state_dict keys that Gemma 4 vision
        # checkpoints carry.
        nn.Module.__init__(self)

        # Lazy import keeps geglu.py usable when HF transformers is absent
        # or when the user's transformers version predates gemma4.
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Gemma4ClippableLinear(config, self.hidden_size, self.intermediate_size)
        self.up_proj = Gemma4ClippableLinear(config, self.hidden_size, self.intermediate_size)
        self.down_proj = Gemma4ClippableLinear(config, self.intermediate_size, self.hidden_size)

    def forward(self, x):
        if not self.gate_proj.use_clipped_linears:
            # Fast path: bypass the (no-op) clip wrappers and feed the
            # underlying nn.Linear directly into the fused GELU-mul kernel.
            gate = self.gate_proj.linear(x)
            up = self.up_proj.linear(x)
            return self.down_proj.linear(LigerGELUMulFunction.apply(gate, up))

        # Clipped path: the input/output clamps inside Gemma4ClippableLinear
        # are load-bearing for quantized checkpoints, so we must go through
        # the full wrapper.forward(). Use HF's reference GELU-mul formula
        # (gelu_pytorch_tanh) to match HF exactly.
        logger.warning_once(
            "LigerGEGLUMLPForGemma4Vision: use_clipped_linears=True detected; "
            "falling back to HF reference GELU path (Liger GEGLU kernel is "
            "bypassed for this MLP so input/output clamps are preserved)."
        )
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
