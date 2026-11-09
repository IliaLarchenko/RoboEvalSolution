"""Select and install the gemma attention backend. Call before the model is constructed: the patch is picked up
at construction time.

cuDNN fused attention for the long prefix pass, reference math for the cached suffix passes
(cudnn_attention_patch.py); self-tested forward + backward, else falls back to jax.nn.dot_product_attention (XLA)
for every pass (flash_attention_patch.py).
"""

import logging

logger = logging.getLogger(__name__)


def install() -> str:
    """Install the attention patch. Returns the backend name."""
    from roboeval_solution.training import cudnn_attention_patch
    if cudnn_attention_patch.install(training=True):
        return "cudnn_prefix"
    logger.warning("cuDNN prefix attention unavailable, falling back")
    from roboeval_solution.training import flash_attention_patch
    flash_attention_patch.install()
    return "flash"
