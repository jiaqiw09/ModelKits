from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
)

from .qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from .qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeModel,
    Qwen3MoeForCausalLM,
    Qwen3MoeForSequenceClassification,
    Qwen3MoeForTokenClassification,
    Qwen3MoeForQuestionAnswering,
)

def register_models():
    """
    Register Qwen3MoE models with HuggingFace Auto classes.
    """
    # Register the config
    # Note: "qwen3_moe" must match the model_type in the config
    AutoConfig.register("qwen3_moe", Qwen3MoeConfig)

    # Register the models
    AutoModel.register(Qwen3MoeConfig, Qwen3MoeModel)
    AutoModelForCausalLM.register(Qwen3MoeConfig, Qwen3MoeForCausalLM)
    AutoModelForSequenceClassification.register(Qwen3MoeConfig, Qwen3MoeForSequenceClassification)
    AutoModelForTokenClassification.register(Qwen3MoeConfig, Qwen3MoeForTokenClassification)
    AutoModelForQuestionAnswering.register(Qwen3MoeConfig, Qwen3MoeForQuestionAnswering)

# Execute registration on import
register_models()
