import sys
import os

# Add the current directory to sys.path to allow importing modelkits
sys.path.append(os.getcwd())

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
try:
    import modelkits.models
    print("Successfully imported modelkits.models")
except ImportError as e:
    print(f"Failed to import modelkits.models: {e}")
    sys.exit(1)

# Check if the config is registered
try:
    config_class = AutoConfig.for_model("qwen3_moe")
    print(f"AutoConfig for 'qwen3_moe' resolved to: {config_class}")
except Exception as e:
    print(f"Failed to resolve AutoConfig for 'qwen3_moe': {e}")

# Check if the model class is registered in AutoModel
# Note: AutoModel.from_config() is a good way to test without weights
try:
    config = AutoConfig.for_model("qwen3_moe")
    # Instantiate config
    conf = config()
    # Attempt to create model from config (this validates registration)
    # We won't actually initialize the full model to save time/memory if possible, 
    # but AutoModel.from_config will look up the class.
    
    # Actually, let's just check the registry directly if possible, or try to instantiate.
    # Accessing _model_mapping is private but good for verification.
    
    # Or just try to instantiate a dummy model
    print("Attempting to instantiate model from config...")
    # This might fail if dependencies are missing, but let's try.
    model = AutoModel.from_config(conf)
    print(f"Successfully instantiated {type(model)}")
    
    model_causal = AutoModelForCausalLM.from_config(conf)
    print(f"Successfully instantiated {type(model_causal)}")
    
except Exception as e:
    print(f"Error during model instantiation verification: {e}")
