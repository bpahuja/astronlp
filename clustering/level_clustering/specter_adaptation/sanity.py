import adapters
from adapters import AdapterConfig
from transformers import AutoModel

m = AutoModel.from_pretrained("allenai/specter2_base")
adapters.init(m)
cfg = AdapterConfig.load("pfeiffer", reduction_factor=16)
m.add_adapter("probe", config=cfg); m.train_adapter("probe"); m.set_active_adapters("probe")
print("OK: adapter added & active")
