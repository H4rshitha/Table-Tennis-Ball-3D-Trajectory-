import os

base_cache = os.path.expanduser("~/.cache/torch/hub/checkpoints/tt_uplifting_extracted/weights")
workspace_root = os.path.dirname(os.path.abspath(__file__))

logs_path = os.path.join(workspace_root, 'logs')
data_path = os.path.join(workspace_root, 'data')
weights_path = base_cache if os.path.exists(base_cache) else os.path.join(workspace_root, 'weights')