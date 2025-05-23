from dataset.dataset_ESC50 import get_global_stats
import config
stats = get_global_stats(config.esc50_path)
print(stats)