import sys
import yaml
from yaml import SafeLoader
from pathlib import Path

def get_root_config(file, Constants):
    # get the desired parent directory as root path
    ROOT = Path(file).resolve().parents[1]
    # add ROOT to sys.path if not present
    if str(ROOT) not in sys.path:
        # add ROOT to sys.path
        sys.path.append(str(ROOT))
    # load the config and parse it into a dictionary
    with open(ROOT / Constants.CONFIG_PATH.value) as f:
        slice_config = yaml.load(f, Loader = SafeLoader)
    return ROOT, slice_config