import yaml
import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--common', type=str, help="Path to common config YAML file", required=True)
    parser.add_argument('--config', type=str, help="Path to config YAML file", required=True)
    args, override_args = parser.parse_known_args()

    with open(args.common, 'r') as f0:
        common = yaml.safe_load(f0)

    # Load base config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # override and insert any from common with config:
    for k,v in config.items():
        common[k] = v           # I should keep configurations as un-nested as possible to avoid having to do a recursive replacement

    # Override YAML with CLI arguments
    for i in range(0, len(override_args), 2):
        key = override_args[i].lstrip('--')
        val = override_args[i+1]
        try:
            # Try to evaluate numbers, bools, dicts, etc.
            val = eval(val)
        except:
            pass
        common[key] = val

    return common   # extended and overwritten
