import os
import yaml
from argparse import ArgumentParser

def find_valid_keys(base, input):
    log={}
    for key, value in base.items():
        if key not in input.keys():
            log[key] = value
        elif isinstance(value, dict):
            if isinstance(input[key], dict):
                tmp = find_valid_keys(value, input[key])
                if tmp != {}:
                    log[key] = tmp
            else:
                log[key] = value
    return log

def add_valid_keys(base, input):
    log={}
    for key, value in base.items():
        if key not in input.keys():
            log[key] = value
            input[key] = value
        elif isinstance(value, dict):
            if isinstance(input[key], dict):
                input[key], tmp = add_valid_keys(value, input[key])
                if tmp != {}:
                    log[key] = tmp
            else:
                log[key] = value
                input[key] = value
    return input, log

def find_invalid_keys(base, input):
    log={}
    for key, value in input.items():
        if key not in base.keys():
            log[key] = value
        elif isinstance(value, dict):
            if isinstance(base[key], dict):
                tmp = find_invalid_keys(base[key], value)
                if tmp != {}:
                    log[key] = tmp
            else:
                log[key] = value
    return log

def remove_invalid_keys(base, input):
    log={}
    for key in list(input.keys()):
        if key not in base.keys():
            log[key] = input[key]
            del input[key]
        elif isinstance(input[key], dict):
            if isinstance(base[key], dict):
                input[key], tmp = remove_invalid_keys(base[key], input[key])
                if tmp != {}:
                    log[key] = tmp
            else:
                log[key] = input[key]
                del input[key]
    return input, log
    

def get_nested_keys(input_dict, join_str="> "):
    key_str=""
    for key, value in input_dict.items():
        if isinstance(value, dict):
            key_str += join_str + f"{key}:\n"
            key_str += get_nested_keys(input_dict[key], join_str=join_str+"\t")
        else:
            key_str += join_str + f"{key}\n"
    return key_str
        

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default=None)
    parser.add_argument('-b', '--base', default="./config.yaml")
    parser.add_argument('-a', '--add-valid-keys', action="store_true")
    parser.add_argument('-r', '--remove-invalid-keys', action="store_true")
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    assert args.input is not None and os.path.exists(args.input), (
        "Must set --input set to a valid path.",
    )
    assert args.input[-5:] == ".yaml", (
        "Must set --input to a path to some .yaml file.",
    )

    assert os.path.exists(args.base), (
        "Must set --base set to a valid path.",
    )
    assert args.base[-5:] == ".yaml", (
        "Must set --base to a path to some .yaml file.",
    )
    
    with open(args.input) as stream:
        try:
            input_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    with open(args.base) as stream:
        try:
            base_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    assert isinstance(input_cfg, dict), (
        "Must set --input to some .yaml encoding a dict.",
    )
    assert isinstance(base_cfg, dict), (
        "Must set --base to some .yaml encoding a dict.",
    )

    if args.remove_invalid_keys:
        input_cfg, log_removed = remove_invalid_keys(base_cfg, input_cfg)
        if args.verbose:
            keys_removed = get_nested_keys(log_removed)
            if keys_removed == "":
                print(f"No invalid keys were found in {args.input}")
            else:
                print(
                    "Removed invalid keys from ",
                    f"{args.input} as follows:\n{keys_removed}",
                )
    else:
        log_invalid = find_invalid_keys(base_cfg, input_cfg)
        if args.verbose:
            keys_invalid = get_nested_keys(log_invalid)
            if keys_invalid == "":
                print(f"No invalid keys were found in {args.input}")
            else:
                print(
                    "Discovered invalid keys in ",
                    f"{args.input} as follows:\n{keys_invalid}",
                )

    if args.add_valid_keys:
        input_cfg, log_added = add_valid_keys(base_cfg, input_cfg)
        if args.verbose:
            keys_added = get_nested_keys(log_added)
            if keys_added == "":
                print(f"No valid keys are missing in {args.input}")
            else:
                print(
                    "Added missing valid keys to ",
                    f"{args.input} as follows:\n{keys_added}",
                )    
    else:
        log_valid = find_valid_keys(base_cfg, input_cfg)
        if args.verbose:
            keys_valid = get_nested_keys(log_valid)
            if keys_valid == "":
                print(f"No valid keys are missing in {args.input}")
            else:
                print(
                    "Discovered missing valid keys in ",
                    f"{args.input} as follows:\n{keys_valid}",
                )
    
    if args.add_valid_keys or args.remove_invalid_keys:
        with open(args.input, 'w') as stream:
            try:
                yaml.safe_dump(input_cfg, stream)
            except yaml.YAMLError as exc:
                print(exc)

    print("Completed config sync.")
    

if __name__ == "__main__":
    main()
