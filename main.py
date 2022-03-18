import yaml
import sys


def main(config):
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    print(params)


if __name__ == '__main__':
    main(sys.argv[1])