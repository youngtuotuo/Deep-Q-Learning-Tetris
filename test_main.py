import yaml
import sys

# TODO test todo


def test_config(config="config.yaml"):
    with open(config, "r") as f:
        params = yaml.safe_load(f)
    assert (
        params["play"]["width"] % params["tile_size"] == 0
    ), "Tetris configuration not feasible!"
    assert (
        params["play"]["height"] % params["tile_size"] == 0
    ), "Tetris configuration not feasible!"
