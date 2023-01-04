# -*- coding: utf-8 -*-
import logging
from os import listdir
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    files = listdir("cookiecuttertest/data/raw")
    train_in = np.empty(shape=(0, 28, 28))
    train_out = np.empty(shape=(0))
    for f in files:
        inputs = np.concatenate((train_in, np.load("data/raw/" + f)['images']), axis=0)
        labels = np.concatenate((train_out, np.load("data/raw/" + f)['labels']), axis=0)
    torch.save(inputs, "data/processed/inputs")
    torch.save(labels, "data/processed/labels")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
