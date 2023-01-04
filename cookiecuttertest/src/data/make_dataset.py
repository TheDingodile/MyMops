# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from os import listdir


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

    test_in = np.empty(shape=(0, 28, 28))
    test_out = np.empty(shape=(0))
    for f in files:
        if f[0:5] == "train":
            train_in = np.concatenate((train_in, np.load("data/corruptmnist/" + f)['images']), axis=0)
            train_out = np.concatenate((train_out, np.load("data/corruptmnist/" + f)['labels']), axis=0)
        else:
            test_in = np.concatenate((test_in, np.load("data/corruptmnist/" + f)['images']), axis=0)
            test_out = np.concatenate((test_out, np.load("data/corruptmnist/" + f)['labels']), axis=0)
    return (train_in, train_out), (test_in, test_out)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
