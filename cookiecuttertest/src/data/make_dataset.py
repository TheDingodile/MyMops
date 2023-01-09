# -*- coding: utf-8 -*-
import logging
from os import listdir
from pathlib import Path
from mnist import mnist

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
    (inputs, labels), (test_in, test_out) = mnist(input_filepath)
    torch.save(inputs, output_filepath + "/train" + "/inputs")
    torch.save(labels, output_filepath + "/train" +"/labels")
    torch.save(test_in, output_filepath + "/test" +"/inputs")
    torch.save(test_out, output_filepath + "/test" +"/labels")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
