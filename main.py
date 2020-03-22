from argparse import ArgumentParser

from people_guidance.pipeline import Pipeline
from people_guidance.utils import init_logging

from people_guidance.modules.drivers_module import DriversModule

if __name__ == '__main__':
    init_logging()

    # Arguments for different hardware configuration cases
    parser = ArgumentParser()
    parser.add_argument('--record', '-c',
                        help='Path of folder where to record dataset to',
                        type=str,
                        default='')
    parser.add_argument('--replay', '-p',
                        help='Path of folder where to replay dataset from',
                        type=str,
                        default='')
    args = parser.parse_args()

    pipeline = Pipeline(args)

    # Handles hardware drivers and interfaces
    pipeline.add_module(DriversModule)

    pipeline.start()
