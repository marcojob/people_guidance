from argparse import ArgumentParser

from people_guidance.pipeline import Pipeline
from people_guidance.utils import init_logging

from people_guidance.modules.drivers_module import DriversModule
from people_guidance.modules.fps_logger_module import FPSLoggerModule
from people_guidance.modules.visualization_module import VisualizationModule

from examples.modules.echo_module import EchoModule
from examples.modules.spam_module import SpamModule

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
    parser.add_argument('--deploy', '-d',
                        help='Deploy the pipeline on a raspberry pi.',
                        action='store_true')
    args = parser.parse_args()

    pipeline = Pipeline(args)

    # Handles hardware drivers and interfaces
    pipeline.add_module(EchoModule)
    pipeline.add_module(SpamModule)
    pipeline.start()
