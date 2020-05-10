from argparse import ArgumentParser
import logging
import platform

from people_guidance.pipeline import Pipeline
from people_guidance.utils import init_logging

# Need to fix this properly
is_rpi: bool = platform.uname().machine == 'armv7l'

from people_guidance.modules.drivers_module import DriversModule

if not is_rpi:
    from people_guidance.modules.feature_tracking_module import FeatureTrackingModule
    from people_guidance.modules.visualization_module import VisualizationModule
    from people_guidance.modules.reprojection_module import ReprojectionModule
    from people_guidance.modules.visual_odometry_module import VisualOdometryModule


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

    parser.add_argument('--visualize', '-v',
                        help='Turn on visualisation',
                        action='store_true')

    parser.add_argument('--save_visualization','-s',
                        help='Save visualization to file',
                        type=str,
                        default='')

    args = parser.parse_args()

    pipeline = Pipeline(args, log_level=logging.WARNING)

    # Handles hardware drivers and interfaces
    pipeline.add_module(DriversModule, log_level=logging.WARNING)

    if not args.record:
        # Handles visual odometry
        pipeline.add_module(VisualOdometryModule, log_level=logging.DEBUG)

        # Reprojection module
        pipeline.add_module(ReprojectionModule, log_level=logging.DEBUG)

        # If argument is specified we start visualization
        if args.visualize:
            pipeline.add_module(VisualizationModule, log_level=logging.WARNING)

    pipeline.start()
