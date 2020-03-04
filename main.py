from people_guidance.pipeline import Pipeline
from people_guidance.utils import init_logging

from people_guidance.modules.drivers_module import DriversModule

if __name__ == '__main__':
    init_logging()

    pipeline = Pipeline()

    # Handles hardware drivers and interfaces
    pipeline.add_module(DriversModule)

    pipeline.start()
