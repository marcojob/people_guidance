from people_guidance.pipeline import Pipeline
from people_guidance.utils import init_logging

# Only example, remove this for own implementation
from examples.modules.spam_module import SpamModule
from examples.modules.echo_module import EchoModule

if __name__ == '__main__':
    init_logging()

    pipeline = Pipeline()
    # Only example, remove this for own implementation
    pipeline.add_module(SpamModule)
    pipeline.add_module(EchoModule)

    pipeline.start()
