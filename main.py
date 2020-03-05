from people_guidance.modules.spam_module import SpamModule
from people_guidance.modules.echo_module import EchoModule
from people_guidance.pipeline import Pipeline
from people_guidance.utils import init_logging

if __name__ == '__main__':
    init_logging()

    pipeline = Pipeline()
    pipeline.add_module(SpamModule)
    pipeline.add_module(EchoModule)
    pipeline.start()
