# people_guidance

## Adding Modules
To add a module simply create a folder in the people_guidance/modules directory. Your Module must be a class that inherits from the Module class found in people_guidance/modules/module.py. The constructor for your class *must* take just one argument: log_dir which will be passed to it by the Pipeline which creates the model. The Module Class (which your Module Class must inherit from) sets up some basic things like the logger and the input/output queues through which your module can send/receive data.

**To run your Model you must add it to the Pipeline. You can do so by importing your class in main.py and simply adding:**
```python
from xxx import MyModule
pipeline.add_module(MyModule)
```

### Example Module
```python
class ExampleModule(Module):

    def __init__(self, log_dir: pathlib.Path):
        super(ExampleModule, self).__init__(name="example_module", outputs=[("spam", 10)], 
        input_topics=["echo_module:echo"], log_dir=log_dir)
        """
        this will create a model that can get data from the "echo_module:echo" (which is the "echo" output from the "echo_module"
        output) and publishes data on the "example_module:spam" channel. The channel size is limited by the integer (10) after 
        the output name, to avoid buffer overflows. This number should be fairly small for streams of 
        large objects (i.e. a pointcloud) and can be larger for small objects (i.e. a single float). All Queues are FiFo.
        
        The init function should be as lightweight as possible as these are run sequentially for all models in the main process. The                   
        self.start method is called in an extra process for each module. Costly initializations should therefore be made in self.start and not in 
        self.init.
        """
    def start(self):
        while True:
            time.sleep(1)
            self.logger.info("Spamming...")
            spam = np.random.random((20, 20, 3))
            # put the newly generated numpy array into the output queue with name "spam".
            self.outputs["spam"].put(spam)
            # get the oldest numpy array from the input queue with name "echo_module:echo"
            data = self.get("echo_module:echo") 
            self.logger.info(f"Received Echo with shape {data.shape} ")
            
   def cleanup(self):
       # any cleanup code (i.e. closing serial connections etc.) should be put here. This function is called even if an exception occurrs.
       pass
       
```

## Logging
Every time you run your pipeline a new log directory is created with todays time and date in the logs folder. There are logfiles for each module and the pipeline itself. All Modules have a self.logger member which can be used to log to the console and to a file. 
