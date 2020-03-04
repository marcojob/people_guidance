# people_guidance

## Adding Modules
To add a module simply create a folder in the people_guidance/modules directory. Your Module must be a class that inherits from the Module class found in people_guidance/modules/module.py. The constructor for your class *must* take three arguments: 
- input_queue
- output_queue 
- log_dir

All of these arguments are passed to your Module by the pipeline class. The Module class needs these arguments to set up some basic utilities for your module such as logging and your input and output queues, through which you will recieve data from modules that come before or after you in the pipeline.