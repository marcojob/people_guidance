# ETH Slam Datasets
## Usage
First you have to download and convert the desired dataset (specified by dataset_name in data.py). 
**Downloading is handled automatically.** Simply run from the people_guidance folder:
```shell
python data.py 
```
Running this will create a new folder called "data/converted_eth_slam_{dataset_name}", which contains the dataset in our format.
The intrinsic parameters of the camera will be loaded automatically.

After switching the pipeline over to the new dataset, you can create a "depth genius" Object in your module. 
You can query this class for ground truth information at a number of points by using:
```python
from people_guidance.tools.depth_genius import DepthGenius
dg = DepthGenius(dataset_name, converted_eth_slam_{dataset_name})
points = np.array(((12, 13), (200, 200), (100, 13)))
true_depth = dg(points, 11873357) #  the second argument specifies the timestamp
print(true_depth)
```
