# ETH Slam Datasets
## Usage
First you have to download and convert the desired dataset (specified by dataset_name in convert_eth3d_slam_datasets.py). 
**Downloading is handled automatically.** Simply run:
```shell
python convert_eth3d_slam_datasets.py
```
Running this will create a new folder called "converted_eth_slam_{dataset_name}", which contains the dataset in our format.
After switching the pipeline over to the new dataset, you can create a "depth genius" Object in your module. 
You can query this class for ground truth information at a number of points by using:
```python
dg = DepthGenius(dataset_name, converted_eth_slam_{dataset_name})
points = np.array(((12, 13), (200, 200), (100, 13)))
true_depth = dg(points, 11873357) #  the second argument specifies the timestamp
print(true_depth)
```
