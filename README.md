# A Python Ray-Mesh Intersector in Slangpy 

this is a GPU LBVH builder implemented in [slang-torch](https://github.com/shader-slang/slang-torch), based on [Vulkan LBVH (Linear Bounding Volume Hierarchy)](https://github.com/MircoWerner/VkLBVH)

I build every step in slangpy, so you can get any buffers in the BVH building process, for example, you can get the information of the bvh trees, so you can query on them in your own kernel, or do anything to them.
## Installation

```
pip install torch, pyexr, slangpy, time, csv, numpy
```

## Usage

run ```test.py```, which will let you know every step of building it.
and you will get a position map of a bunny.
![img](imgs/bunny.png)

