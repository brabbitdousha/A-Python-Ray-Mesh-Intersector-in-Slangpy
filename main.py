import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import trimesh
import pyexr
import slangpy
import time
import csv
import numpy as np


m_gen_ele = slangpy.loadModule('bvhworkers/get_elements.slang')
m_morton_codes = slangpy.loadModule('bvhworkers/lbvh_morton_codes.slang')
m_radixsort = slangpy.loadModule('bvhworkers/lbvh_single_radixsort.slang')
m_hierarchy = slangpy.loadModule('bvhworkers/lbvh_hierarchy.slang')
m_bounding_box = slangpy.loadModule('bvhworkers/lbvh_bounding_boxes.slang')
m_intersect_test = slangpy.loadModule('bvhworkers/intersect_test.slang')

#debug
#'''
input = torch.tensor((0.6,0.7,0.8), dtype=torch.float).cuda()
output = torch.zeros(input.shape, dtype=torch.int).cuda()
m_gen_ele.debug_cb(a=input, b=output)\
.launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1))
print(output)
#'''

mesh = trimesh.load('./models/bunny.obj')

vrt = torch.from_numpy(mesh.vertices).cuda().float()
v_ind = torch.from_numpy(mesh.faces).cuda().int()

start_time = time.time()
#first part, get element and bbox---------------
primitive_num = v_ind.shape[0]
ele_primitiveIdx = torch.zeros((primitive_num, 1), dtype=torch.int).cuda()
ele_aabb = torch.zeros((primitive_num, 6), dtype=torch.float).cuda()

# Invoke normally
m_gen_ele.generateElements(vert=vrt, v_indx=v_ind, ele_primitiveIdx=ele_primitiveIdx, ele_aabb=ele_aabb)\
    .launchRaw(blockSize=(256, 1, 1), gridSize=((primitive_num+255)//256, 1, 1))
extent_min_x = ele_aabb[:,0].min()
extent_min_y = ele_aabb[:,1].min()
extent_min_z = ele_aabb[:,2].min()

extent_max_x = ele_aabb[:,3].max()
extent_max_y = ele_aabb[:,4].max()
extent_max_z = ele_aabb[:,5].max()
num_ELEMENTS = ele_aabb.shape[0]
#-------------------------------------------------
#morton codes part
pcMortonCodes = m_morton_codes.pushConstantsMortonCodes(
    g_num_elements=num_ELEMENTS, g_min_x=extent_min_x, g_min_y=extent_min_y, g_min_z=extent_min_z,
    g_max_x=extent_max_x, g_max_y=extent_max_y, g_max_z=extent_max_z
)
morton_codes_ele = torch.zeros((num_ELEMENTS, 2), dtype=torch.int).cuda()

m_morton_codes.morton_codes(pc=pcMortonCodes, ele_aabb=ele_aabb, morton_codes_ele=morton_codes_ele)\
.launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

#--------------------------------------------------
# radix sort part
morton_codes_ele_pingpong = torch.zeros((num_ELEMENTS, 2), dtype=torch.int).cuda()
m_radixsort.radix_sort(g_num_elements=int(num_ELEMENTS), g_elements_in=morton_codes_ele, g_elements_out=morton_codes_ele_pingpong)\
.launchRaw(blockSize=(256, 1, 1), gridSize=(1, 1, 1))

#--------------------------------------------------
# hierarchy
num_LBVH_ELEMENTS = num_ELEMENTS + num_ELEMENTS - 1
LBVHNode_info = torch.zeros((num_LBVH_ELEMENTS, 3), dtype=torch.int).cuda()
LBVHNode_aabb = torch.zeros((num_LBVH_ELEMENTS, 6), dtype=torch.float).cuda()
LBVHConstructionInfo = torch.zeros((num_LBVH_ELEMENTS, 2), dtype=torch.int).cuda()

m_hierarchy.hierarchy(g_num_elements=int(num_ELEMENTS), ele_primitiveIdx=ele_primitiveIdx, ele_aabb=ele_aabb,
                      g_sorted_morton_codes=morton_codes_ele, g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, g_lbvh_construction_infos=LBVHConstructionInfo)\
.launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

#--------------------------------------------------
# bounding_boxes
'''
m_bounding_box.bounding_boxes(g_num_elements=int(num_ELEMENTS), g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, g_lbvh_construction_infos=LBVHConstructionInfo)\
.launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))


LBVHConstructionInfo[:,1] = 0
m_bounding_box.bounding_boxes(g_num_elements=int(num_ELEMENTS), g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, g_lbvh_construction_infos=LBVHConstructionInfo)\
.launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))
'''

#'''
tree_heights = torch.zeros((num_ELEMENTS, 1), dtype=torch.int).cuda()
m_bounding_box.get_bvh_height(g_num_elements=int(num_ELEMENTS), g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                              g_lbvh_construction_infos=LBVHConstructionInfo, tree_heights=tree_heights)\
.launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

tree_height_max = tree_heights.max()
for i in range(tree_height_max):
    m_bounding_box.get_bbox(g_num_elements=int(num_ELEMENTS), expected_height=int(i+1),
                        g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb, 
                                g_lbvh_construction_infos=LBVHConstructionInfo)\
    .launchRaw(blockSize=(256, 1, 1), gridSize=((num_ELEMENTS+255)//256, 1, 1))

m_bounding_box.set_root(
              g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb)\
    .launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1))    
#'''
#'''
end_time = time.time()
elapsed_time = end_time - start_time
print(f"GPU bvh build finished in: {elapsed_time} s")

#write
LBVHbuffer = np.column_stack((LBVHNode_info.cpu().numpy(), LBVHNode_aabb.cpu().numpy()))

np.set_printoptions(suppress=True)
with open('./data.csv', 'w') as csvfile:
    csvfile.write("left right primitiveIdx aabb_min_x aabb_min_y aabb_min_z aabb_max_x aabb_max_y aabb_max_z\n")  
    np.savetxt(csvfile, LBVHbuffer, delimiter=' ', fmt='%d %d %d %f %f %f %f %f %f')

#debug
'''
sorted_mc_codes = morton_codes_ele.cpu().numpy()
with open('./sorted_mc.txt', 'w') as mc:
    #np.set_printoptions(suppress=True)
    np.savetxt(mc, sorted_mc_codes, delimiter=' ', fmt='%d')

lbvhconinfo = LBVHConstructionInfo.cpu().numpy()
with open('./lbvhconinfo.txt', 'w') as mc:
    np.set_printoptions(suppress=True)
    np.savetxt(mc, lbvhconinfo, delimiter=' ', fmt='%d')

lbvh_aabb = LBVHNode_aabb.cpu().numpy()
with open('./lbvhaabb.txt', 'w') as mc:
    np.set_printoptions(suppress=True)
    np.savetxt(mc, lbvh_aabb, delimiter=' ', fmt='%f')

my_ele_aabb = ele_aabb.cpu().numpy()
with open('./my_ele_aabb.txt', 'w') as mc:
    np.set_printoptions(suppress=True)
    np.savetxt(mc, my_ele_aabb, delimiter=' ', fmt='%g')

'''
print("bvh build over!")

# generating rays
y, x = torch.meshgrid([torch.linspace(1, -1, 800), 
                       torch.linspace(-1, 1, 800)], indexing='ij')
z = -torch.ones_like(x)
ray_directions = torch.stack([x, y, z], dim=-1).cuda()
ray_origins = torch.Tensor([0, 0.1, 0.3]).cuda().broadcast_to(ray_directions.shape)

ray_origins = ray_origins.contiguous().reshape(-1,3)
ray_directions = ray_directions.contiguous().reshape(-1,3)

num_rays=ray_origins.shape[0]

start_time = time.time()
hit = torch.zeros((num_rays, 1), dtype=torch.int).cuda()
hit_pos_map = torch.zeros((num_rays, 3), dtype=torch.float).cuda()

m_intersect_test.intersect(num_rays=int(num_rays), rays_o=ray_origins, rays_d=ray_directions,
               g_lbvh_info=LBVHNode_info, g_lbvh_aabb=LBVHNode_aabb,
               vert=vrt, v_indx=v_ind,
               hit_map=hit, hit_pos_map=hit_pos_map)\
.launchRaw(blockSize=(256, 1, 1), gridSize=((num_rays+255)//256, 1, 1))
end_time = time.time()

elapsed_time = end_time - start_time
print("ray query time:", elapsed_time, "s")

# drawing result
#locs = hit.repeat(1,3)
locs = hit_pos_map

pyexr.write(f'./color.exr', locs.reshape(800,800,3).cpu().numpy())