import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import trimesh
import pyexr
import slangpy


m_gen_ele = slangpy.loadModule('bvhworkers/get_elements.slang')
m_morton_codes = slangpy.loadModule('bvhworkers/lbvh_morton_codes.slang')
m_radixsort = slangpy.loadModule('bvhworkers/lbvh_single_radixsort.slang')

mesh = trimesh.load('./models/dragon.obj')

vrt = torch.from_numpy(mesh.vertices).cuda().float()
v_ind = torch.from_numpy(mesh.faces).cuda().int()

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


#debug
input = torch.tensor((32, 31, 8, 7), dtype=torch.int).cuda()
output = torch.zeros_like(input).cuda()
m_gen_ele.debug_cb(a=input, b=output)\
.launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1))
print(output)
print("over!")