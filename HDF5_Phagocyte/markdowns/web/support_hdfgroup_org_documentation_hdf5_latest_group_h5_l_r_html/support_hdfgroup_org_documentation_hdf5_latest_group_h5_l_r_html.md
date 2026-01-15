Help us improve by taking our short survey: <https://www.hdfgroup.org/website-survey/>
![Logo](https://support.hdfgroup.org/documentation/hdf5/latest/HDFG-logo.png) |  HDF5 Last Updated on 2026-01-10 The HDF5 Field Guide  
---|---  
  * [Main Page](https://support.hdfgroup.org/documentation/hdf5/latest/index.html)
  * [Getting started](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
  * [User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * [Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html)
  * [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
  * [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * [RFCs](https://support.hdfgroup.org/documentation/hdf5/latest/_r_f_c.html)
  * [Specifications](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html)
  * [Glossary](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html)
  * [Full-Text Search](https://support.hdfgroup.org/documentation/hdf5/latest/_f_t_s.html)
  * [About](https://support.hdfgroup.org/documentation/hdf5/latest/_about.html)
  * [![](https://support.hdfgroup.org/documentation/hdf5/latest/search/close.svg)](javascript:searchBox.CloseResultsWindow\(\))


### Table of contents
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title2 "H2")
  * [◆ H5LRcopy_reference()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title3 "H2")
  * [◆ H5LRcopy_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title4 "H2")
  * [◆ H5LRcreate_ref_to_all()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title5 "H2")
  * [◆ H5LRcreate_region_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title6 "H2")
  * [◆ H5LRget_region_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title7 "H2")
  * [◆ H5LRmake_dataset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title8 "H2")
  * [◆ H5LRread_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title9 "H2")
  * [◆ H5LTcopy_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title10 "H2")
  * [◆ H5LTread_bitfield_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title11 "H2")
  * [◆ H5LTread_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#title12 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#func-members)
Extensions
## Detailed Description
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html)
* * *
_Working with region references, hyperslab selections, and bit-fields (H5LR, H5LT)_
The following reference manual entries describe high-level HDF5 C and Fortran APIs for working with region references, hyperslab selections, and bit-fields. These functions were created as part of a project supporting NPP/NPOESS Data Production and Exploitation ( [project](https://support.hdfgroup.org/archive/support/projects/jpss/documentation/index.html), [software](https://support.hdfgroup.org/archive/support/projects/npoess/HL_JPSS/index.html)). While they were written to facilitate access to NPP, NPOESS, and JPSS data in the HDF5 format, these functions may be useful to anyone working with region references, hyperslab selections, or bit-fields.
Note that these functions are not part of the standard HDF5 distribution; the [software](https://support.hdfgroup.org/archive/support/projects/npoess/HL_JPSS/index.html) must be separately downloaded and installed.
A comprehensive guide to this library, [_User Guide to the HDF5 High-level Library for Handling Region References and Hyperslab Selections_](https://support.hdfgroup.org/archive/support/projects/jpss/documentation/HL/UG/NPOESS_HL-UG.pdf) is available at <https://support.hdfgroup.org/archive/support/projects/jpss/documentation/HL/UG/NPOESS_HL-UG.pdf>.
  * [H5LRcopy_reference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga889b19b609e4e44565467cb84c6eeb0d)   
Copies data from the specified dataset to a new location and creates a reference to it.
  * [H5LRcopy_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga3056840a05520e3a5e1b77f8cdd373c4)   
Copies data from a referenced region to a region in a destination dataset.
  * [H5LRcreate_ref_to_all](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga75787f84657d69421c6a8be57a9f070c)   
Creates a dataset with the region references to the data in all datasets located under a specified group in a file or creates a dataset with object references to all objects (groups or datasets) located under a specified group in a file.
  * [H5LRcreate_region_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gacef1564aa36bdfceac13b7b2bdd8e2d8)   
Creates an array of region references using an array of paths to datasets and an array of corresponding hyperslab descriptions.
  * [H5LRget_region_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gabb053f246fd0cfcb588bafe98278b9b8)   
Retrieves information about the data a region reference points to.
  * [H5LRmake_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gabbf9f8ea4d13d44d45efae5e143a157a)   
Creates and writes a dataset containing a list of region references.
  * [H5LRread_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga0cab2622e4ffba04c363d37b98b0a83b)   
Retrieves raw data pointed to by a region reference to an application buffer.
  * [H5LTcopy_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga73fb3ab3e9b756a818e711cd335c819d)   
Copies data from a specified region in a source dataset to a specified region in a destination dataset.
  * [H5LTread_bitfield_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga1eec1ca149e662c4937e4ab10c30e0f6)   
Retrieves the values of quality flags for each element to the application provided buffer.
  * [H5LTread_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga291c97f18425bb1136f1eab304fb0722)   
Reads selected data to an application buffer. 


##  Functions  
---  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LRcopy_reference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga889b19b609e4e44565467cb84c6eeb0d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) *ref, const char *file, const char *path, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *block_coord, [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) *ref_new)  
| Copies data from the specified dataset to a new location and creates a reference to it.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LRcopy_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga3056840a05520e3a5e1b77f8cdd373c4) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) *ref, const char *file, const char *path, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *block_coord)  
| Copies data from a referenced region to a region in a destination dataset.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LRcreate_ref_to_all](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga75787f84657d69421c6a8be57a9f070c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_path, const char *ds_path, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) index_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) ref_type)  
| Creates a dataset with the region references to the data in all datasets located under a specified group in a file or creates a dataset with object references to all objects (groups or datasets) located under a specified group in a file.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LRcreate_region_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gacef1564aa36bdfceac13b7b2bdd8e2d8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, size_t num_elem, const char **path, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *block_coord, [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) *buf)  
| Creates an array of region references using an array of paths to datasets and an array of corresponding hyperslab descriptions.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LRget_region_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gabb053f246fd0cfcb588bafe98278b9b8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) *ref, size_t *len, char *path, int *rank, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *dtype, [H5S_sel_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a1e9590539381e3922a1582067d496304) *sel_type, size_t *numelem, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *buf)  
| Retrieves information about the data a region reference points to.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LRmake_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gabbf9f8ea4d13d44d45efae5e143a157a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *path, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, const size_t buf_size, const [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *loc_id_ref, const [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) *buf)  
| Creates and writes a dataset containing a list of region references.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LRread_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga0cab2622e4ffba04c363d37b98b0a83b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) *ref, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type, size_t *numelem, void *buf)  
| Retrieves raw data pointed to by a region reference to an application buffer.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTcopy_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga73fb3ab3e9b756a818e711cd335c819d) (const char *file_src, const char *path_src, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *block_coord_src, const char *file_dest, const char *path_dest, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *block_coord_dset)  
| Copies data from a specified region in a source dataset to a specified region in a destination dataset.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_bitfield_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga1eec1ca149e662c4937e4ab10c30e0f6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, int num_values, const unsigned *offset, const unsigned *lengths, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space, int *buf)  
| Retrieves the values of quality flags for each element to the application provided buffer.   
  
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5LTread_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga291c97f18425bb1136f1eab304fb0722) (const char *file, const char *path, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *block_coord, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type, void *buf)  
| Reads selected data to an application buffer.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga889b19b609e4e44565467cb84c6eeb0d)H5LRcopy_reference()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LRcopy_reference  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  |  [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) * |  _ref_ ,   
|  | const char * |  _file_ ,   
|  | const char * |  _path_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _block_coord_ ,   
|  |  [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) * |  _ref_new_ )  
Copies data from the specified dataset to a new location and creates a reference to it. 
* * * 

Parameters
     [in] | obj_id | Identifier of any object in a file an HDF5 reference belongs to   
---|---|---  
[in] | ref | Reference to the datasets region   
[in] | file | Name of the destination file   
[in] | path | Full path to the destination dataset   
[in] | block_coord | Hyperslab coordinates in the destination dataset   
[out] | ref_new | Region reference to the new location of data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Given a data set pointed to by a region reference, the function [H5LRcopy_reference()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga889b19b609e4e44565467cb84c6eeb0d "Copies data from the specified dataset to a new location and creates a reference to it.") will copy the hyperslab data referenced by a datasets region reference into existing dataset specified by its path `path` in the file with the name `file`, and to location specified by the hyperslab coordinates `block_coord`. It will create the region reference `ref_new` to point to the new location. The number of elements in the old and newly specified regions has to be the same.
Buffer `block_coord` has size 2*rank and is the coordinates of the starting point following by the coordinates of the ending point of the hyperslab. For example, to extract a rectangular hyperslab region starting at element (2,2) to element (5,4) then `block_coord` would be {2, 2, 5, 4}. 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga3056840a05520e3a5e1b77f8cdd373c4)H5LRcopy_region()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LRcopy_region  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  |  [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) * |  _ref_ ,   
|  | const char * |  _file_ ,   
|  | const char * |  _path_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _block_coord_ )  
Copies data from a referenced region to a region in a destination dataset. 
* * * 

Parameters
     [in] | obj_id | Identifier of any object in a file dataset region reference belongs to   
---|---|---  
[in] | ref | Dataset region reference   
[in] | file | Name of the destination file   
[in] | path | Full path to the destination dataset   
[in] | block_coord | Hyperslab coordinates in the destination dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Given a dataset region reference `ref` in a source file specified by an identifier of any object in that file `obj_id`, the function will write data to the existing dataset `path` in file `file` to the simple hyperslab specified by `block_coord`.
Buffer `block_coord` has size 2*rank and is the coordinates of the starting point following by the coordinates of the ending point of the hyperslab. For example, to specify a rectangular hyperslab destination region starting at element (2,2) to element (5,4) then `block_coord` would be {2, 2, 5, 4}.
If `path` does not exist in the destination file (as may be the case when writing to a new file) then the dataset will be copied directly to the `path` and `block_coord` will be disregarded. 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga75787f84657d69421c6a8be57a9f070c)H5LRcreate_ref_to_all()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LRcreate_ref_to_all  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_path_ ,   
|  | const char * |  _ds_path_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _index_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [H5R_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a7d148ddc165e65b01efabfd738ac056e) |  _ref_type_ )  
Creates a dataset with the region references to the data in all datasets located under a specified group in a file or creates a dataset with object references to all objects (groups or datasets) located under a specified group in a file. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | group_path | Absolute or relative path to the group at which traversal starts   
[in] | ds_path | Absolute or relative path to the dataset with region references to be created   
[in] | index_type | Index_type; see valid values below in description   
[in] | order | Order in which index is traversed; see valid values below in description   
[in] | ref_type | Reference type; see valid values below in description 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5LRcreate_ref_to_all()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga75787f84657d69421c6a8be57a9f070c "Creates a dataset with the region references to the data in all datasets located under a specified gr...") creates a dataset with the region references to the data in all datasets located under a specified group in a file or creates a dataset with object references to all objects (groups or datasets) located under a specified group in a file.
Given a dataset path `ds_path` in a file specified by the `loc_id` identifier, the function [H5LRcreate_ref_to_all()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga75787f84657d69421c6a8be57a9f070c "Creates a dataset with the region references to the data in all datasets located under a specified gr...") will create a contiguous one-dimensional dataset with the region references or object references depending on the value of the `ref_type` parameter. When `ref_type` is [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), each region reference points to all data in a dataset encountered by an internally called [H5Lvisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) routine, which starts at the group specified by the `loc_id` and `group_path` parameters. In a like manner, when `ref_type` is [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), each object reference points to an object (a group or a dataset) encountered by [H5Lvisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69).
If `ds_path` does not exist in `loc_id` then the function will create the path specified by `ds_path` automatically.
`index_type` specifies the index to be used. Valid values include the following:
  * [H5_INDEX_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3a644e6701706be4d37660864336c7bd3e) Alphanumeric index on name
  * [H5_INDEX_CRT_ORDER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3a315475479b34056043b4b6583bafb280) Index on creation order


`order` specifies the order in which objects are to be inspected along the index specified in `index_type`. Valid values include the following:
  * [H5_ITER_INC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a80c3e083c0a77063b1a66553decfcb08) Increasing order
  * [H5_ITER_DEC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a2f123d7a4d68054e8aa2d0f1d0a3fcd2) Decreasing order
  * [H5_ITER_NATIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a15eebaf7d85d5e37e918858634e29d01) Fastest available order


For more detailed information on these two parameters, see [H5Lvisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69).
`ref_type` specifies the type of the reference to be used. Valid values include the following:
  * [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0) Dataset region reference
  * [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2) Object reference



Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gacef1564aa36bdfceac13b7b2bdd8e2d8)H5LRcreate_region_references()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LRcreate_region_references  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | size_t |  _num_elem_ ,   
|  | const char ** |  _path_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _block_coord_ ,   
|  |  [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) * |  _buf_ )  
Creates an array of region references using an array of paths to datasets and an array of corresponding hyperslab descriptions. 
* * * 

Parameters
     [in] | obj_id | File identifier for the HDF5 file containing the referenced regions or an object identifier for any object in that file   
---|---|---  
[in] | num_elem | Number of elements in the `path` and `buf` arrays   
[in] | path | Array of pointers to strings, which contain the paths to the target datasets for the region references   
[in] | block_coord | Array of hyperslab coordinate   
[out] | buf | Buffer for returning an array of region references 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Note
     **Motivation:**      [H5LRcreate_region_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gacef1564aa36bdfceac13b7b2bdd8e2d8 "Creates an array of region references using an array of paths to datasets and an array of correspondi...") is useful when creating large numbers of similar region references.  
[H5LRcreate_region_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gacef1564aa36bdfceac13b7b2bdd8e2d8 "Creates an array of region references using an array of paths to datasets and an array of correspondi...") creates a list of region references given an array of paths to datasets and another array listing the corner coordinates of the corresponding hyperslabs.
`path` parameter is an array of pointers to strings.
`num_elem` specifies the number of region references to be created, thus specifying the size of the `path` and `_buf` arrays.
Buffer `block_coord` has size 2*rank and is the coordinates of the starting point following by the coordinates of the ending point of the hyperslab, repeated `num_elem` times for each hyperslab. For example, creating two region references to two hyperslabs, one with a rectangular hyperslab region starting at element (2,2) to element (5,4) and the second rectangular region starting at element (7,7) to element (9,10), results in `block_coord` being {2,2,5,4, 7,7,9,10}.
The rank of the hyperslab will be the same as the rank of the target dataset. [H5LRcreate_region_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gacef1564aa36bdfceac13b7b2bdd8e2d8 "Creates an array of region references using an array of paths to datasets and an array of correspondi...") will retrieve the rank for each dataset and will use those values to interpret the values in the buffer. Please note that rank may vary from one dataset to another. 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gabb053f246fd0cfcb588bafe98278b9b8)H5LRget_region_info()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LRget_region_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | const [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) * |  _ref_ ,   
|  | size_t * |  _len_ ,   
|  | char * |  _path_ ,   
|  | int * |  _rank_ ,   
|  |  [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) * |  _dtype_ ,   
|  |  [H5S_sel_type](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a1e9590539381e3922a1582067d496304) * |  _sel_type_ ,   
|  | size_t * |  _numelem_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _buf_ )  
Retrieves information about the data a region reference points to. 
* * * 

Parameters
     [in] | obj_id | Identifier of any object in an HDF5 file the region reference belongs to.   
---|---|---  
[in] | ref | Region reference to query   
[in,out] | len | Size of the buffer to store `path` in. NOTE: if `*path` is not NULL then `*len` must be the appropriate length   
[out] | path | Full path that a region reference points to   
[out] | rank | The number of dimensions of the dataset dimensions of the dataset pointed by region reference.   
[out] | dtype | Datatype of the dataset pointed by the region reference.   
[out] | sel_type | Type of the selection (point or hyperslab)   
[in,out] | numelem | Number of coordinate blocks or selected elements.   
[out] | buf | Buffer containing description of the region pointed by region reference 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5LRget_region_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gabb053f246fd0cfcb588bafe98278b9b8 "Retrieves information about the data a region reference points to.") queries information about the data pointed by a region reference `ref`. It returns one of the absolute paths to a dataset, length of the path, dataset's rank and datatype, description of the referenced region and type of the referenced region. Any output argument can be NULL if that argument does not need to be returned.
The parameter `obj_id` is an identifier for any object in the HDF5 file containing the referenced object. For example, it can be an identifier of a dataset the region reference belongs to or an identifier of an HDF5 file the dataset with region references is stored in.
The parameter `ref` is a region reference to query.
The parameter `path` is a pointer to application allocated buffer of size `len+1` to return an absolute path to a dataset the region reference points to.
The parameter `len` is a length of absolute path string plus the \0 string terminator. If path parameter is NULL, actual length of the path (+1 for \0 string terminator) is returned to application and can be used to allocate buffer path of an appropriate length `len`.
The parameter `sel_type` describes the type of the selected region. Possible values can be [H5S_SEL_POINTS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a1e9590539381e3922a1582067d496304a858daf05b19591a8f5d6ffbee281f81c) for point selection and [H5S_SEL_HYPERSLABS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a1e9590539381e3922a1582067d496304aeb9dc925cd7c6a6702fc9c766e20b01a) for hyperslab selection.
The parameter `numelem` describes how many elements will be placed in the buffer `buf`. The number should be interpreted using the value of `sel_type`.
If value of `sel_type` is [H5S_SEL_HYPERSLABS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a1e9590539381e3922a1582067d496304aeb9dc925cd7c6a6702fc9c766e20b01a), the parameter `buf` contains `numelem` blocks of the coordinates for each simple hyperslab of the referenced region. Each block has length `2*`rank` and` is organized as follows: <"start" coordinate>, immediately followed by <"opposite" corner coordinate>. The total size of the buffer to hold the description of the region will be `2*`rank*`numelem`.` If` region reference points to a contiguous sub-array, then the value of `numelem` is 1 and the block contains coordinates of the upper left and lower right corners of the sub-array (or simple hyperslab).
If value of `sel_type` is [H5S_SEL_POINTS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a1e9590539381e3922a1582067d496304a858daf05b19591a8f5d6ffbee281f81c), the parameter `buf` contains `numelem` blocks of the coordinates for each selected point of the referenced region. Each block has length `rank` and contains coordinates of the element. The total size of the buffer to hold the description of the region will be `rank*` `numelem`. 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#gabbf9f8ea4d13d44d45efae5e143a157a)H5LRmake_dataset()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LRmake_dataset  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _path_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | const size_t |  _buf_size_ ,   
|  | const [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) * |  _loc_id_ref_ ,   
|  | const [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) * |  _buf_ )  
Creates and writes a dataset containing a list of region references. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | path | Path to the dataset being created   
[in] | type_id | Datatype of the dataset   
[in] | buf_size | Size of the `loc_id_ref` and `buf` arrays   
[in] | loc_id_ref | Array of object identifiers; each identifier describes to which HDF5 file the corresponding region reference belongs to   
[in] | buf | Array of region references 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Given an array of size `buf_size` of region references `buf`, the function will create a dataset with path `path`, at location specified by `loc_id` and of a datatype specified by `type_id`, and will write data associated with each region reference in the order corresponding to the order of the region references in the buffer. It is assumed that all referenced hyperslabs have the same dimensionality, and only the size of the slowest changing dimension may differ. Each reference in the `buf` array belongs to the file identified by the corresponding object identifiers in the array `loc_id_ref`.
If `path` does not exist in `loc_id` then the function will create the path specified by `path` automatically. 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga0cab2622e4ffba04c363d37b98b0a83b)H5LRread_region()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LRread_region  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | const [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) * |  _ref_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_ ,   
|  | size_t * |  _numelem_ ,   
|  | void * |  _buf_ )  
Retrieves raw data pointed to by a region reference to an application buffer. 
* * * 

Parameters
     [in] | obj_id | File identifier for the HDF5 file containing the dataset with the referenced region or an object identifier for any object in that file   
---|---|---  
[in] | ref | Region reference specifying data to be read in   
[in] | mem_type | Memory datatype of data read from referenced region into the application buffer   
[in,out] | numelem | Number of elements to be read into buffer `buf`  
[out] | buf | Buffer in which data is returned to the application 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5LRread_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga0cab2622e4ffba04c363d37b98b0a83b "Retrieves raw data pointed to by a region reference to an application buffer.") reads data pointed to by the region reference `ref` into the buffer `buf`.
`numelem` specifies the number of elements to be read into `buf`. When the size of the reference region is unknown, [H5LRread_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga0cab2622e4ffba04c363d37b98b0a83b "Retrieves raw data pointed to by a region reference to an application buffer.") can be called with `buf` set to NULL; the number of elements in the referenced region will be returned in `numelem`.
The buffer buf must be big enough to hold `numelem` elements of type `mem_type`. For example, if data is read from the referenced region into an integer buffer, `mem_type` should be [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) and the buffer must be at least `sizeof(int)` * `numelem` bytes in size. This buffer must be allocated by the application. 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga73fb3ab3e9b756a818e711cd335c819d)H5LTcopy_region()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LTcopy_region  | ( | const char * |  _file_src_ ,   
---|---|---|---  
|  | const char * |  _path_src_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _block_coord_src_ ,   
|  | const char * |  _file_dest_ ,   
|  | const char * |  _path_dest_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _block_coord_dset_ )  
Copies data from a specified region in a source dataset to a specified region in a destination dataset. 
* * * 

Parameters
     [in] | file_src | Name of the source file   
---|---|---  
[in] | path_src | Full path to the source dataset   
[in] | block_coord_src | Hyperslab coordinates in the source dataset   
[in] | file_dest | Name of the destination file   
[in] | path_dest | Full path to the destination dataset   
[in] | block_coord_dset | Hyperslab coordinates in the destination dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Given a path to a dataset `path_src` in a file with the name `file_src`, and description of a simple hyperslab of the source `block_coord_src`, the function will write data to the dataset `path_dest` in file `file_dest` to the simple hyperslab specified by `block_coord_dset`. The arrays `block_coord_src` and `block_coord_dset` have a length of 2*rank and are the coordinates of the starting point following by the coordinates of the ending point of the hyperslab. For example, to specify a rectangular hyperslab destination region starting at element (2,2) to element (5,4) then `block_coord_dset` would be {2, 2, 5, 4}.
If `path_dest` does not exist in the destination file (as may be the case when writing to a new file) then the dataset will be copied directly to the `path_dest` and `block_coord_dset` will be disregarded. 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga1eec1ca149e662c4937e4ab10c30e0f6)H5LTread_bitfield_value()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LTread_bitfield_value  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | int |  _num_values_ ,   
|  | const unsigned * |  _offset_ ,   
|  | const unsigned * |  _lengths_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_ ,   
|  | int * |  _buf_ )  
Retrieves the values of quality flags for each element to the application provided buffer. 
* * * 

Parameters
     [in] | dset_id | Identifier of the dataset with bit-field values   
---|---|---  
[in] | num_values | Number of the values to be extracted   
[in] | offset | Array of staring bits to be extracted from the element; valid values: 0 (zero) through 7   
[in] | lengths | Array of the number of bits to be extracted for each value   
[in] | space | Dataspace identifier, describing the elements to be read from the dataset with bit-field values   
[out] | buf | Buffer to read the values in 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5LTread_bitfield_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga1eec1ca149e662c4937e4ab10c30e0f6 "Retrieves the values of quality flags for each element to the application provided buffer.") reads selected elements from a dataset specified by its identifier `dset_id`, and unpacks the bit-field values to a buffer `buf`.
The parameter `space` is a space identifier that indicates which elements of the dataset should be read.
The parameter `offset` is an array of length `num_values`; the ith element of the array holds the value of the starting bit of the ith bit-field value. Valid values are: 0 (zero) through 7.
The parameter `lengths` is an array of length `num_values`; the ith element of the array holds the number of bits to be extracted for the ith bit-field value. Extracted bits will be interpreted as a base-2 integer value. Each value will be converted to the base-10 integer value and stored in the application buffer.
Buffer `buf` is allocated by the application and should be big enough to hold `num_sel_elem` * `num_values` elements of the specified type, where `num_sel_elem` is a number of the elements to be read from the dataset. Data in the buffer is organized as `num_values` values for the first element, followed by the `num_values` values for the second element, ... , followed by the `num_values` values for the `num_selected_elemth` element. 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga291c97f18425bb1136f1eab304fb0722)H5LTread_region()
H5_HLRDLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5LTread_region  | ( | const char * |  _file_ ,   
---|---|---|---  
|  | const char * |  _path_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _block_coord_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_ ,   
|  | void * |  _buf_ )  
Reads selected data to an application buffer. 
* * * 

Parameters
     [in] | file | Name of file   
---|---|---  
[in] | path | Full path to a dataset   
[in] | block_coord | Hyperslab coordinates   
[in] | mem_type | Memory datatype, describing the buffer the referenced data will be read into   
[out] | buf | Buffer containing data from the referenced region 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5LTread_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_r.html#ga291c97f18425bb1136f1eab304fb0722 "Reads selected data to an application buffer.") reads data from a region described by the hyperslab coordinates in `block_coord`, located in the dataset specified by its absolute path `path` in a file specified by its name `file`. Data is read into a buffer `buf` of the datatype that corresponds to the HDF5 datatype specified by `mem_type`.
Buffer `block_coord` has size 2*rank and is the coordinates of the starting point following by the coordinates of the ending point of the hyperslab. For example, to extract a rectangular hyperslab region starting at element (2,2) to element (5,4) then `block_coord` would be {2, 2, 5, 4}.
Buffer `buf` should be big enough to hold selected elements of the type that corresponds to the `mem_type` 

Version
    1.1 Fortran wrapper introduced in this release. 

Since
    1.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


