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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#title2 "H2")
  * [◆ H5Pget_create_intermediate_group()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#title3 "H2")
  * [◆ H5Pset_create_intermediate_group()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#func-members)
Link Creation Properties
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) » [String Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_t_r_c_p_l.html)
## Detailed Description
This creation property applies to links only, and advises the library to automatically create missing intermediate groups when creating new objects. 
Link creation property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_create_intermediate_group](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c "Specifies in property list whether to create missing intermediate groups.")/[H5Pget_create_intermediate_group](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#gaf7db1b7ce19703f30f1827b7c899c3b0 "Determines whether property is set to enable creating missing intermediate groups.") | Specifies/retrieves whether to create missing intermediate groups.  

See also
     [String Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_t_r_c_p_l.html)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_create_intermediate_group](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#gaf7db1b7ce19703f30f1827b7c899c3b0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned *crt_intmd)  
| Determines whether property is set to enable creating missing intermediate groups.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_create_intermediate_group](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned crt_intmd)  
| Specifies in property list whether to create missing intermediate groups.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#gaf7db1b7ce19703f30f1827b7c899c3b0)H5Pget_create_intermediate_group()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_create_intermediate_group  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned * |  _crt_intmd_ )  
Determines whether property is set to enable creating missing intermediate groups.  

Parameters
     [in] | plist_id | Link creation property list identifier   
---|---|---  
[out] | crt_intmd | Flag specifying whether to create intermediate groups upon creation of an object 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_create_intermediate_group()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#gaf7db1b7ce19703f30f1827b7c899c3b0 "Determines whether property is set to enable creating missing intermediate groups.") determines whether the link creation property list `plist_id` is set to allow functions that create objects in groups different from the current working group to create intermediate groups that may be missing in the path of a new or moved object.
Functions that create objects in or move objects to a group other than the current working group make use of this property. [H5Gcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gab52641f0736281faaaae4e3039bbb344 "Creates a new empty group without linking it into the file structure.") and [H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file.") are examples of such functions.
If `crt_intmd` is positive, missing intermediate groups will be created; if `crt_intmd` is non-positive, missing intermediate groups will not be created. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c)H5Pset_create_intermediate_group()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_create_intermediate_group  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned |  _crt_intmd_ )  
Specifies in property list whether to create missing intermediate groups.  

Parameters
     [in] | plist_id | Link creation property list identifier   
---|---|---  
[in] | crt_intmd | Flag specifying whether to create intermediate groups upon the creation of an object 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_create_intermediate_group()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c "Specifies in property list whether to create missing intermediate groups.") 

Since
    1.8.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


