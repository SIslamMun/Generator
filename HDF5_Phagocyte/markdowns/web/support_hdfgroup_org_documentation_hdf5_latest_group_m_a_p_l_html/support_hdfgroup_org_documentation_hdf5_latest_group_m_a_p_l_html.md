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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#title2 "H2")
  * [◆ H5Pget_map_iterate_hints()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#title3 "H2")
  * [◆ H5Pset_map_iterate_hints()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#func-members)
VOL Data Mapping Properties
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html)
## Detailed Description
Empty property class. 
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_map_iterate_hints](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#ga0bfdb1161ba396054be37f5c580b0b94) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mapl_id, size_t *key_prefetch_size, size_t *key_alloc_size)  
| Set map iteration hints.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_map_iterate_hints](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#gaac4a9850acde5e7f872e2b10e440f98e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mapl_id, size_t key_prefetch_size, size_t key_alloc_size)  
| Set map iteration hints.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#ga0bfdb1161ba396054be37f5c580b0b94)H5Pget_map_iterate_hints()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_map_iterate_hints  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mapl_id_ ,   
---|---|---|---  
|  | size_t * |  _key_prefetch_size_ ,   
|  | size_t * |  _key_alloc_size_ )  
Set map iteration hints.  

Parameters
     [in] | mapl_id | Map access property list identifier   
---|---|---  
[out] | key_prefetch_size | Pointer to number of keys to prefetch at a time during iteration   
[out] | key_alloc_size | Pointer to the initial size of the buffer allocated to hold prefetched keys  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
H5Pget_map_iterate() returns the map iterate hints, `key_prefetch_size` and `key_alloc_size`, as set by [H5Pset_map_iterate_hints()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#gaac4a9850acde5e7f872e2b10e440f98e "Set map iteration hints."). 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#gaac4a9850acde5e7f872e2b10e440f98e)H5Pset_map_iterate_hints()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_map_iterate_hints  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mapl_id_ ,   
---|---|---|---  
|  | size_t |  _key_prefetch_size_ ,   
|  | size_t |  _key_alloc_size_ )  
Set map iteration hints.  

Parameters
     [in] | mapl_id | Map access property list identifier   
---|---|---  
[in] | key_prefetch_size | Number of keys to prefetch at a time during iteration   
[in] | key_alloc_size | The initial size of the buffer allocated to hold prefetched keys  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_map_iterate_hints()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html#gaac4a9850acde5e7f872e2b10e440f98e "Set map iteration hints.") adjusts the behavior of [H5Miterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gaff5dc81a8a1150acee7e729a7747f1f2 "Iterates over all key-value pairs in a map object.") when prefetching keys for iteration. The `key_prefetch_size` parameter specifies the number of keys to prefetch at a time during iteration. The `key_alloc_size` parameter specifies the initial size of the buffer allocated to hold these prefetched keys. If this buffer is too small it will be reallocated to a larger size, though this may result in an additional I/O. 

Since
    1.12.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


