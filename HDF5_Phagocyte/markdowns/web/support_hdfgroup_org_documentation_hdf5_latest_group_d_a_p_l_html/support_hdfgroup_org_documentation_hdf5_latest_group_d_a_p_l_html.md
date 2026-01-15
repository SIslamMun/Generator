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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title2 "H2")
  * [◆ H5Pget_append_flush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title3 "H2")
  * [◆ H5Pget_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title4 "H2")
  * [◆ H5Pget_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title5 "H2")
  * [◆ H5Pget_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title6 "H2")
  * [◆ H5Pget_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title7 "H2")
  * [◆ H5Pget_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title8 "H2")
  * [◆ H5Pset_append_flush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title9 "H2")
  * [◆ H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title10 "H2")
  * [◆ H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title11 "H2")
  * [◆ H5Pset_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title12 "H2")
  * [◆ H5Pset_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title13 "H2")
  * [◆ H5Pset_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#title14 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#func-members)
Dataset Access Properties
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) » [Link Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html)
## Detailed Description
Use dataset access properties to modify the default behavior of the HDF5 library when accessing datasets. The properties include adjusting the size of the chunk cache, providing prefixes for external content and virtual dataset file paths, and controlling flush behavior, etc. These properties are _not_ persisted with datasets, and can be adjusted at runtime before a dataset is created or opened. 
Dataset access property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_buffer](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga777e8c171c9e462230a9fa40874b38ce "Sets type conversion and background buffers.") | Sets type conversion and background buffers.   
[H5Pget_buffer](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga1278b9979cc833e77d699cc878c6dab4 "Reads buffer settings.") | Reads buffer settings.   
[H5Pset_append_flush](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga2f685a7b3f3a4fa35ddcd1659ab4a835 "Sets two actions to perform when the size of a dataset's dimension being appended reaches a specified...")/[H5Pget_append_flush](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gacd6803640eebd20e408c330192b09fa6 "Retrieves the values of the append property that is set up in the dataset access property list.") | Sets/gets the values of the append property that is set up in the dataset access property list.   
[H5Pset_chunk_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters.")/[H5Pget_chunk_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gaeda015dfee4167cc60baab1d1f0560fe "Retrieves the raw data chunk cache parameters.") | Sets/gets the raw data chunk cache parameters.   
[H5Pset_efile_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.")/[H5Pget_efile_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga442647d48171db920c71a7baf6fdeee6 "Retrieves the prefix for external raw data storage files as set in the dataset access property list.") | Sets/gets the prefix for external raw data storage files as set in the dataset access property list.   
[H5Pset_virtual_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819 "Sets prefix to be applied to VDS source file paths.")/[H5Pget_virtual_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995 "Retrieves prefix applied to VDS source file paths.") | Sets/gets the prefix to be applied to VDS source file paths.   
[H5Pset_virtual_printf_gap](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf "Sets the maximum number of missing source files and/or datasets with the printf-style names when gett...")/[H5Pget_virtual_printf_gap](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f "Returns the maximum number of missing source files and/or datasets with the printf-style names when g...") | Sets/gets the maximum number of missing source files and/or datasets with the printf-style names when getting the extent for an unlimited virtual dataset.   
[H5Pset_virtual_view](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4 "Sets the view of the virtual dataset \(VDS\) to include or exclude missing mapped elements.")/[H5Pget_virtual_view](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c "Retrieves the view of a virtual dataset accessed with dapl_id.") | Sets/gets the view of the virtual dataset (VDS) to include or exclude missing mapped elements.   
[H5Pset_virtual_spatial_tree](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga5a83d2c1075cc5ff47460b5526ec23ff "Sets the flag to use a spatial tree for mappings.")/[H5Pget_virtual_spatial_tree](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga39ac9ba6cc5c86c3dd8c46bae0ee17e9 "Retrieves the setting for whether or not to use a spatial tree for VDS mappings.") | Sets/gets the flag to use spatial trees when searching many VDS mappings   
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_append_flush](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gacd6803640eebd20e408c330192b09fa6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, unsigned dims, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) boundary[], [H5D_append_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#af74a89b6cff26752aade055f0e913718) *func, void **udata)  
| Retrieves the values of the append property that is set up in the dataset access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_chunk_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gaeda015dfee4167cc60baab1d1f0560fe) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, size_t *rdcc_nslots, size_t *rdcc_nbytes, double *rdcc_w0)  
| Retrieves the raw data chunk cache parameters.   
  
ssize_t  |  [H5Pget_efile_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga442647d48171db920c71a7baf6fdeee6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, char *[prefix](https://support.hdfgroup.org/documentation/hdf5/latest/h5dump_8h.html#ad2849cf781a4db22cc1b31eaaee50a4f), size_t size)  
| Retrieves the prefix for external raw data storage files as set in the dataset access property list.   
  
ssize_t  |  [H5Pget_virtual_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, char *[prefix](https://support.hdfgroup.org/documentation/hdf5/latest/h5dump_8h.html#ad2849cf781a4db22cc1b31eaaee50a4f), size_t size)  
| Retrieves prefix applied to VDS source file paths.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_virtual_printf_gap](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *gap_size)  
| Returns the maximum number of missing source files and/or datasets with the printf-style names when getting the extent for an unlimited virtual dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_virtual_view](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [H5D_vds_view_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a13217d859a956fababf5b139ac64b8a0) *view)  
| Retrieves the view of a virtual dataset accessed with `dapl_id`.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_append_flush](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga2f685a7b3f3a4fa35ddcd1659ab4a835) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, unsigned ndims, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) boundary[], [H5D_append_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#af74a89b6cff26752aade055f0e913718) func, void *udata)  
| Sets two actions to perform when the size of a dataset's dimension being appended reaches a specified boundary.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_chunk_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, size_t rdcc_nslots, size_t rdcc_nbytes, double rdcc_w0)  
| Sets the raw data chunk cache parameters.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_efile_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, const char *[prefix](https://support.hdfgroup.org/documentation/hdf5/latest/h5dump_8h.html#ad2849cf781a4db22cc1b31eaaee50a4f))  
| Sets the external dataset storage file prefix in the dataset access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_virtual_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, const char *[prefix](https://support.hdfgroup.org/documentation/hdf5/latest/h5dump_8h.html#ad2849cf781a4db22cc1b31eaaee50a4f))  
| Sets prefix to be applied to VDS source file paths.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_virtual_printf_gap](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) gap_size)  
| Sets the maximum number of missing source files and/or datasets with the printf-style names when getting the extent of an unlimited virtual dataset.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_virtual_view](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [H5D_vds_view_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a13217d859a956fababf5b139ac64b8a0) view)  
| Sets the view of the virtual dataset (VDS) to include or exclude missing mapped elements.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gacd6803640eebd20e408c330192b09fa6)H5Pget_append_flush()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_append_flush  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | unsigned |  _dims_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _boundary_[],   
|  |  [H5D_append_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#af74a89b6cff26752aade055f0e913718) * |  _func_ ,   
|  | void ** |  _udata_ )  
Retrieves the values of the append property that is set up in the dataset access property list.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[in] | dims | The number of elements for `boundary`  
[out] | boundary | The dimension sizes used to determine the boundary   
[out] | func | The user-defined callback function   
[out] | udata | The user-defined input data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_append_flush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gacd6803640eebd20e408c330192b09fa6 "Retrieves the values of the append property that is set up in the dataset access property list.") obtains the following information from the dataset access property list, `dapl_id`.
`boundary` consists of the sizes set up in the access property list that are used to determine when a dataset dimension size hits the boundary. Only at most `dims` boundary sizes are retrieved, and `dims` will not exceed the corresponding value that is set in the property list.
`func` is the user-defined callback function to invoke when a dataset's appended dimension size reaches a boundary and `udata` is the user-defined input data for the callback function. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gaeda015dfee4167cc60baab1d1f0560fe)H5Pget_chunk_cache()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_chunk_cache  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | size_t * |  _rdcc_nslots_ ,   
|  | size_t * |  _rdcc_nbytes_ ,   
|  | double * |  _rdcc_w0_ )  
Retrieves the raw data chunk cache parameters.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[out] | rdcc_nslots | Number of chunk slots in the raw data chunk cache hash table   
[out] | rdcc_nbytes | Total size of the raw data chunk cache, in bytes   
[out] | rdcc_w0 | Preemption policy 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gaeda015dfee4167cc60baab1d1f0560fe "Retrieves the raw data chunk cache parameters.") retrieves the number of chunk slots in the raw data chunk cache hash table, the maximum possible number of bytes in the raw data chunk cache, and the preemption policy value.
These values are retrieved from a dataset access property list. If the values have not been set on the property list, then values returned will be the corresponding values from a default file access property list.
Any (or all) pointer arguments may be null pointers, in which case the corresponding data is not returned. 

Since
    1.8.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga442647d48171db920c71a7baf6fdeee6)H5Pget_efile_prefix()
ssize_t H5Pget_efile_prefix  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | char * |  _prefix_ ,   
|  | size_t |  _size_ )  
Retrieves the prefix for external raw data storage files as set in the dataset access property list.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[in,out] | prefix | Dataset external storage prefix in UTF-8 or ASCII (_Path_ and _filename_ must be ASCII on Windows systems.)   
[in] | size | Size of prefix buffer in bytes 

Returns
    Returns the size of `prefix` and the prefix string will be stored in `prefix` if successful. Otherwise returns a negative value and the contents of `prefix` will be undefined.  
[H5Pget_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga442647d48171db920c71a7baf6fdeee6 "Retrieves the prefix for external raw data storage files as set in the dataset access property list.") retrieves the file system path prefix for locating external files associated with a dataset that uses external storage. This will be the value set with [H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.") or the HDF5 library's default.
The value of `size` is the size in bytes of the prefix, including the NULL terminator. If the size is unknown, a preliminary [H5Pget_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga7960f746797bcf35f70746cd644f8b5a "Retrieves prefix applied to external link paths.") call with the pointer `prefix` set to NULL will return the size of the prefix without the NULL terminator.
The `prefix` buffer must be allocated by the caller. In a call that retrieves the actual prefix, that buffer must be of the size specified in `size`. 

Note
    See [H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.") for a more complete description of file location behavior and for notes on the use of the HDF5_EXTFILE_PREFIX environment variable. 

Since
    1.8.17 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995)H5Pget_virtual_prefix()
ssize_t H5Pget_virtual_prefix  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | char * |  _prefix_ ,   
|  | size_t |  _size_ )  
Retrieves prefix applied to VDS source file paths.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[out] | prefix | Prefix applied to VDS source file paths   
[in] | size | Size of prefix, including null terminator 

Returns
    If successful, returns a non-negative value specifying the size in bytes of the prefix without the NULL terminator; otherwise returns a negative value.  
[H5Pget_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995 "Retrieves prefix applied to VDS source file paths.") retrieves the prefix applied to the path of any VDS source files traversed.
When an VDS source file is traversed, the prefix is retrieved from the dataset access property list `dapl_id`, returned in the user-allocated buffer pointed to by `prefix`, and prepended to the filename stored in the VDS virtual file, set with [H5Pset_virtual()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5 "Sets the mapping between virtual and source datasets.").
The size in bytes of the prefix, including the NULL terminator, is specified in `size`. If `size` is unknown, a preliminary [H5Pget_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995 "Retrieves prefix applied to VDS source file paths.") call with the pointer `prefix` set to NULL will return the size of the prefix without the NULL terminator. 

See also
    Supporting Functions: [H5Pget_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga655530b0f40990507fedeef6b3068db3 "Returns the layout of the raw data for a dataset."), [H5Pset_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9 "Sets the type of storage used to store the raw data for a dataset."), [H5Sget_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabc974bbc041538a1d3032729df2ddfc0 "Retrieves a regular hyperslab selection."), [H5Sis_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8a5bc33fae4be442093329f2cfec3f49 "Determines if a hyperslab selection is regular."), [H5Sselect_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.")      VDS Functions: [H5Pget_virtual_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga83dcce1ce110d1ff6eae0fb77d4a7c85 "Gets the number of mappings for the virtual dataset."), [H5Pget_virtual_dsetname()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf50620fd5d83dc9ca1e5c3f374c5a952 "Gets the name of a source dataset used in the mapping."), [H5Pget_virtual_filename()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga5c17780cc9a72a0f62d70f6138510afa "Gets the filename of a source dataset used in the mapping."), [H5Pget_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995 "Retrieves prefix applied to VDS source file paths."), [H5Pget_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f "Returns the maximum number of missing source files and/or datasets with the printf-style names when g..."), [H5Pget_virtual_srcspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga8319e9386cdb9b3881a8b698edfc78fc "Gets a dataspace identifier for the selection within the source dataset used in the mapping."), [H5Pget_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c "Retrieves the view of a virtual dataset accessed with dapl_id."), [H5Pget_virtual_vspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6425cabbc055b66e218b4728d6eb911d "Gets a dataspace identifier for the selection within the virtual dataset used in the mapping."), [H5Pset_virtual()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5 "Sets the mapping between virtual and source datasets."), [H5Pset_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819 "Sets prefix to be applied to VDS source file paths."), [H5Pset_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf "Sets the maximum number of missing source files and/or datasets with the printf-style names when gett..."), [H5Pset_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4 "Sets the view of the virtual dataset \(VDS\) to include or exclude missing mapped elements.") 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f)H5Pget_virtual_printf_gap()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_virtual_printf_gap  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _gap_size_ )  
Returns the maximum number of missing source files and/or datasets with the printf-style names when getting the extent for an unlimited virtual dataset.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[out] | gap_size | Maximum number of the files and/or datasets allowed to be missing for determining the extent of an unlimited virtual dataset with printf-style mappings. (_Default:_ 0) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f "Returns the maximum number of missing source files and/or datasets with the printf-style names when g...") returns the maximum number of missing printf-style files and/or datasets for determining the extent of an unlimited virtual dataaset, `gap_size`, using the access property list for the virtual dataset, `dapl_id`.
The default library value for `gap_size` is 0 (zero). 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c)H5Pget_virtual_view()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_virtual_view  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  |  [H5D_vds_view_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a13217d859a956fababf5b139ac64b8a0) * |  _view_ )  
Retrieves the view of a virtual dataset accessed with `dapl_id`.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[out] | view | The flag specifying the view of the virtual dataset. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c "Retrieves the view of a virtual dataset accessed with dapl_id.") takes the virtual dataset access property list, `dapl_id`, and retrieves the flag, `view`, set by the [H5Pset_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4 "Sets the view of the virtual dataset \(VDS\) to include or exclude missing mapped elements.") call. 

See also
    Supporting Functions: [H5Pget_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga655530b0f40990507fedeef6b3068db3 "Returns the layout of the raw data for a dataset."), [H5Pset_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9 "Sets the type of storage used to store the raw data for a dataset."), [H5Sget_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabc974bbc041538a1d3032729df2ddfc0 "Retrieves a regular hyperslab selection."), [H5Sis_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8a5bc33fae4be442093329f2cfec3f49 "Determines if a hyperslab selection is regular."), [H5Sselect_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.")      VDS Functions: [H5Pget_virtual_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga83dcce1ce110d1ff6eae0fb77d4a7c85 "Gets the number of mappings for the virtual dataset."), [H5Pget_virtual_dsetname()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf50620fd5d83dc9ca1e5c3f374c5a952 "Gets the name of a source dataset used in the mapping."), [H5Pget_virtual_filename()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga5c17780cc9a72a0f62d70f6138510afa "Gets the filename of a source dataset used in the mapping."), [H5Pget_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995 "Retrieves prefix applied to VDS source file paths."), [H5Pget_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f "Returns the maximum number of missing source files and/or datasets with the printf-style names when g..."), [H5Pget_virtual_srcspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga8319e9386cdb9b3881a8b698edfc78fc "Gets a dataspace identifier for the selection within the source dataset used in the mapping."), [H5Pget_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c "Retrieves the view of a virtual dataset accessed with dapl_id."), [H5Pget_virtual_vspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6425cabbc055b66e218b4728d6eb911d "Gets a dataspace identifier for the selection within the virtual dataset used in the mapping."), [H5Pset_virtual()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5 "Sets the mapping between virtual and source datasets."), [H5Pset_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819 "Sets prefix to be applied to VDS source file paths."), [H5Pset_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf "Sets the maximum number of missing source files and/or datasets with the printf-style names when gett..."), [H5Pset_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4 "Sets the view of the virtual dataset \(VDS\) to include or exclude missing mapped elements.") 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga2f685a7b3f3a4fa35ddcd1659ab4a835)H5Pset_append_flush()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_append_flush  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | unsigned |  _ndims_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _boundary_[],   
|  | [H5D_append_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#af74a89b6cff26752aade055f0e913718) |  _func_ ,   
|  | void * |  _udata_ )  
Sets two actions to perform when the size of a dataset's dimension being appended reaches a specified boundary.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[in] | ndims | The number of elements for boundary   
[in] | boundary | The dimension sizes used to determine the boundary   
[in] | func | The user-defined callback function   
[in] | udata | The user-defined input data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_append_flush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga2f685a7b3f3a4fa35ddcd1659ab4a835 "Sets two actions to perform when the size of a dataset's dimension being appended reaches a specified...") sets the following two actions to perform for a dataset associated with the dataset access property list `dapl_id:`
  * Call the callback function `func` set in the property list 
  * Flush the dataset associated with the dataset access property list


When a user is appending data to a dataset via [H5DOappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga1fcbd6fc898ccb603e85e75f9507a76c "Appends data to a dataset along a specified dimension.") and the dataset's newly extended dimension size hits a specified boundary, the library will perform the first action listed above. Upon return from the callback function, the library will then perform the second action listed above and return to the user. If no boundary is hit or set, the two actions above are not invoked.
The specified boundary is indicated by the parameter `boundary`. It is a 1-dimensional array with `ndims` elements, which should be the same as the rank of the dataset's dataspace. While appending to a dataset along a particular dimension index via H5Dappend(), the library determines a boundary is reached when the resulting dimension size is divisible by `boundary`[index]. A zero value for `boundary`[index] indicates no boundary is set for that dimension index.
The setting of this property will apply only for a chunked dataset with an extendible dataspace. A dataspace is extendible when it is defined with either one of the following:
  * A dataspace with fixed current and maximum dimension sizes 
  * A dataspace with at least one unlimited dimension for its maximum dimension size


When creating or opening a chunked dataset, the library will check whether the boundary as specified in the access property list is set up properly. The library will fail the dataset create or open if the following conditions are true:
  * `ndims`, the number of elements for boundary, is not the same as the rank of the dataset's dataspace. 
  * A non-zero boundary value is specified for a non-extendible dimension.


The callback function `func` must conform to the following prototype: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5D_append_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#af74a89b6cff26752aade055f0e913718))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dataset_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *cur_dims, void *op_data);
The parameters of the callback function, per the above prototype, are defined as follows:
  * `dataset_id` is the dataset identifier. 
  * `cur_dims` is the dataset's current dimension sizes when a boundary is hit. 
  * `user_data` is the user-defined input data.



Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1)H5Pset_chunk_cache()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_chunk_cache  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | size_t |  _rdcc_nslots_ ,   
|  | size_t |  _rdcc_nbytes_ ,   
|  | double |  _rdcc_w0_ )  
Sets the raw data chunk cache parameters.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[in] | rdcc_nslots | The number of chunk slots in the raw data chunk cache for this dataset. Increasing this value reduces the number of cache collisions, but slightly increases the memory used. Due to the hashing strategy, this value should ideally be a prime number. As a rule of thumb, this value should be at least 10 times the number of chunks that can fit in `rdcc_nbytes` bytes. For maximum performance, this value should be set approximately 100 times that number of chunks. The default value is 8191. If the value passed is [H5D_CHUNK_CACHE_NSLOTS_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa74c0dd5694bcc907ea627a45cf2cb08), then the property will not be set on `dapl_id` and the parameter will come from the file access property list used to open the file.   
[in] | rdcc_nbytes | The total size of the raw data chunk cache for this dataset. In most cases increasing this number will improve performance, as long as you have enough free memory. The default size is 8 MiB. If the value passed is [H5D_CHUNK_CACHE_NBYTES_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a00636a76dcac774da4cd79d32f6ddf49), then the property will not be set on `dapl_id` and the parameter will come from the file access property list.   
[in] | rdcc_w0 | The chunk preemption policy for this dataset. This must be between 0 and 1 inclusive and indicates the weighting according to which chunks which have been fully read or written are penalized when determining which chunks to flush from cache. A value of 0 means fully read or written chunks are treated no differently than other chunks (the preemption is strictly LRU) while a value of 1 means fully read or written chunks are always preempted before other chunks. If your application only reads or writes data once, this can be safely set to 1. Otherwise, this should be set lower, depending on how often you re-read or re-write the same data. The default value is 0.75. If the value passed is [H5D_CHUNK_CACHE_W0_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a4a01949e0020fa9471811cc8ed271ee7), then the property will not be set on `dapl_id` and the parameter will come from the file access property list. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters.") sets the number of elements, the total number of bytes, and the preemption policy value in the raw data chunk cache on a dataset access property list. After calling this function, the values set in the property list will override the values in the file's file access property list.
The raw data chunk cache inserts chunks into the cache by first computing a hash value using the address of a chunk, then using that hash value as the chunk's index into the table of cached chunks. The size of this hash table, i.e., and the number of possible hash values, is determined by the `rdcc_nslots` parameter. If a different chunk in the cache has the same hash value, this causes a collision, which reduces efficiency. If inserting the chunk into cache would cause the cache to be too big, then the cache is pruned according to the `rdcc_w0` parameter.
**Motivation:** [H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters.") is used to adjust the chunk cache parameters on a per-dataset basis, as opposed to a global setting for the file using [H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters."). The optimum chunk cache parameters may vary widely with different data layout and access patterns, so for optimal performance they must be set individually for each dataset. It may also be beneficial to reduce the size of the chunk cache for datasets whose performance is not important in order to save memory space.
**Example** **Usage:** The following code sets the chunk cache to use a hash table with 12421 elements and a maximum size of 16 MiB, while using the preemption policy specified for the entire file: ` H5Pset_chunk_cache(dapl_id, 12421, 16*1024*1024,          H5D_CHUNK_CACHE_W0_DEFAULT);`
**Usage** **Notes:** The chunk cache size is a property for accessing a dataset and is not stored with a dataset or a file. To guarantee the same chunk cache settings each time the dataset is opened, call [H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) with a dataset access property list where the chunk cache size is set by calling [H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters.") for that property list. The property list can be used for multiple accesses in the same application.
For files where the same chunk cache size will be appropriate for all or most datasets, [H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.") can be called with a file access property list to set the chunk cache size for accessing all datasets in the file.
Both methods can be used in combination, in which case the chunk cache size set by [H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.") will apply except for specific datasets where [H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) is called with dataset property list with the chunk cache size set by [H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters.").
In the absence of any cache settings, [H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) will by default create an 8 MiB chunk cache for the opened dataset. If this size happens to be appropriate, no call will be needed to either function to set the chunk cache size.
It is also possible that a change in access pattern for later access to a dataset will change the appropriate chunk cache size. 

Since
    1.8.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8)H5Pset_efile_prefix()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_efile_prefix  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | const char * |  _prefix_ )  
Sets the external dataset storage file prefix in the dataset access property list.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[in] | prefix | Dataset external storage prefix in UTF-8 or ASCII (_Path and filename must be ASCII on Windows systems._) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.") sets the prefix used to locate raw data files for a dataset that uses external storage. This prefix can provide either an absolute path or a relative path to the external files.
[H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.") is used in conjunction with [H5Pset_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga64bf68bda464eaaabfe24846533b1935 "Adds an external file to the list of external files.") to control the behavior of the HDF5 library when searching for the raw data files associated with a dataset that uses external storage:
  * The default behavior of the library is to search for the dataset's external storage raw data files in the current working directory of the program. 
  * If the prefix is set to an absolute path, the target directory will be searched for the dataset's external storage raw data files. 
  * If the prefix is set to a relative path, the target directory, relative to the current working directory, will be searched for the dataset's external storage raw data files. 
  * If the prefix is set to a relative path that begins with the special token ${ORIGIN}, that directory, relative to the HDF5 file containing the dataset, will be searched for the dataset's external storage raw data files.


The HDF5_EXTFILE_PREFIX environment variable can be used to override the above behavior (the environment variable supersedes the API call). Setting the variable to a path string and calling [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) or [H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) is the equivalent of calling [H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.") and calling the same create or open function. The environment variable is checked at the time of the create or open action and copied so it can be safely changed after the [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) or [H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) call.
Calling [H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.") with `prefix` set to NULL or the empty string returns the search path to the default. The result would be the same as if [H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.") had never been called. 

Note
    If the external file prefix is not an absolute path and the HDF5 file is moved, the external storage files will also need to be moved so they can be accessed at the new location.      As stated above, the use of the HDF5_EXTFILE_PREFIX environment variable overrides any property list setting. [H5Pset_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gad487f84157fd0944cbe1cbd4dea4e1b8 "Sets the external dataset storage file prefix in the dataset access property list.") and [H5Pget_efile_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga442647d48171db920c71a7baf6fdeee6 "Retrieves the prefix for external raw data storage files as set in the dataset access property list."), being property functions, set and retrieve only the property list setting; they are unaware of the environment variable.      On Windows, the prefix must be an ASCII string since the Windows standard C library's I/O functions cannot handle UTF-8 file names. 

Since
    1.8.17 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819)H5Pset_virtual_prefix()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_virtual_prefix  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | const char * |  _prefix_ )  
Sets prefix to be applied to VDS source file paths.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[in] | prefix | Prefix to be applied to VDS source file paths 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819 "Sets prefix to be applied to VDS source file paths.") sets the prefix to be applied to the path of any VDS source files traversed. The prefix is prepended to the filename stored in the VDS virtual file, set with [H5Pset_virtual()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5 "Sets the mapping between virtual and source datasets.").
The prefix is specified in the user-allocated buffer `prefix` and set in the dataset access property list `dapl_id`. The buffer should not be freed until the property list has been closed. 

See also
    Supporting Functions: [H5Pget_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga655530b0f40990507fedeef6b3068db3 "Returns the layout of the raw data for a dataset."), [H5Pset_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9 "Sets the type of storage used to store the raw data for a dataset."), [H5Sget_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabc974bbc041538a1d3032729df2ddfc0 "Retrieves a regular hyperslab selection."), [H5Sis_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8a5bc33fae4be442093329f2cfec3f49 "Determines if a hyperslab selection is regular."), [H5Sselect_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.")      VDS Functions: [H5Pget_virtual_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga83dcce1ce110d1ff6eae0fb77d4a7c85 "Gets the number of mappings for the virtual dataset."), [H5Pget_virtual_dsetname()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf50620fd5d83dc9ca1e5c3f374c5a952 "Gets the name of a source dataset used in the mapping."), [H5Pget_virtual_filename()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga5c17780cc9a72a0f62d70f6138510afa "Gets the filename of a source dataset used in the mapping."), [H5Pget_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995 "Retrieves prefix applied to VDS source file paths."), [H5Pget_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f "Returns the maximum number of missing source files and/or datasets with the printf-style names when g..."), [H5Pget_virtual_srcspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga8319e9386cdb9b3881a8b698edfc78fc "Gets a dataspace identifier for the selection within the source dataset used in the mapping."), [H5Pget_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c "Retrieves the view of a virtual dataset accessed with dapl_id."), [H5Pget_virtual_vspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6425cabbc055b66e218b4728d6eb911d "Gets a dataspace identifier for the selection within the virtual dataset used in the mapping."), [H5Pset_virtual()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5 "Sets the mapping between virtual and source datasets."), [H5Pset_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819 "Sets prefix to be applied to VDS source file paths."), [H5Pset_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf "Sets the maximum number of missing source files and/or datasets with the printf-style names when gett..."), [H5Pset_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4 "Sets the view of the virtual dataset \(VDS\) to include or exclude missing mapped elements.") 

Since
    1.10.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf)H5Pset_virtual_printf_gap()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_virtual_printf_gap  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _gap_size_ )  
Sets the maximum number of missing source files and/or datasets with the printf-style names when getting the extent of an unlimited virtual dataset.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[in] | gap_size | Maximum number of files and/or datasets allowed to be missing for determining the extent of an unlimited virtual dataset with printf-style mappings (_Default value_ : 0) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf "Sets the maximum number of missing source files and/or datasets with the printf-style names when gett...") sets the access property list for the virtual dataset, `dapl_id`, to instruct the library to stop looking for the mapped data stored in the files and/or datasets with the printf-style names after not finding `gap_size` files and/or datasets. The found source files and datasets will determine the extent of the unlimited virtual dataset with the printf-style mappings.
Consider the following examples where the regularly spaced blocks of a virtual dataset are mapped to datasets with the names d-1, d-2, d-3, ..., d-N, ... :
  * If the dataset d-2 is missing and `gap_size` is set to 0, then the virtual dataset will contain only data found in d-1. 
  * If d-2 and d-3 are missing and `gap_size` is set to 2, then the virtual dataset will contain the data from d-1, d-3, ..., d-N, ... . The blocks that are mapped to d-2 and d-3 will be filled according to the virtual dataset's fill value setting.



See also
    Supporting Functions: [H5Pget_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga655530b0f40990507fedeef6b3068db3 "Returns the layout of the raw data for a dataset."), [H5Pset_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9 "Sets the type of storage used to store the raw data for a dataset."), [H5Sget_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabc974bbc041538a1d3032729df2ddfc0 "Retrieves a regular hyperslab selection."), [H5Sis_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8a5bc33fae4be442093329f2cfec3f49 "Determines if a hyperslab selection is regular."), [H5Sselect_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.")      VDS Functions: [H5Pget_virtual_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga83dcce1ce110d1ff6eae0fb77d4a7c85 "Gets the number of mappings for the virtual dataset."), [H5Pget_virtual_dsetname()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf50620fd5d83dc9ca1e5c3f374c5a952 "Gets the name of a source dataset used in the mapping."), [H5Pget_virtual_filename()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga5c17780cc9a72a0f62d70f6138510afa "Gets the filename of a source dataset used in the mapping."), [H5Pget_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995 "Retrieves prefix applied to VDS source file paths."), [H5Pget_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f "Returns the maximum number of missing source files and/or datasets with the printf-style names when g..."), [H5Pget_virtual_srcspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga8319e9386cdb9b3881a8b698edfc78fc "Gets a dataspace identifier for the selection within the source dataset used in the mapping."), [H5Pget_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c "Retrieves the view of a virtual dataset accessed with dapl_id."), [H5Pget_virtual_vspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6425cabbc055b66e218b4728d6eb911d "Gets a dataspace identifier for the selection within the virtual dataset used in the mapping."), [H5Pset_virtual()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5 "Sets the mapping between virtual and source datasets."), [H5Pset_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819 "Sets prefix to be applied to VDS source file paths."), [H5Pset_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf "Sets the maximum number of missing source files and/or datasets with the printf-style names when gett..."), [H5Pset_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4 "Sets the view of the virtual dataset \(VDS\) to include or exclude missing mapped elements.") 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4)H5Pset_virtual_view()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_virtual_view  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
---|---|---|---  
|  | [H5D_vds_view_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a13217d859a956fababf5b139ac64b8a0) |  _view_ )  
Sets the view of the virtual dataset (VDS) to include or exclude missing mapped elements.  

Parameters
     [in] | dapl_id | Dataset access property list identifier   
---|---|---  
[in] | view | Flag specifying the extent of the data to be included in the view. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4 "Sets the view of the virtual dataset \(VDS\) to include or exclude missing mapped elements.") takes the access property list for the virtual dataset, `dapl_id`, and the flag, `view`, and sets the VDS view according to the flag value.
If `view` is set to [H5D_VDS_FIRST_MISSING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a13217d859a956fababf5b139ac64b8a0af11d58c657429ca942108c4a54671bd1), the view includes all data before the first missing mapped data. This setting provides a view containing only the continuous data starting with the dataset's first data element. Any break in continuity terminates the view.
If `view` is set to [H5D_VDS_LAST_AVAILABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a13217d859a956fababf5b139ac64b8a0aa92ec0bc9e045a435ca2206018bc2211), the view includes all available mapped data.
Missing mapped data is filled with the fill value set in the VDS creation property list. 

See also
    Supporting Functions: [H5Pget_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga655530b0f40990507fedeef6b3068db3 "Returns the layout of the raw data for a dataset."), [H5Pset_layout()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9 "Sets the type of storage used to store the raw data for a dataset."), [H5Sget_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabc974bbc041538a1d3032729df2ddfc0 "Retrieves a regular hyperslab selection."), [H5Sis_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8a5bc33fae4be442093329f2cfec3f49 "Determines if a hyperslab selection is regular."), [H5Sselect_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.")      VDS Functions: [H5Pget_virtual_count()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga83dcce1ce110d1ff6eae0fb77d4a7c85 "Gets the number of mappings for the virtual dataset."), [H5Pget_virtual_dsetname()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf50620fd5d83dc9ca1e5c3f374c5a952 "Gets the name of a source dataset used in the mapping."), [H5Pget_virtual_filename()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga5c17780cc9a72a0f62d70f6138510afa "Gets the filename of a source dataset used in the mapping."), [H5Pget_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga9a48c80955877c20d53e8fd3f49a2995 "Retrieves prefix applied to VDS source file paths."), [H5Pget_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga833dfc6d9c87738c9d94b610e70a818f "Returns the maximum number of missing source files and/or datasets with the printf-style names when g..."), [H5Pget_virtual_srcspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga8319e9386cdb9b3881a8b698edfc78fc "Gets a dataspace identifier for the selection within the source dataset used in the mapping."), [H5Pget_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga7173663654b085e8583ab609c988b47c "Retrieves the view of a virtual dataset accessed with dapl_id."), [H5Pget_virtual_vspace()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6425cabbc055b66e218b4728d6eb911d "Gets a dataspace identifier for the selection within the virtual dataset used in the mapping."), [H5Pset_virtual()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gadec895092dbbedb94f85d9cacf8924f5 "Sets the mapping between virtual and source datasets."), [H5Pset_virtual_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga6816e0de35a335f636922c3cd5569819 "Sets prefix to be applied to VDS source file paths."), [H5Pset_virtual_printf_gap()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga8bb25e402e860133b8af3715e429bacf "Sets the maximum number of missing source files and/or datasets with the printf-style names when gett..."), [H5Pset_virtual_view()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#gac65520e7cd7748f93d94c4a42abd01b4 "Sets the view of the virtual dataset \(VDS\) to include or exclude missing mapped elements.") 

Since
    1.10.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


