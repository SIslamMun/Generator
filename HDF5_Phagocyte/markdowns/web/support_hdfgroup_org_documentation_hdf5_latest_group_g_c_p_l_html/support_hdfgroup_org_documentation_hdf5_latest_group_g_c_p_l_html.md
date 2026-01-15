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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title0 "H2")
  * [ Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title2 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title3 "H2")
  * [◆ H5Pget_est_link_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title4 "H2")
  * [◆ H5Pget_link_creation_order()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title5 "H2")
  * [◆ H5Pget_link_phase_change()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title6 "H2")
  * [◆ H5Pget_local_heap_size_hint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title7 "H2")
  * [◆ H5Pset_est_link_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title8 "H2")
  * [◆ H5Pset_link_creation_order()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title9 "H2")
  * [◆ H5Pset_link_phase_change()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title10 "H2")
  * [◆ H5Pset_local_heap_size_hint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#groups) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#func-members)
Group Creation Properties
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) » [Object Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html)
## Detailed Description
Use group creation properties to control aspects of group creation such as storage layout, compression, and link creation order tracking. Unlike file access properties, creation properties _are_ stored with the group, and cannot be changed once a group has been created. 
Group creation property list functions (H5P) Function  | Purpose   
---|---  
[H5Pall_filters_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga70f5346250698afc950532e9593c3988 "Verifies that all required filters are available.") | Verifies that all required filters are available.   
[H5Pget_filter](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7e070dfec9cb3a3aaf9c188a987e6a15) | Returns information about a filter in a pipeline. The C function is a macro:  

See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html).   
[H5Pget_filter_by_id](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac7aa336e7b1b9033cea2448ba623951f) | Returns information about the specified filter. The C function is a macro:  

See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html).   
[H5Pget_nfilters](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gacbad1ca36a61246b439a25f28e7575fb "Returns the number of filters in the pipeline.") | Returns the number of filters in the pipeline.   
[H5Pmodify_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga12a358b3725a889c1768bbd2b5f541d8 "Modifies a filter in the filter pipeline.") | Modifies a filter in the filter pipeline.   
[H5Premove_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gabffbf6d013c090fa052ac4bafce8e532 "Delete one or more filters in the filter pipeline.") | Deletes one or more filters in the filter pipeline.   
[H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d "Sets deflate \(GNU gzip\) compression method and compression level.") | Sets the deflate (GNU gzip) compression method and compression level.   
[H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline.") | Adds a filter to the filter pipeline.   
[H5Pset_fletcher32](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga8bc81abfbd0393b0a46e121f817a3f81 "Sets up use of the Fletcher32 checksum filter.") | Sets up use of the Fletcher32 checksum filter.   
[H5Pset_local_heap_size_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga870728af2bf3c0b16edafd762a1c44d6 "Specifies the anticipated maximum size of a local heap.")/[H5Pget_local_heap_size_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga49e14718767fa160248e3852c2abdd74 "Retrieves the anticipated size of the local heap for original-style groups.") | Sets/gets the anticipated maximum size of a local heap.   
[H5Pset_link_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gab463ac9355728469eddfd973b4a5964f "Sets the parameters for conversion between compact and dense groups.") | Sets the parameters for conversion between compact and dense groups.   
[H5Pget_link_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gacab66461dca6c2beafd624c2e4d9f94d "Queries the settings for conversion between compact and dense groups.") | Queries the settings for conversion between compact and dense groups.   
[H5Pset_est_link_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa8571642d45e73ab5a9ae71cf00501f9 "Sets estimated number of links and length of link names in a group.") | Sets estimated number of links and length of link names in a group.   
[H5Pget_est_link_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga701867215546a345dea7b8e9cf7a1b61 "Returns the estimated link count and average link name length in a group.") | Queries data required to estimate required local heap or object header size.   
[H5Pset_nlinks](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaa46c63c196a0cf5cd94dede039c030f4 "Sets maximum number of soft or user-defined link traversals.") | Sets maximum number of soft or user-defined link traversals.   
[H5Pget_nlinks](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga6bfa33fa9a77011cbdc06d0fbc907177 "Retrieves the maximum number of link traversals.") | Retrieves the maximum number of link traversals.   
[H5Pset_link_creation_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga24817b5c9553df3872de57c20bf11512 "Sets creation order tracking and indexing for links in a group.") | Sets creation order tracking and indexing for links in a group.   
[H5Pget_link_creation_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa2c2f433c7e65f694e0444e7f0ed2d33 "Queries whether link creation order is tracked and/or indexed in a group.") | Queries whether link creation order is tracked and/or indexed in a group.   
[H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.") | Sets the character encoding used to encode a string. Use to set ASCII or UTF-8 character encoding for object names.   
[H5Pget_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d "Retrieves the character encoding used to create a link or attribute name.") | Retrieves the character encoding used to create a string.   
##  Topics  
---  
| [File Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_est_link_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga701867215546a345dea7b8e9cf7a1b61) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned *est_num_entries, unsigned *est_name_len)  
| Returns the estimated link count and average link name length in a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_link_creation_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa2c2f433c7e65f694e0444e7f0ed2d33) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned *crt_order_flags)  
| Queries whether link creation order is tracked and/or indexed in a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_link_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gacab66461dca6c2beafd624c2e4d9f94d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned *max_compact, unsigned *min_dense)  
| Queries the settings for conversion between compact and dense groups.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_local_heap_size_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga49e14718767fa160248e3852c2abdd74) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, size_t *size_hint)  
| Retrieves the anticipated size of the local heap for original-style groups.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_est_link_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa8571642d45e73ab5a9ae71cf00501f9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned est_num_entries, unsigned est_name_len)  
| Sets estimated number of links and length of link names in a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_link_creation_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga24817b5c9553df3872de57c20bf11512) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned crt_order_flags)  
| Sets creation order tracking and indexing for links in a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_link_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gab463ac9355728469eddfd973b4a5964f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned max_compact, unsigned min_dense)  
| Sets the parameters for conversion between compact and dense groups.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_local_heap_size_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga870728af2bf3c0b16edafd762a1c44d6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, size_t size_hint)  
| Specifies the anticipated maximum size of a local heap.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga701867215546a345dea7b8e9cf7a1b61)H5Pget_est_link_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_est_link_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned * |  _est_num_entries_ ,   
|  | unsigned * |  _est_name_len_ )  
Returns the estimated link count and average link name length in a group.  

Parameters
     [in] | plist_id | Group creation property list identifier   
---|---|---  
[out] | est_num_entries | The estimated number of links in the group referenced by `plist_id`  
[out] | est_name_len | The estimated average length of line names in the group referenced by `plist_id` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_est_link_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga701867215546a345dea7b8e9cf7a1b61 "Returns the estimated link count and average link name length in a group.") retrieves two settings from the group creation property list `plist_id:` the estimated number of links that are expected to be inserted into a group created with the property list and the estimated average length of those link names.
The estimated number of links is returned in `est_num_entries`. The limit for `est_num_entries` is 64 K.
The estimated average length of the anticipated link names is returned in `est_name_len`. The limit for `est_name_len` is 64 K.
See [Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) for a discussion of the available types of HDF5 group structures. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa2c2f433c7e65f694e0444e7f0ed2d33)H5Pget_link_creation_order()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_link_creation_order  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned * |  _crt_order_flags_ )  
Queries whether link creation order is tracked and/or indexed in a group.  

Parameters
     [in] | plist_id | Group or file creation property list identifier   
---|---|---  
[out] | crt_order_flags | Creation order flag(s) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_link_creation_order()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa2c2f433c7e65f694e0444e7f0ed2d33 "Queries whether link creation order is tracked and/or indexed in a group.") queries the group or file creation property list, `plist_id`, and returns a flag indicating whether link creation order is tracked and/or indexed in a group.
See [H5Pset_link_creation_order()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga24817b5c9553df3872de57c20bf11512 "Sets creation order tracking and indexing for links in a group.") for a list of valid creation order flags, as passed in `crt_order_flags`, and their meanings. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gacab66461dca6c2beafd624c2e4d9f94d)H5Pget_link_phase_change()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_link_phase_change  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned * |  _max_compact_ ,   
|  | unsigned * |  _min_dense_ )  
Queries the settings for conversion between compact and dense groups.  

Parameters
     [in] | plist_id | Group creation property list identifier   
---|---|---  
[out] | max_compact | Maximum number of links for compact storage   
[out] | min_dense | Minimum number of links for dense storage 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_link_phase_change()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gacab66461dca6c2beafd624c2e4d9f94d "Queries the settings for conversion between compact and dense groups.") queries the maximum number of entries for a compact group and the minimum number of links to require before converting a group to a dense form.
In the compact format, links are stored as messages in the group's header. In the dense format, links are stored in a fractal heap and indexed with a version 2 B-tree.
`max_compact` is the maximum number of links to store as header messages in the group header before converting the group to the dense format. Groups that are in the compact format and exceed this number of links are automatically converted to the dense format.
`min_dense` is the minimum number of links to store in the dense format. Groups which are in dense format and in which the number of links falls below this number are automatically converted back to the compact format.
In the compact format, links are stored as messages in the group's header. In the dense format, links are stored in a fractal heap and indexed with a version 2 B-tree.
See [H5Pset_link_phase_change()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gab463ac9355728469eddfd973b4a5964f "Sets the parameters for conversion between compact and dense groups.") for a discussion of traditional, compact, and dense group storage. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga49e14718767fa160248e3852c2abdd74)H5Pget_local_heap_size_hint()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_local_heap_size_hint  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | size_t * |  _size_hint_ )  
Retrieves the anticipated size of the local heap for original-style groups.  

Parameters
     [in] | plist_id | Group creation property list identifier   
---|---|---  
[out] | size_hint | Anticipated size of local heap  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_local_heap_size_hint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga49e14718767fa160248e3852c2abdd74 "Retrieves the anticipated size of the local heap for original-style groups.") queries the group creation property list, `plist_id`, for the anticipated size of the local heap, `size_hint`, for original-style groups, i.e., for groups of the style used prior to HDF5 Release 1.8.0. See [H5Pset_local_heap_size_hint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga870728af2bf3c0b16edafd762a1c44d6 "Specifies the anticipated maximum size of a local heap.") for further discussion. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa8571642d45e73ab5a9ae71cf00501f9)H5Pset_est_link_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_est_link_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned |  _est_num_entries_ ,   
|  | unsigned |  _est_name_len_ )  
Sets estimated number of links and length of link names in a group.  

Parameters
     [in] | plist_id | Group creation property list identifier   
---|---|---  
[in] | est_num_entries | Estimated number of links to be inserted into group   
[in] | est_name_len | Estimated average length of link names  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_est_link_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa8571642d45e73ab5a9ae71cf00501f9 "Sets estimated number of links and length of link names in a group.") inserts two settings into the group creation property list plist_id: the estimated number of links that are expected to be inserted into a group created with the property list and the estimated average length of those link names.
The estimated number of links is passed in `est_num_entries`. The limit for `est_num_entries` is 64 K.
The estimated average length of the anticipated link names is passed in `est_name_len`. The limit for `est_name_len` is 64 K.
The values for these two settings are multiplied to compute the initial local heap size (for old-style groups, if the local heap size hint is not set) or the initial object header size for (new-style compact groups; see [Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html)). Accurately setting these parameters will help reduce wasted file space.
If a group is expected to have many links and to be stored in dense format, set `est_num_entries` to 0 (zero) for maximum efficiency. This will prevent the group from being created in the compact format.
See [Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) for a discussion of the available types of HDF5 group structures. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga24817b5c9553df3872de57c20bf11512)H5Pset_link_creation_order()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_link_creation_order  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned |  _crt_order_flags_ )  
Sets creation order tracking and indexing for links in a group.  

Parameters
     [in] | plist_id | Group or file creation property list identifier   
---|---|---  
[out] | crt_order_flags | Creation order flag(s) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_link_creation_order()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga24817b5c9553df3872de57c20bf11512 "Sets creation order tracking and indexing for links in a group.") sets flags for tracking and indexing links on creation order in groups created with the group (or file) creation property list `plist_id`.
`crt_order_flags` contains flags with the following meanings:
[H5P_CRT_ORDER_TRACKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa52f444ce2ba8bc5a062612f195e899f) | Link creation order is tracked but not necessarily indexed   
---|---  
[H5P_CRT_ORDER_INDEXED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#adfd355619b7da5792a16d7bc491f963d) | Link creation order is indexed (requires [H5P_CRT_ORDER_TRACKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa52f444ce2ba8bc5a062612f195e899f))   
The default behavior is that links are tracked and indexed by name, and link creation order is neither tracked nor indexed. The name is always the primary index for links in a group.
[H5Pset_link_creation_order()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga24817b5c9553df3872de57c20bf11512 "Sets creation order tracking and indexing for links in a group.") can be used to set link creation order tracking, or to set link creation order tracking and indexing.
If ([H5P_CRT_ORDER_TRACKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa52f444ce2ba8bc5a062612f195e899f) | [H5P_CRT_ORDER_INDEXED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#adfd355619b7da5792a16d7bc491f963d)) is specified for `crt_order_flags`, then links will be tracked and indexed by creation order. The creation order is added as a secondary index and enables faster queries and iterations by creation order.
If just [H5P_CRT_ORDER_TRACKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa52f444ce2ba8bc5a062612f195e899f) is specified for `crt_order_flags`, then links will be tracked by creation order, but not indexed by creation order. Queries and iterations by creation order will work but will be much slower for large groups than if [H5P_CRT_ORDER_INDEXED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#adfd355619b7da5792a16d7bc491f963d) had been included. 

Note
    If a creation order index is to be built, it must be specified in the group creation property list. HDF5 currently provides no mechanism to turn on link creation order tracking at group creation time and to build the index later. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gab463ac9355728469eddfd973b4a5964f)H5Pset_link_phase_change()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_link_phase_change  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned |  _max_compact_ ,   
|  | unsigned |  _min_dense_ )  
Sets the parameters for conversion between compact and dense groups.  

Parameters
     [in] | plist_id | Group creation property list identifier   
---|---|---  
[in] | max_compact | Maximum number of links for compact storage (_Default:_ 8)   
[in] | min_dense | Minimum number of links for dense storage (_Default:_ 6) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_link_phase_change()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gab463ac9355728469eddfd973b4a5964f "Sets the parameters for conversion between compact and dense groups.") sets the maximum number of entries for a compact group and the minimum number of links to allow before converting a dense group back to the compact format.
`max_compact` is the maximum number of links to store as header messages in the group header before converting the group to the dense format. Groups that are in compact format and which exceed this number of links are automatically converted to dense format.
`min_dense` is the minimum number of links to store in the dense format. Groups which are in dense format and in which the number of links falls below this threshold are automatically converted to compact format. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga870728af2bf3c0b16edafd762a1c44d6)H5Pset_local_heap_size_hint()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_local_heap_size_hint  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | size_t |  _size_hint_ )  
Specifies the anticipated maximum size of a local heap.  

Parameters
     [in] | plist_id | Group creation property list identifier   
---|---|---  
[in] | size_hint | Anticipated maximum size in bytes of local heap  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_local_heap_size_hint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga870728af2bf3c0b16edafd762a1c44d6 "Specifies the anticipated maximum size of a local heap.") is used with original-style HDF5 groups (see “Motivation” below) to specify the anticipated maximum local heap size, size_hint, for groups created with the group creation property list `plist_id`. The HDF5 library then uses `size_hint` to allocate contiguous local heap space in the file for each group created with `plist_id`.
For groups with many members or very few members, an appropriate initial value of `size_hint` would be the anticipated number of group members times the average length of group member names, plus a small margin: 
size_hint = max_number_of_group_members *
(average_length_of_group_member_link_names + 2)
If it is known that there will be groups with zero members, the use of a group creation property list with `size_hint` set to to 1 (one) will guarantee the smallest possible local heap for each of those groups.
Setting `size_hint` to zero (0) causes the library to make a reasonable estimate for the default local heap size. 

Motivation:
    In situations where backward-compatibility is required, specifically, when libraries prior to HDF5 Release 1.8.0 may be used to read the file, groups must be created and maintained in the original style. This is HDF5's default behavior. If backward compatibility with pre-1.8.0 libraries is not a concern, greater efficiencies can be obtained with the new-format compact and indexed groups. See the **Group implementations in HDF5:** in the [Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) API introduction (at the bottom).  
[H5Pset_local_heap_size_hint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga870728af2bf3c0b16edafd762a1c44d6 "Specifies the anticipated maximum size of a local heap.") is useful for tuning file size when files contain original-style groups with either zero members or very large numbers of members.  
The original style of HDF5 groups, the only style available prior to HDF5 Release 1.8.0, was well-suited for moderate-sized groups but was not optimized for either very small or very large groups. This original style remains the default, but two new group implementations were introduced in HDF5 Release 1.8.0: compact groups to accommodate zero to small numbers of members and indexed groups for thousands or tens of thousands of members ... or millions, if that's what your application requires.  
The local heap size hint, `size_hint`, is a performance tuning parameter for original-style groups. As indicated above, an HDF5 group may have zero, a handful, or tens of thousands of members. Since the original style of HDF5 groups stores the metadata for all of these group members in a uniform format in a local heap, the size of that metadata (and hence, the size of the local heap) can vary wildly from group to group. To intelligently allocate space and to avoid unnecessary fragmentation of the local heap, it can be valuable to provide the library with a hint as to the local heap's likely eventual size. This can be particularly valuable when it is known that a group will eventually have a great many members. It can also be useful in conserving space in a file when it is known that certain groups will never have any members. 

Since
    1.8.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


