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
  * [ The HDF5 Filter Interface](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title1 "H2")
  * [ Built-in Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title2 "H2")
  * [ Using Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title3 "H2")
  * [ Filter Pipelines](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title4 "H2")
  * [ Custom Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title5 "H2")
  * [ Querying Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title6 "H2")
  * [ Filter Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title7 "H2")
  * [ Summary](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_z__u_g.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Filters
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  The HDF5 Filter Interface
##  Introduction
The HDF5 Filter interface (H5Z) provides a flexible pipeline mechanism for processing dataset data during I/O operations. Filters can perform data compression, error checking, data transformation, and other custom operations on dataset chunks.
Filters operate on chunked datasets only (see [Using HDF5 Filters](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#subsec_dataset_filters) for details on dataset chunking) and are applied independently to each chunk. Multiple filters can be chained together in a pipeline, where the output of one filter becomes the input to the next.
##  Built-in Filters
HDF5 includes several standard filters:
  * **DEFLATE (gzip)** : General-purpose compression using the gzip algorithm ([H5Z_FILTER_DEFLATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a9e802e9612b3647e7d3ffe4ce3b8dcce)). Provides good compression ratios with moderate CPU usage.


  * **SZIP** : Compression algorithm designed for scientific data ([H5Z_FILTER_SZIP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a421d9941c68ebb776573baeb9aa77cd2)). Offers better performance than DEFLATE for certain data patterns.


  * **SHUFFLE** : Rearranges byte order to improve compression ([H5Z_FILTER_SHUFFLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#aa723f1a71601bf22c95620a490ecf1af)). Typically used before compression filters.


  * **FLETCHER32** : Computes and verifies checksums for error detection ([H5Z_FILTER_FLETCHER32](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a59ca894c9c2b99b1614b0c46a7407f1c)). Ensures data integrity.


  * **NBIT** : Lossless compression for datasets with unused bits ([H5Z_FILTER_NBIT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a8cc463fa1979bd4bfa0dd9aa6a41e49d)).


  * **SCALEOFFSET** : Lossy compression using scaling and offset ([H5Z_FILTER_SCALEOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a745d2ccb4f7712ed78ef5e562e27d2ca)).


##  Using Filters
Filters are configured through dataset creation property lists. Enable chunking first using [H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b "Sets the size of the chunks used to store a chunked layout dataset."), then add compression with functions like [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d "Sets deflate \(GNU gzip\) compression method and compression level.").
##  Filter Pipelines
Multiple filters can be combined in a pipeline. Filters are applied in the order they are added during write operations and in reverse order during read operations. Common pipelines combine [H5Pset_shuffle](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga31e09cb0bf2da2893eed8a72220e6521 "Sets up use of the shuffle filter."), [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d "Sets deflate \(GNU gzip\) compression method and compression level."), and [H5Pset_fletcher32](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga8bc81abfbd0393b0a46e121f817a3f81 "Sets up use of the Fletcher32 checksum filter.").
##  Custom Filters
Applications can create and register custom filters:
  * Define a filter function with signature matching [H5Z_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a130d8964a46667029c7d3c14572577c6 "The filter operation callback function, defining a filter's operation on data.")
  * Create a [H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) structure describing the filter 
  * Register the filter using [H5Zregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b "Registers a new filter with the HDF5 library.")
  * Apply the filter using [H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline.") with the filter ID


Custom filters enable domain-specific data transformations, specialized compression algorithms, encryption, and other custom processing.
##  Querying Filters
The H5Z interface provides functions to query available filters:
  * [H5Zfilter_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga3594e10d70739ccda55ebb55b17b50ee "Determines whether a filter is available.") checks if a filter is available 
  * [H5Zget_filter_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga9ef800ceec249c8819492545def9adba "Retrieves information about a filter.") retrieves information about a filter's capabilities 
  * [H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8 "Unregisters a filter.") removes a filter from the pipeline


##  Filter Plugins
HDF5 supports dynamic loading of filter plugins, allowing filters to be added without recompiling applications. See [HDF5 Filter Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#sec_filter_plugins) for details on creating and using filter plugins.
##  Summary
The H5Z filter interface provides: 
  * Built-in compression and error checking filters 
  * Flexible filter pipeline mechanism 
  * Support for custom user-defined filters 
  * Dynamic filter plugin loading 
  * Per-chunk processing for optimal performance


Filters are essential for reducing storage requirements and ensuring data integrity in HDF5 files while maintaining compatibility and performance.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


