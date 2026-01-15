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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title2 "H2")
  * [◆ H5Pget_alignment()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title3 "H2")
  * [◆ H5Pget_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title4 "H2")
  * [◆ H5Pget_coll_metadata_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title5 "H2")
  * [◆ H5Pget_core_write_tracking()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title6 "H2")
  * [◆ H5Pget_driver()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title7 "H2")
  * [◆ H5Pget_driver_config_str()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title8 "H2")
  * [◆ H5Pget_driver_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title9 "H2")
  * [◆ H5Pget_elink_file_cache_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title10 "H2")
  * [◆ H5Pget_evict_on_close()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title11 "H2")
  * [◆ H5Pget_family_offset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title12 "H2")
  * [◆ H5Pget_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title13 "H2")
  * [◆ H5Pget_fapl_direct()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title14 "H2")
  * [◆ H5Pget_fapl_family()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title15 "H2")
  * [◆ H5Pget_fapl_hdfs()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title16 "H2")
  * [◆ H5Pget_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title17 "H2")
  * [◆ H5Pget_fapl_mirror()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title18 "H2")
  * [◆ H5Pget_fapl_mpio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title19 "H2")
  * [◆ H5Pget_fapl_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title20 "H2")
  * [◆ H5Pget_fapl_onion()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title21 "H2")
  * [◆ H5Pget_fapl_ros3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title22 "H2")
  * [◆ H5Pget_fapl_ros3_endpoint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title23 "H2")
  * [◆ H5Pget_fapl_ros3_token()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title24 "H2")
  * [◆ H5Pget_fapl_splitter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title25 "H2")
  * [◆ H5Pget_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title26 "H2")
  * [◆ H5Pget_fclose_degree()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title27 "H2")
  * [◆ H5Pget_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title28 "H2")
  * [◆ H5Pget_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title29 "H2")
  * [◆ H5Pget_file_locking()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title30 "H2")
  * [◆ H5Pget_gc_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title31 "H2")
  * [◆ H5Pget_libver_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title32 "H2")
  * [◆ H5Pget_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title33 "H2")
  * [◆ H5Pget_mdc_image_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title34 "H2")
  * [◆ H5Pget_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title35 "H2")
  * [◆ H5Pget_meta_block_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title36 "H2")
  * [◆ H5Pget_metadata_read_attempts()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title37 "H2")
  * [◆ H5Pget_mpi_params()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title38 "H2")
  * [◆ H5Pget_multi_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title39 "H2")
  * [◆ H5Pget_object_flush_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title40 "H2")
  * [◆ H5Pget_page_buffer_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title41 "H2")
  * [◆ H5Pget_relax_file_integrity_checks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title42 "H2")
  * [◆ H5Pget_sieve_buf_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title43 "H2")
  * [◆ H5Pget_small_data_block_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title44 "H2")
  * [◆ H5Pget_vol_cap_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title45 "H2")
  * [◆ H5Pget_vol_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title46 "H2")
  * [◆ H5Pget_vol_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title47 "H2")
  * [◆ H5Pset_alignment()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title48 "H2")
  * [◆ H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title49 "H2")
  * [◆ H5Pset_coll_metadata_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title50 "H2")
  * [◆ H5Pset_core_write_tracking()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title51 "H2")
  * [◆ H5Pset_driver()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title52 "H2")
  * [◆ H5Pset_driver_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title53 "H2")
  * [◆ H5Pset_driver_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title54 "H2")
  * [◆ H5Pset_elink_file_cache_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title55 "H2")
  * [◆ H5Pset_evict_on_close()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title56 "H2")
  * [◆ H5Pset_family_offset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title57 "H2")
  * [◆ H5Pset_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title58 "H2")
  * [◆ H5Pset_fapl_direct()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title59 "H2")
  * [◆ H5Pset_fapl_family()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title60 "H2")
  * [◆ H5Pset_fapl_hdfs()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title61 "H2")
  * [◆ H5Pset_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title62 "H2")
  * [◆ H5Pset_fapl_log()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title63 "H2")
  * [◆ H5Pset_fapl_mirror()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title64 "H2")
  * [◆ H5Pset_fapl_mpio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title65 "H2")
  * [◆ H5Pset_fapl_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title66 "H2")
  * [◆ H5Pset_fapl_onion()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title67 "H2")
  * [◆ H5Pset_fapl_ros3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title68 "H2")
  * [◆ H5Pset_fapl_ros3_endpoint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title69 "H2")
  * [◆ H5Pset_fapl_ros3_token()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title70 "H2")
  * [◆ H5Pset_fapl_sec2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title71 "H2")
  * [◆ H5Pset_fapl_split()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title72 "H2")
  * [◆ H5Pset_fapl_splitter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title73 "H2")
  * [◆ H5Pset_fapl_stdio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title74 "H2")
  * [◆ H5Pset_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title75 "H2")
  * [◆ H5Pset_fapl_windows()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title76 "H2")
  * [◆ H5Pset_fclose_degree()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title77 "H2")
  * [◆ H5Pset_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title78 "H2")
  * [◆ H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title79 "H2")
  * [◆ H5Pset_file_locking()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title80 "H2")
  * [◆ H5Pset_gc_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title81 "H2")
  * [◆ H5Pset_libver_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title82 "H2")
  * [◆ H5Pset_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title83 "H2")
  * [◆ H5Pset_mdc_image_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title84 "H2")
  * [◆ H5Pset_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title85 "H2")
  * [◆ H5Pset_meta_block_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title86 "H2")
  * [◆ H5Pset_metadata_read_attempts()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title87 "H2")
  * [◆ H5Pset_mpi_params()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title88 "H2")
  * [◆ H5Pset_multi_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title89 "H2")
  * [◆ H5Pset_object_flush_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title90 "H2")
  * [◆ H5Pset_page_buffer_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title91 "H2")
  * [◆ H5Pset_relax_file_integrity_checks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title92 "H2")
  * [◆ H5Pset_sieve_buf_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title93 "H2")
  * [◆ H5Pset_small_data_block_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title94 "H2")
  * [◆ H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#title95 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#func-members)
File Access Properties
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html)
## Detailed Description
Use file access properties to modify the default behavior of the HDF5 library when accessing files. The properties include selecting a virtual file driver (VFD), configuring the metadata cache (MDC), control file locking, etc. These properties are _not_ persisted with files, and can be adjusted at runtime before a file is created or opened. 
File access property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.")/[H5Pget_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6735afde382cfd746b92a1a3b0e6a2ab "Retrieves the current settings for alignment properties from a file access property list.") | Sets/retrieves alignment properties.   
[H5Pset_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.")/[H5Pget_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9481a0b08d729ec68897d57db1827861 "Queries the raw data chunk cache parameters.") | Sets/retrieves metadata cache and raw data chunk cache parameters.   
[H5Pset_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250 "Sets write tracking information for core driver, H5FD_CORE.")/[H5Pget_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469 "Gets information about the write tracking feature used by the core VFD.") | Sets/retrieves write tracking information for core driver.   
[H5Pset_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325 "Sets the number of files that can be held open in an external link open file cache.")/[H5Pget_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4c9bcfff90f48bfefa2c25e551485923 "Retrieves the size of the external link open file cache.") | Sets/retrieves the size of the external link open file cache from the specified file access property list.   
[H5Pset_evict_on_close](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaca1f462af9b441a0fdc6d0da6d4e17de "Controls the library's behavior of evicting metadata associated with a closed object.")/[H5Pget_evict_on_close](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga465a52e20e031a604b719bbfe9883b66 "Retrieves the file access property list setting that determines whether an HDF5 object will be evicte...") | Set/get the file access property list setting that determines whether an HDF5 object will be evicted from the library's metadata cache when it is closed.   
[H5Pset_gc_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga61f01a12d5392ccf1321168f3c28f36f "Sets garbage collecting references flag.")/[H5Pget_gc_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa81d8427b419d80eff6e1d216d99b71 "Returns garbage collecting references setting.") | Sets/retrieves garbage collecting references flag.   
[H5Pset_family_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6b24e6daf4816bbfb89b63bab40aa982 "Sets offset property for low-level access to a file in a family of files.") | Sets offset property for low-level access to a file in a family of files.   
[H5Pget_family_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14977eaaf6565ba871b575de3163f1b3 "Retrieves a data offset from the file access property list.") | Retrieves a data offset from the file access property list.   
[H5Pset_fclose_degree](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c "Sets the file close degree.")/[H5Pget_fclose_degree](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga41da04bb4f823ba9f7d6c57dc8fe2878 "Returns the file close degree.") | Sets/retrieves file close degree property.   
[H5Pset_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer.") | Sets an initial file image in a memory buffer.   
[H5Pget_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga337626cc516d5d1e3303ea6bc350e56b "Retrieves a copy of the file image designated as the initial content and structure of a file.") | Retrieves a copy of the file image designated as the initial content and structure of a file.   
[H5Pset_file_image_callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.")/[H5Pget_file_image_callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e "Retrieves callback routines for working with file images.") | Sets/gets the callbacks for working with file images.   
[H5Pset_file_locking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaae4de80c533a38da9f269c02b8e71de2 "Sets the file locking property values.")/[H5Pget_file_locking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6f68d03de151117cefa29b571865a394 "Retrieves the file locking property values.") | Sets/retrieves file locking property values.   
[H5Pset_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1 "Sets the minimum metadata block size.")/[H5Pget_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac17861181246af0209c0da5209305461 "Returns the current metadata block size setting.") | Sets the minimum metadata blocksize or retrieves the current metadata block size setting.   
[H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5 "Sets the number of read attempts in a file access property list.")/[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e "Retrieves the number of read attempts from a file access property list.") | Sets/gets the number of read attempts from a file access property list.   
[H5Pset_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaf234199ad4cf9c708f45893f7f9cd4d3 "Set the initial metadata cache configuration in the indicated File Access Property List to the suppli...")/[H5Pget_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga3012f7f3310c7d25ada7617896bef1ee "Get the current initial metadata cache configuration from the provided file access property list.") | Set/get the initial metadata cache configuration in the indicated file access property list.   
[H5Pset_mdc_image_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65cf9fea33d1324009efc2d5db848434 "Sets the metadata cache image option for a file access property list.")/[H5Pget_mdc_image_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa18d59ee9efb12626410b1638f76f00 "Retrieves the metadata cache image configuration values for a file access property list.") | Set/get the metadata cache image option for a file access property list.   
[H5Pset_mdc_log_options](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.")/[H5Pget_mdc_log_options](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374 "Gets metadata cache logging options.") | Set/get the metadata cache logging options.   
[H5Pset_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga507341f31848c57008a3225bff3fe128 "Specifies type of data to be accessed via the MULTI driver, enabling more direct access.")/[H5Pget_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga251515e9fee4641037b4866a4f7c49fe "Retrieves type of data property for MULTI driver.") | Sets/gets the type of data property for the MULTI driver.   
[H5Pset_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19 "Sets a callback function to invoke when an object flush occurs in the file.")/[H5Pget_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadb66d434fd8d2f600213b0eec539564e "Retrieves the object flush property values from the file access property list.") | Set/get the object flush property values from the file access property list.   
[H5Pset_page_buffer_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8008cddafa81bd1ddada23f6d9a161ca "Sets the maximum size for the page buffer and the minimum percentage for metadata and raw data pages.")/[H5Pget_page_buffer_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0da11baf31cf424d053aa7952c933d98 "Retrieves the maximum size for the page buffer and the minimum percentage for metadata and raw data p...") | Set/get the maximum size for the page buffer.   
[H5Pset_sieve_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24fd737955839194bf5605d5f47928ee "Sets the maximum size of the data sieve buffer.")/[H5Pget_sieve_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac2321d0c34bb2b3cf33cd7bf02ca8e66 "Returns maximum data sieve buffer size.") | Sets/retrieves maximum size of data sieve buffer.   
[H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.") | Sets bounds on library versions, and indirectly format versions, to be used when creating objects.   
[H5Pget_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad5d7e671c3a06bcee64bc25841aaf607 "Retrieves library version bounds settings that indirectly control the format versions used when creat...") | Retrieves library version bounds settings that indirectly control the format versions used when creating objects.   
[H5Pset_small_data_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5a99962a79412814b79be830f14c23dd "Sets the size of a contiguous block reserved for small data.") | Sets the size of a contiguous block reserved for small data.   
[H5Pget_small_data_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6896bea06d7744b56e22347f572f5470 "Retrieves the current small data block size setting.") | Retrieves the current small data block size setting.   
[H5Pset_vol](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0 "Set the file VOL connector for a file access property list.") | Sets the file VOL connector for a file access property list.   
[H5Pget_vol_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2ad4dc5c6e5e4271334a7b1c6ee0777d "Query the capability flags for the VOL connector that will be used with this file access property lis...") | Retrieves the capability flags for the VOL connector that will be used with a file access property list.   
[H5Pget_vol_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5f133bdf09ca5a32622688d1ba5cc838 "Returns the identifier of the current VOL connector.") | Retrieves the identifier of the current VOL connector.   
[H5Pget_vol_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafc58db23c257cdcf2f0c1c3ae911ab0f "Returns a copy of the VOL information for a connector.") | Retrieves a copy of the VOL information for a connector.   
[H5Pset_mpi_params](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6daceb4a9e51fca7cb198f964b67baf0 "Set the MPI communicator and info.")/[H5Pget_mpi_params](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5554cf0775f9d7ac3b0cd844533d4486 "Get the MPI communicator and info.") | Sets/retrieves the MPI communicator and info.   
[H5Pset_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca "Sets metadata write mode to be collective or independent \(default\)")/[H5Pget_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga565428cd6c090a8c7836b7adf7720403 "Retrieves metadata write mode setting.") | Sets/retrieves metadata write mode setting.   
File driver property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e "Sets a file driver.") | Sets a file driver.   
[H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") | Returns the identifier for the driver used to create a file.   
[H5Pget_driver_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga1b072297fed53cd8586604e45c483a56 "Returns a pointer to file driver information.") | Returns a pointer to file driver information.   
[H5Pset_driver_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga156702db27ece40d21b37be5fe5e8b15 "Sets a file driver according to a given driver name.") | Sets a file driver according to a given driver name.   
[H5Pset_driver_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac4426b1d36aa8766fbe2deaf67a18c06 "Sets a file driver according to a given driver value \(ID\).") | Sets a file driver according to a given driver value.   
[H5Pget_driver_config_str](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac0d9eb573b84ce125433e81b2a475085 "Retrieves a string representation of the configuration for the driver set on the given FAPL....") | Retrieves a string representation of the configuration for the driver.   
[H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.")/[H5Pget_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50 "Queries core file driver properties.") | Sets the driver for buffered memory files (in RAM) or retrieves information regarding the driver.   
[H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.")/[H5Pget_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d "Retrieves direct I/O driver settings.") | Sets up use of the direct I/O driver or retrieves the direct I/O driver settings.   
[H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver.")/[H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375 "Returns file access property list information.") | Sets driver for file families, designed for systems that do not support files larger than 2 gigabytes, or retrieves information regarding driver.   
[H5Pset_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931 "Modifies the file access property list to use the H5FD_HDFS driver.")/[H5Pget_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafb4ec2b60ab86eade5aa55f5e4f5fd35 "Queries a File Access Property List for H5FD_HDFS file driver properties.") | .   
[H5Pset_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0 "Modifies the specified File Access Property List to use the H5FD_IOC driver.")/[H5Pget_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8482cdac1272818e4e0f86a665f8ef98 "Queries a File Access Property List for H5FD_IOC file driver properties.") | Modifies/queries the file driver properties of the I/O concentrator driver.   
[H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d "Sets up the logging virtual file driver \(H5FD_LOG\) for use.") | The logging driver is a clone of the standard SEC2 ([H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606)) driver with additional facilities for logging metrics and activity to a file.   
[H5Pset_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf "Modifies the file access property list to use the H5FD_MIRROR driver.")/[H5Pget_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga62a0c63720675e1698d61b7321beb1c1 "Queries a File Access Property List for H5FD_MIRROR file driver properties.") | Modifies/queries the file driver properties of the mirror driver.   
[H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.")/[H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a "Returns MPI IO communicator information.") | Sets driver for files on parallel file systems (MPI I/O) or retrieves information regarding the driver.   
H5Pset_fapl_mpiposix/H5Pget_fapl_mpiposix  | No longer available.   
[H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.")/[H5Pget_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d "Returns information about the multi-file access property list.") | Sets driver for multiple files, separating categories of metadata and raw data, or retrieves information regarding driver.   
[H5Pset_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762 "set the onion info for the file access property list")/[H5Pget_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaedf8b83d7571bfffb7336a8e11a98107 "get the onion info from the file access property list") | Modifies/queries the file driver properties of the onion driver.   
[H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.")/[H5Pget_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e "Queries a File Access Property List for H5FD_ROS3 file driver properties.") | Modifies/queries the file driver properties of the ros3 driver.   
[H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db "Modifies the file access property list to use the H5FD_SEC2 driver.") | Sets driver for unbuffered permanent files or retrieves information regarding driver.   
[H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver.") | Sets driver for split files, a limited case of multi driver with one metadata file and one raw data file.   
[H5Pset_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5 "Sets the file access property list to use the splitter driver.")/[H5Pget_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6e850a0319dc6839e2dd06a0522b3831 "Gets splitter driver properties from the the file access property list.") | Modifies/queries the file driver properties of the splitter driver.   
[H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052 "Sets the standard I/O driver.") | Sets driver for buffered permanent files.   
[H5Pset_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver.")/[H5Pget_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45 "Queries a File Access Property List for H5FD_SUBFILING file driver properties.") | Modifies/queries the file driver properties of the subfiling driver.   
[H5Pset_fapl_windows](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e "Sets the Windows I/O driver.") | Sets the Windows I/O driver.   
[H5Pset_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga507341f31848c57008a3225bff3fe128 "Specifies type of data to be accessed via the MULTI driver, enabling more direct access.") | Specifies type of data to be accessed via the MULTI driver enabling more direct access.   
[H5Pget_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga251515e9fee4641037b4866a4f7c49fe "Retrieves type of data property for MULTI driver.") | Retrieves type of data property for MULTI driver.   
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6735afde382cfd746b92a1a3b0e6a2ab) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *threshold, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *alignment)  
| Retrieves the current settings for alignment properties from a file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9481a0b08d729ec68897d57db1827861) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, int *mdc_nelmts, size_t *rdcc_nslots, size_t *rdcc_nbytes, double *rdcc_w0)  
| Queries the raw data chunk cache parameters.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga565428cd6c090a8c7836b7adf7720403) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, bool *is_collective)  
| Retrieves metadata write mode setting.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, bool *is_enabled, size_t *page_size)  
| Gets information about the write tracking feature used by the core VFD.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id)  
| Returns low-lever driver identifier.   
  
ssize_t  |  [H5Pget_driver_config_str](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac0d9eb573b84ce125433e81b2a475085) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, char *config_buf, size_t buf_size)  
| Retrieves a string representation of the configuration for the driver set on the given FAPL. The returned string can be used to configure the same driver in an identical way.   
  
const void *  |  [H5Pget_driver_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga1b072297fed53cd8586604e45c483a56) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id)  
| Returns a pointer to file driver information.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4c9bcfff90f48bfefa2c25e551485923) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned *efc_size)  
| Retrieves the size of the external link open file cache.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_evict_on_close](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga465a52e20e031a604b719bbfe9883b66) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, bool *evict_on_close)  
| Retrieves the file access property list setting that determines whether an HDF5 object will be evicted from the library's metadata cache when it is closed.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_family_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14977eaaf6565ba871b575de3163f1b3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset)  
| Retrieves a data offset from the file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t *increment, bool *backing_store)  
| Queries core file driver properties.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t *boundary, size_t *block_size, size_t *cbuf_size)  
| Retrieves direct I/O driver settings.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *memb_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *memb_fapl_id)  
| Returns file access property list information.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafb4ec2b60ab86eade5aa55f5e4f5fd35) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_hdfs_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html) *fa_out)  
| Queries a File Access Property List for [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) file driver properties.   
  
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8482cdac1272818e4e0f86a665f8ef98) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html) *config_out)  
| Queries a File Access Property List for [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) file driver properties.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga62a0c63720675e1698d61b7321beb1c1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html) *fa_out)  
| Queries a File Access Property List for [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) file driver properties.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, MPI_Comm *comm, MPI_Info *info)  
| Returns MPI IO communicator information.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) *memb_map, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *memb_fapl, char **memb_name, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *memb_addr, bool *relax)  
| Returns information about the multi-file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaedf8b83d7571bfffb7336a8e11a98107) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) *fa_out)  
| get the onion info from the file access property list   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html) *fa_out)  
| Queries a File Access Property List for [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) file driver properties.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_ros3_endpoint](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24b1d9fd0e33117b735635a9d198becb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t size, char *endpoint)  
| Queries a File Access Property List for [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) file driver endpoint url.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_ros3_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga255ea505c742118f8fd25b1eb948f8fb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t size, char *token)  
| Queries a File Access Property List for [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) file driver session/security token.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6e850a0319dc6839e2dd06a0522b3831) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_splitter_vfd_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__splitter__vfd__config__t.html) *config_ptr)  
| Gets splitter driver properties from the the file access property list.   
  
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_subfiling_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__subfiling__config__t.html) *config_out)  
| Queries a File Access Property List for [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) file driver properties.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fclose_degree](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga41da04bb4f823ba9f7d6c57dc8fe2878) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5F_close_degree_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475f) *degree)  
| Returns the file close degree.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga337626cc516d5d1e3303ea6bc350e56b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, void **buf_ptr_ptr, size_t *buf_len_ptr)  
| Retrieves a copy of the file image designated as the initial content and structure of a file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_file_image_callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) *callbacks_ptr)  
| Retrieves callback routines for working with file images.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_file_locking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6f68d03de151117cefa29b571865a394) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, bool *use_file_locking, bool *ignore_when_disabled)  
| Retrieves the file locking property values.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_gc_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa81d8427b419d80eff6e1d216d99b71) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, unsigned *gc_ref)  
| Returns garbage collecting references setting.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad5d7e671c3a06bcee64bc25841aaf607) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) *low, [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) *high)  
| Retrieves library version bounds settings that indirectly control the format versions used when creating objects.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga3012f7f3310c7d25ada7617896bef1ee) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) *config_ptr)  
| Get the current initial metadata cache configuration from the provided file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_mdc_image_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa18d59ee9efb12626410b1638f76f00) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html) *config_ptr)  
| Retrieves the metadata cache image configuration values for a file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_mdc_log_options](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, bool *is_enabled, char *location, size_t *location_size, bool *start_on_access)  
| Gets metadata cache logging options.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac17861181246af0209c0da5209305461) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *size)  
| Returns the current metadata block size setting.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned *attempts)  
| Retrieves the number of read attempts from a file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_mpi_params](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5554cf0775f9d7ac3b0cd844533d4486) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, MPI_Comm *comm, MPI_Info *info)  
| Get the MPI communicator and info.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga251515e9fee4641037b4866a4f7c49fe) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) *type)  
| Retrieves type of data property for MULTI driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadb66d434fd8d2f600213b0eec539564e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5F_flush_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a07cc80d29d745646218aa8cb068cf944) *func, void **udata)  
| Retrieves the object flush property values from the file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_page_buffer_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0da11baf31cf424d053aa7952c933d98) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, size_t *buf_size, unsigned *min_meta_perc, unsigned *min_raw_perc)  
| Retrieves the maximum size for the page buffer and the minimum percentage for metadata and raw data pages.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_relax_file_integrity_checks](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab1e57751a5f34c1a7cb6dd0194877d3e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, uint64_t *flags)  
| Retrieve relaxed file integrity check flags.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_sieve_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac2321d0c34bb2b3cf33cd7bf02ca8e66) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t *size)  
| Returns maximum data sieve buffer size.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_small_data_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6896bea06d7744b56e22347f572f5470) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *size)  
| Retrieves the current small data block size setting.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_vol_cap_flags](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2ad4dc5c6e5e4271334a7b1c6ee0777d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, uint64_t *cap_flags)  
| Query the capability flags for the VOL connector that will be used with this file access property list (FAPL).   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_vol_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5f133bdf09ca5a32622688d1ba5cc838) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *vol_id)  
| Returns the identifier of the current VOL connector.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_vol_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafc58db23c257cdcf2f0c1c3ae911ab0f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, void **vol_info)  
| Returns a copy of the VOL information for a connector.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) threshold, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) alignment)  
| Sets alignment properties of a file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, int mdc_nelmts, size_t rdcc_nslots, size_t rdcc_nbytes, double rdcc_w0)  
| Sets the raw data chunk cache parameters.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, bool is_collective)  
| Sets metadata write mode to be collective or independent (default)   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_core_write_tracking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, bool is_enabled, size_t page_size)  
| Sets write tracking information for core driver, [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a).   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) driver_id, const void *driver_info)  
| Sets a file driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_driver_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga156702db27ece40d21b37be5fe5e8b15) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, const char *driver_name, const char *driver_config)  
| Sets a file driver according to a given driver name.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_driver_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac4426b1d36aa8766fbe2deaf67a18c06) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5FD_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a17ff64731b589ac818c2309d0d0ce8fb) driver_value, const char *driver_config)  
| Sets a file driver according to a given driver value (ID).   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned efc_size)  
| Sets the number of files that can be held open in an external link open file cache.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_evict_on_close](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaca1f462af9b441a0fdc6d0da6d4e17de) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, bool evict_on_close)  
| Controls the library's behavior of evicting metadata associated with a closed object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_family_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6b24e6daf4816bbfb89b63bab40aa982) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) offset)  
| Sets offset property for low-level access to a file in a family of files.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t increment, bool backing_store)  
| Modifies the file access property list to use the [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t alignment, size_t block_size, size_t cbuf_size)  
| Sets up use of the direct I/O driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) memb_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) memb_fapl_id)  
| Sets the file access property list to use the family driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_hdfs_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html) *fa)  
| Modifies the file access property list to use the [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) driver.   
  
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html) *vfd_config)  
| Modifies the specified File Access Property List to use the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const char *logfile, unsigned long long flags, size_t buf_size)  
| Sets up the logging virtual file driver ([H5FD_LOG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a027aaf28f5104c77c4f51ecd29a5f7f4)) for use.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html) *fa)  
| Modifies the file access property list to use the [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, MPI_Comm comm, MPI_Info info)  
| Stores MPI IO communicator information to the file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) *memb_map, const [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *memb_fapl, const char *const *memb_name, const [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *memb_addr, bool relax)  
| Sets up use of the multi-file driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) *fa)  
| set the onion info for the file access property list   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html) *fa)  
| Modifies the specified File Access Property List to use the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_ros3_endpoint](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga316f9801b3b0a02916fd4a721532aeda) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const char *endpoint)  
| Modifies the specified File Access Property List to use an alternative endpoint URL when opening files with the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_ros3_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga74d2ae62171700a1a50c9fd76713cf4d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const char *token)  
| Modifies the specified File Access Property List to use the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver by adding the specified session/security token.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)  
| Modifies the file access property list to use the [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl, const char *meta_ext, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) meta_plist_id, const char *raw_ext, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) raw_plist_id)  
| Emulates the old split file driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_splitter_vfd_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__splitter__vfd__config__t.html) *config_ptr)  
| Sets the file access property list to use the splitter driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)  
| Sets the standard I/O driver.   
  
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const [H5FD_subfiling_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__subfiling__config__t.html) *vfd_config)  
| Modifies the specified File Access Property List to use the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_windows](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)  
| Sets the Windows I/O driver.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fclose_degree](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5F_close_degree_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475f) degree)  
| Sets the file close degree.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, void *buf_ptr, size_t buf_len)  
| Sets an initial file image in a memory buffer.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_file_image_callbacks](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) *callbacks_ptr)  
| Sets the callbacks for working with file images.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_file_locking](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaae4de80c533a38da9f269c02b8e71de2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, bool use_file_locking, bool ignore_when_disabled)  
| Sets the file locking property values.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_gc_references](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga61f01a12d5392ccf1321168f3c28f36f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, unsigned gc_ref)  
| Sets garbage collecting references flag.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) low, [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) high)  
| Controls the range of library release versions used when creating objects in a file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_mdc_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaf234199ad4cf9c708f45893f7f9cd4d3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) *config_ptr)  
| Set the initial metadata cache configuration in the indicated File Access Property List to the supplied value.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_mdc_image_config](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65cf9fea33d1324009efc2d5db848434) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html) *config_ptr)  
| Sets the metadata cache image option for a file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_mdc_log_options](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, bool is_enabled, const char *location, bool start_on_access)  
| Sets metadata cache logging options.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_meta_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size)  
| Sets the minimum metadata block size.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, unsigned attempts)  
| Sets the number of read attempts in a file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_mpi_params](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6daceb4a9e51fca7cb198f964b67baf0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, MPI_Comm comm, MPI_Info info)  
| Set the MPI communicator and info.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_multi_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga507341f31848c57008a3225bff3fe128) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) type)  
| Specifies type of data to be accessed via the `MULTI` driver, enabling more direct access.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5F_flush_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a07cc80d29d745646218aa8cb068cf944) func, void *udata)  
| Sets a callback function to invoke when an object flush occurs in the file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_page_buffer_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8008cddafa81bd1ddada23f6d9a161ca) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, size_t buf_size, unsigned min_meta_per, unsigned min_raw_per)  
| Sets the maximum size for the page buffer and the minimum percentage for metadata and raw data pages.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_relax_file_integrity_checks](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafa8e677af3200e155e9208522f8e05c0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, uint64_t flags)  
| Relax file integrity checks that may issue errors for some valid files.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_sieve_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24fd737955839194bf5605d5f47928ee) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t size)  
| Sets the maximum size of the data sieve buffer.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_small_data_block_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5a99962a79412814b79be830f14c23dd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size)  
| Sets the size of a contiguous block reserved for small data.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_vol](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_vol_id, const void *new_vol_info)  
| Set the file VOL connector for a file access property list.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6735afde382cfd746b92a1a3b0e6a2ab)H5Pget_alignment()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_alignment  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _threshold_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _alignment_ )  
Retrieves the current settings for alignment properties from a file access property list.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | threshold | Pointer to location of return threshold value   
[out] | alignment | Pointer to location of return alignment value 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_alignment()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6735afde382cfd746b92a1a3b0e6a2ab "Retrieves the current settings for alignment properties from a file access property list.") retrieves the current settings for alignment properties from a file access property list. The `threshold` and/or `alignment` pointers may be null pointers (NULL). 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9481a0b08d729ec68897d57db1827861)H5Pget_cache()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_cache  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | int * |  _mdc_nelmts_ ,   
|  | size_t * |  _rdcc_nslots_ ,   
|  | size_t * |  _rdcc_nbytes_ ,   
|  | double * |  _rdcc_w0_ )  
Queries the raw data chunk cache parameters.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in,out] | mdc_nelmts |  _No longer used_  
[in,out] | rdcc_nslots | Number of elements (objects) in the raw data chunk cache   
[in,out] | rdcc_nbytes | Total size of the raw data chunk cache, in bytes   
[in,out] | rdcc_w0 | Preemption policy 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9481a0b08d729ec68897d57db1827861 "Queries the raw data chunk cache parameters.") retrieves the maximum possible number of elements in the raw data chunk cache, the maximum possible number of bytes in the raw data chunk cache, and the preemption policy value.
Any (or all) arguments may be null pointers, in which case the corresponding datum is not returned.
Note that the `mdc_nelmts` parameter is no longer used. 

Version
    1.8.0 Use of the `mdc_nelmts` parameter discontinued. Metadata cache configuration is managed with [H5Pset_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaf234199ad4cf9c708f45893f7f9cd4d3 "Set the initial metadata cache configuration in the indicated File Access Property List to the suppli...") and [H5Pget_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga3012f7f3310c7d25ada7617896bef1ee "Get the current initial metadata cache configuration from the provided file access property list.")      1.6.0 The `rdcc_nbytes` and `rdcc_nslots` parameters changed from type int to size_t. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga565428cd6c090a8c7836b7adf7720403)H5Pget_coll_metadata_write()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_coll_metadata_write  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | bool * |  _is_collective_ )  
Retrieves metadata write mode setting.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | is_collective | Pointer to a boolean value indicating whether metadata writes are collective (`>0`) or independent (`0`). _Default mode:_ Independent (`0`)  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_coll_metadata_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga565428cd6c090a8c7836b7adf7720403 "Retrieves metadata write mode setting.") retrieves the collective metadata write setting from the file access property into `is_collective`. 

See also
    
  * [H5Pget_all_coll_metadata_ops()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gaa278583dd1a989fa35a915def6518126 "Retrieves metadata read mode setting.")
  * [H5Pget_coll_metadata_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga565428cd6c090a8c7836b7adf7720403 "Retrieves metadata write mode setting.")
  * [H5Pset_all_coll_metadata_ops()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gac45976cb82e4d3f085f191e5872d0316 "Sets metadata I/O mode for read operations to be collective or independent \(default\)")
  * [H5Pset_coll_metadata_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca "Sets metadata write mode to be collective or independent \(default\)")
  * [Functions with No Access Property List Parameter that May Generate Metadata Reads](https://support.hdfgroup.org/documentation/hdf5/latest/maybe_metadata_reads.html)



Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469)H5Pget_core_write_tracking()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_core_write_tracking  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | bool * |  _is_enabled_ ,   
|  | size_t * |  _page_size_ )  
Gets information about the write tracking feature used by the core VFD.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | is_enabled | Whether the feature is enabled   
[out] | page_size | Size, in bytes, of write aggregation pages 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_core_write_tracking()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga767be5f6f42a14bb1550c47ac08ad469 "Gets information about the write tracking feature used by the core VFD.") retrieves information about the write tracking feature used by the core VFD.
When a file is created or opened for writing using the core virtual file driver (VFD) with the backing store option turned on, the VFD can be configured to track changes to the file and only write out the modified bytes. To avoid a large number of small writes, the changes can be aggregated into pages of a user-specified size. The core VFD is also known as the memory VFD. The driver identifier is [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a). 

Note
    This function is only for use with the core VFD and must be used after the call to [H5Pset_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver."). It is an error to use this function with any other VFD.      This function only applies to the backing store write operation, which typically occurs when the file is flushed or closed. This function has no relationship to the increment parameter passed to [H5Pset_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.").      For optimum performance, the `page_size` parameter should be a power of two. 

Since
    1.8.13 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8)H5Pget_driver()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Pget_driver  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _plist_id_ | ) |   
---|---|---|---|---|---  
Returns low-lever driver identifier.  

Parameters
     [in] | plist_id | Property list identifier  
---|---|--- 

Returns
    Returns a low level driver identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Pget_driver()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") returns the identifier of the low-level file driver associated with the file access property list or data transfer property list `plist_id`.
Valid driver identifiers distributed with HDF5 are listed and described in the following table. 
Supported file drivers Driver Name  | Driver Identifier  | Description  | Related API   
---|---|---|---  
POSIX  |  [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) | This driver uses POSIX file-system functions like read and write to perform I/O to a single, permanent file on local disk with no system buffering. This driver is POSIX-compliant and is the default file driver for all systems.  |  [H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db "Modifies the file access property list to use the H5FD_SEC2 driver.")  
Memory  |  [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) | With this driver, an application can work with a file in memory for faster reads and writes. File contents are kept in memory until the file is closed. At closing, the memory version of the file can be written back to disk or abandoned.  |  [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.")  
Log  |  [H5FD_LOG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a027aaf28f5104c77c4f51ecd29a5f7f4) | This is the [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) driver with logging capabilities.  |  [H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d "Sets up the logging virtual file driver \(H5FD_LOG\) for use.")  
Family  |  [H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6) | With this driver, the HDF5 file's address space is partitioned into pieces and sent to separate storage files using an underlying driver of the user's choice. This driver is for systems that do not support files larger than 2 gigabytes.  |  [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver.")  
Multi  |  [H5FD_MULTI](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmulti_8h.html#a754e05ae5e0f2d86f64002b338c0fd5c) | With this driver, data can be stored in multiple files according to the type of the data. I/O might work better if data is stored in separate files based on the type of data. The Split driver is a special case of this driver.  |  [H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.") / [H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver.")  
STDIO  |  [H5FD_STDIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dstdio_8h.html#a030a03b96a9f6e46035ce64e25389085) | This driver uses functions from the standard C stdio.h to perform I/O to a single, permanent file on local disk with additional system buffering.  |  [H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052 "Sets the standard I/O driver.")  
Split  |  [H5FD_SPLITTER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#ac6c45c6a8e1cb7f5b4400d95bf651eae) | This file driver splits a file into two parts. One part stores metadata, and the other part stores raw data. This splitting a file into two parts is a limited case of the Multi driver.  |  [H5Pset_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5 "Sets the file access property list to use the splitter driver.")  
Parallel  |  [H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf) | This is the standard HDF5 file driver for parallel file systems. This driver uses the MPI standard for both communication and file I/O.  |  [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.")  
Direct  |  [H5FD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a99213f218f9ab0c51f9c679228a1e436) | This is the [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) driver except data is written to or read from the file synchronously without being cached by the system.  |  [H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.")  
Mirror  |  [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) | Serial I/O to file using Unix “stdio” functions.  |  [H5Pset_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf "Modifies the file access property list to use the H5FD_MIRROR driver.")  
HDFS  |  [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) | Read-Only access to Hadoop Distributed File System (HDFS).  |  [H5Pset_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931 "Modifies the file access property list to use the H5FD_HDFS driver.")  
ros3  |  [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) | Read-Only access to Amazon's S3 service.  |  [H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.")  
Subfiling  |  [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) | Derived from other "stacked" VFDs such as the splitter, mirror, and family VFDs.  |  [H5Pset_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver.")  
IOC  |  [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) | Relays VFD calls to one VFD, and write calls to another VFD. Maintains two files.  |  [H5Pset_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0 "Modifies the specified File Access Property List to use the H5FD_IOC driver.")  
Onion  |  [H5FD_ONION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a1d6673897b4ebd1bad9846b5695ba346) | Provide in-file provenance and revision/version control.  |  [H5Pset_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762 "set the onion info for the file access property list")  
Windows  |  [H5FD_WINDOWS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dwindows_8h.html#ab5173993ddefd103bfb3d37c2837a9a4) | This driver was modified in HDF5-1.8.8 to be a wrapper of the POSIX driver, [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606). This change should not affect user applications.  |  [H5Pset_fapl_windows](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e "Sets the Windows I/O driver.")  
Parallel POSIX  | H5FD_MPIPOSIX  | This driver is no longer available  |   
Stream  | H5FD_STREAM  | This driver is no longer available.  |   
```
     This list does not include custom drivers that might be
     defined and registered by a user.

     The returned driver identifier is only valid as long as the
     file driver remains registered.

```


Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac0d9eb573b84ce125433e81b2a475085)H5Pget_driver_config_str()
ssize_t H5Pget_driver_config_str  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | char * |  _config_buf_ ,   
|  | size_t |  _buf_size_ )  
Retrieves a string representation of the configuration for the driver set on the given FAPL. The returned string can be used to configure the same driver in an identical way.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | config_buf | Driver configuration string output buffer   
[in] | buf_size | Size of driver configuration string output buffer 

Returns
    Returns the length of the driver configuration string on success (not including the NUL terminator). Returns negative on failure.  
[H5Pget_driver_config_str()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac0d9eb573b84ce125433e81b2a475085 "Retrieves a string representation of the configuration for the driver set on the given FAPL....") retrieves a string representation of the configuration for the driver set on the given FAPL. The returned string can be used to configure the same driver in an identical way.
If `config_buf` is NULL, the length of the driver configuration string is simply returned. The caller can then allocate a buffer of the appropriate size and call this routine again. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga1b072297fed53cd8586604e45c483a56)H5Pget_driver_info()
const void * H5Pget_driver_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _plist_id_ | ) |   
---|---|---|---|---|---  
Returns a pointer to file driver information.  

Parameters
     [in] | plist_id | File access or data transfer property list identifier  
---|---|--- 

Returns
    Returns a pointer to a struct containing low-level driver information. Otherwise returns NULL. NULL is also returned if no driver-specific properties have been registered. No error is pushed on the stack in this case.  
[H5Pget_driver_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga1b072297fed53cd8586604e45c483a56 "Returns a pointer to file driver information.") returns a pointer to file driver-specific information for the low-level driver associated with the file access or data transfer property list `plist_id`.
The pointer returned by this function points to an “uncopied” struct. Driver-specific versions of that struct are defined for each low-level driver in the relevant source code file H5FD*.c. For example, the struct used for the MULTI driver is `H5FD_multi_fapl_t` defined in H5FDmulti.c.
If no driver-specific properties have been registered, [H5Pget_driver_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga1b072297fed53cd8586604e45c483a56 "Returns a pointer to file driver information.") returns NULL. 

Note
     [H5Pget_driver_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga1b072297fed53cd8586604e45c483a56 "Returns a pointer to file driver information.") and [H5Pset_driver()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e "Sets a file driver.") are used only when creating a virtual file driver (VFD) in the virtual file layer (VFL). 

Version
    1.10.1 Return value was changed from _void_ * to _const_ _void_ *.       1.8.2 Function publicized in this release; previous releases described this function only in the virtual file driver documentation. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4c9bcfff90f48bfefa2c25e551485923)H5Pget_elink_file_cache_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_elink_file_cache_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned * |  _efc_size_ )  
Retrieves the size of the external link open file cache.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | efc_size | External link open file cache size in number of files 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_elink_file_cache_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4c9bcfff90f48bfefa2c25e551485923 "Retrieves the size of the external link open file cache.") retrieves the number of files that can be held open in an external link open file cache. 

Since
    1.8.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga465a52e20e031a604b719bbfe9883b66)H5Pget_evict_on_close()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_evict_on_close  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | bool * |  _evict_on_close_ )  
Retrieves the file access property list setting that determines whether an HDF5 object will be evicted from the library's metadata cache when it is closed.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | evict_on_close | Pointer to a variable that will indicate if the object will be evicted on close 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The library's metadata cache is fairly conservative about holding on to HDF5 object metadata (object headers, chunk index structures, etc.), which can cause the cache size to grow, resulting in memory pressure on an application or system. When enabled, the "evict on close" property will cause all metadata for an object to be immediately evicted from the cache as long as it is not referenced by any other open object.
See [H5Pset_evict_on_close()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaca1f462af9b441a0fdc6d0da6d4e17de "Controls the library's behavior of evicting metadata associated with a closed object.") for additional notes on behavior. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14977eaaf6565ba871b575de3163f1b3)H5Pget_family_offset()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_family_offset  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ )  
Retrieves a data offset from the file access property list.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | offset | Offset in bytes within the HDF5 file 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_family_offset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14977eaaf6565ba871b575de3163f1b3 "Retrieves a data offset from the file access property list.") retrieves the value of offset from the file access property list `fapl_id` so that the user application can retrieve a file handle for low-level access to a particular member of a family of files. The file handle is retrieved with a separate call to [H5Fget_vfd_handle()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae4020a66fb8da0586e3b74c81ffccea4 "Returns pointer to the file handle from the virtual file driver.") (or, in special circumstances, to [H5FDget_vfd_handle()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a9fcfb5d6708f4c3f5d319b801ac252bc), see [HDF5 Virtual File Layer](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html)). 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50)H5Pget_fapl_core()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_core  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | size_t * |  _increment_ ,   
|  | bool * |  _backing_store_ )  
Queries core file driver properties.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | increment | Size, in bytes, of memory increments   
[out] | backing_store | Boolean flag indicating whether to write the file contents to disk when the file is closed  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50 "Queries core file driver properties.") queries the [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) driver properties as set by [H5Pset_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver."). 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d)H5Pget_fapl_direct()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_direct  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | size_t * |  _boundary_ ,   
|  | size_t * |  _block_size_ ,   
|  | size_t * |  _cbuf_size_ )  
Retrieves direct I/O driver settings.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | boundary | Required memory alignment boundary   
[out] | block_size | File system block size   
[out] | cbuf_size | Copy buffer size  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_direct()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d "Retrieves direct I/O driver settings.") retrieves the required memory alignment (`alignment`), file system block size (`block_size`), and copy buffer size (`cbuf_size`) settings for the direct I/O driver, [H5FD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a99213f218f9ab0c51f9c679228a1e436), from the file access property list `fapl_id`.
See [H5Pset_fapl_direct()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.") for discussion of these values, requirements, and important considerations. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375)H5Pget_fapl_family()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_family  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _memb_size_ ,   
|  |  [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) * |  _memb_fapl_id_ )  
Returns file access property list information.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | memb_size | Size in bytes of each file member   
[out] | memb_fapl_id | Identifier of file access property list for each family member  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_family()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375 "Returns file access property list information.") returns file access property list for use with the family driver. This information is returned through the output parameters. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafb4ec2b60ab86eade5aa55f5e4f5fd35)H5Pget_fapl_hdfs()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_hdfs  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_hdfs_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html) * |  _fa_out_ )  
Queries a File Access Property List for [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) file driver properties.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | fa_out | Pointer to [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) driver configuration structure  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_hdfs()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafb4ec2b60ab86eade5aa55f5e4f5fd35 "Queries a File Access Property List for H5FD_HDFS file driver properties.") queries the [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) driver properties as set by [H5Pset_fapl_hdfs()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931 "Modifies the file access property list to use the H5FD_HDFS driver."). 

Since
    1.10.6 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8482cdac1272818e4e0f86a665f8ef98)H5Pget_fapl_ioc()
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_ioc  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html) * |  _config_out_ )  
Queries a File Access Property List for [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) file driver properties.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | config_out | Pointer to [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html "Configuration structure for H5Pset_fapl_ioc\(\) / H5Pget_fapl_ioc\(\)") structure through which the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) file driver properties will be returned. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8482cdac1272818e4e0f86a665f8ef98 "Queries a File Access Property List for H5FD_IOC file driver properties.") queries the specified File Access Property List for [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver properties as set by [H5Pset_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0 "Modifies the specified File Access Property List to use the H5FD_IOC driver."). If the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver has not been set on the File Access Property List, a default configuration is returned. An HDF5 application may use this functionality to manually configure the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver by calling [H5Pget_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8482cdac1272818e4e0f86a665f8ef98 "Queries a File Access Property List for H5FD_IOC file driver properties.") on a newly-created File Access Property List, adjusting the default values and then calling [H5Pset_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0 "Modifies the specified File Access Property List to use the H5FD_IOC driver.") with the configured [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html "Configuration structure for H5Pset_fapl_ioc\(\) / H5Pget_fapl_ioc\(\)") structure. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga62a0c63720675e1698d61b7321beb1c1)H5Pget_fapl_mirror()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_mirror  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html) * |  _fa_out_ )  
Queries a File Access Property List for [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) file driver properties.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | fa_out | Pointer to [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) driver configuration structure  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_mirror()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga62a0c63720675e1698d61b7321beb1c1 "Queries a File Access Property List for H5FD_MIRROR file driver properties.") queries the [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) driver properties as set by [H5Pset_fapl_mirror()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf "Modifies the file access property list to use the H5FD_MIRROR driver."). 

Since
    1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a)H5Pget_fapl_mpio()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_mpio  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | MPI_Comm * |  _comm_ ,   
|  | MPI_Info * |  _info_ )  
Returns MPI IO communicator information.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | comm | MPI communicator   
[out] | info | MPI info object  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
If the file access property list is set to the [H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf) driver, [H5Pget_fapl_mpio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a "Returns MPI IO communicator information.") returns duplicates of the stored MPI communicator and Info object through the `comm` and `info` pointers, if those values are non-null.
Since the MPI communicator and Info object are duplicates of the stored information, future modifications to the access property list will not affect them. It is the responsibility of the application to free these objects. 

Version
    1.4.5 Handling of the MPI Communicator and Info object changed at this release. A duplicate of each of these is now stored in the property list instead of pointers to each.  

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d)H5Pget_fapl_multi()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_multi  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) * |  _memb_map_ ,   
|  |  [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) * |  _memb_fapl_ ,   
|  | char ** |  _memb_name_ ,   
|  |  [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) * |  _memb_addr_ ,   
|  | bool * |  _relax_ )  
Returns information about the multi-file access property list.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | memb_map | Maps memory usage types to other memory usage types   
[out] | memb_fapl | Property list for each memory usage type   
[out] | memb_name | Name generator for names of member files   
[out] | memb_addr | The offsets within the virtual address space, from 0 (zero) to [HADDR_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a49742d33813ee38ef58eca9fbeda6b86), at which each type of data storage begins   
[out] | relax | Allows read-only access to incomplete file sets when `true` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab72fe4184a465d8e0c30d95d3961125d "Returns information about the multi-file access property list.") returns information about the multi-file access property list. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaedf8b83d7571bfffb7336a8e11a98107)H5Pget_fapl_onion()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_onion  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) * |  _fa_out_ )  
get the onion info from the file access property list 
* * * 

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | fa_out | The pointer to the structure [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_onion()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaedf8b83d7571bfffb7336a8e11a98107 "get the onion info from the file access property list") retrieves the structure [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) from the file access property list that is set for the onion VFD driver. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e)H5Pget_fapl_ros3()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_ros3  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html) * |  _fa_out_ )  
Queries a File Access Property List for [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) file driver properties.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | fa_out | Pointer to [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver configuration structure.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24b1d9fd0e33117b735635a9d198becb)H5Pget_fapl_ros3_endpoint()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_ros3_endpoint  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | size_t |  _size_ ,   
|  | char * |  _endpoint_ )  
Queries a File Access Property List for [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) file driver endpoint url.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | size | Size of the provided char array for storing the endpoint url.   
[out] | endpoint | Alternate endpoint url.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Since
    2.0.0   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga255ea505c742118f8fd25b1eb948f8fb)H5Pget_fapl_ros3_token()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_ros3_token  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | size_t |  _size_ ,   
|  | char * |  _token_ )  
Queries a File Access Property List for [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) file driver session/security token.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | size | Size of the provided char array for storing the session/security token.   
[out] | token | Session/security token.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Since
    1.14.2   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6e850a0319dc6839e2dd06a0522b3831)H5Pget_fapl_splitter()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_splitter  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_splitter_vfd_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__splitter__vfd__config__t.html) * |  _config_ptr_ )  
Gets splitter driver properties from the the file access property list.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | config_ptr | Configuration options for the VFD  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_splitter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5 "Sets the file access property list to use the splitter driver.") sets the file access property list identifier, `fapl_id`, to use the splitter driver.
The splitter VFD echoes file manipulation (e.g. create, truncate) and write calls to a second file. 

Note
    The splitter VFD should not be confused with the split VFD, which is a simplification of the multi VFD and creates separate files for metadata and data. 

Since
    1.12.1, back-ported to 1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45)H5Pget_fapl_subfiling()
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fapl_subfiling  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_subfiling_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__subfiling__config__t.html) * |  _config_out_ )  
Queries a File Access Property List for [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) file driver properties.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | config_out | Pointer to [H5FD_subfiling_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__subfiling__config__t.html "Configuration structure for H5Pset_fapl_subfiling\(\) / H5Pget_fapl_subfiling\(\)") structure through which the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) file driver properties will be returned. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45 "Queries a File Access Property List for H5FD_SUBFILING file driver properties.") queries the specified File Access Property List for [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver properties as set by [H5Pset_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver."). If the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver has not been set on the File Access Property List, a default configuration is returned. An HDF5 application may use this functionality to manually configure the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver by calling [H5Pget_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45 "Queries a File Access Property List for H5FD_SUBFILING file driver properties.") on a newly-created File Access Property List, adjusting the default values and then calling [H5Pset_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver.") with the configured [H5FD_subfiling_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__subfiling__config__t.html "Configuration structure for H5Pset_fapl_subfiling\(\) / H5Pget_fapl_subfiling\(\)") structure.
The `ioc_fapl_id` field of the returned structure will be the ID of a copied or newly-created property list which should be closed with [H5Pclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb "Terminates access to a property list.") when the configuration structure is no longer in use. An application should also be sure to close this property list ID first if a different property list ID will be assigned to the `ioc_fapl_id` field. 

Note
     [H5Pget_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45 "Queries a File Access Property List for H5FD_SUBFILING file driver properties.") returns the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver properties as they were initially set for the File Access Property List using [H5Pset_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver."). Alternatively, the driver properties can be modified at runtime according to values set for the [H5FD_SUBFILING_STRIPE_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#ae3da9be3aeeedafd84cca1371b6c5da4), [H5FD_SUBFILING_IOC_PER_NODE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a5a5883d4d3c713fdbefd27971b8a67b7) and [H5FD_SUBFILING_IOC_SELECTION_CRITERIA](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a8350a509d56b739b6cc8f4f1934d0e00) environment variables. However, driver properties set through environment variables will not be reflected in what is returned by [H5Pget_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2af70900a3ea50a83d65f8285730ef45 "Queries a File Access Property List for H5FD_SUBFILING file driver properties."), so an application may need to check those environment variables to get accurate values for the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver properties. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga41da04bb4f823ba9f7d6c57dc8fe2878)H5Pget_fclose_degree()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_fclose_degree  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5F_close_degree_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475f) * |  _degree_ )  
Returns the file close degree.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | degree | Pointer to a location to which to return the file close degree property, the value of `degree` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_fclose_degree()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga41da04bb4f823ba9f7d6c57dc8fe2878 "Returns the file close degree.") returns the current setting of the file close degree property `degree` in the file access property list `fapl_id`. The value of `degree` determines how aggressively [H5Fclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") deals with objects within a file that remain open when [H5Fclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") is called to close that file. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga337626cc516d5d1e3303ea6bc350e56b)H5Pget_file_image()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_file_image  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | void ** |  _buf_ptr_ptr_ ,   
|  | size_t * |  _buf_len_ptr_ )  
Retrieves a copy of the file image designated as the initial content and structure of a file.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in,out] | buf_ptr_ptr | On input, `NULL` or a pointer to a pointer to a buffer that contains the file image.  
On successful return, if `buf_ptr_ptr` is not `NULL`, `*buf_ptr_ptr` will contain a pointer to a copy of the initial image provided in the last call to [H5Pset_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer.") for the supplied `fapl_id`. If no initial image has been set, `*buf_ptr_ptr` will be `NULL`.   
[in,out] | buf_len_ptr | On input, `NULL` or a pointer to a buffer specifying the required size of the buffer to hold the file image.  
On successful return, if `buf_len_ptr` was not passed in as `NULL`, `buf_len_ptr` will return the required size in bytes of the buffer to hold the initial file image in the supplied file access property list, `fapl_id`. If no initial image is set, the value of `*buf_len_ptr` will be set to 0 (zero)  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga337626cc516d5d1e3303ea6bc350e56b "Retrieves a copy of the file image designated as the initial content and structure of a file.") allows an application to retrieve a copy of the file image designated for a VFD to use as the initial contents of a file.
If file image callbacks are defined, [H5Pget_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga337626cc516d5d1e3303ea6bc350e56b "Retrieves a copy of the file image designated as the initial content and structure of a file.") will use them when allocating and loading the buffer to return to the application (see [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.")). If file image callbacks are not defined, the function will use `malloc` and `memcpy`. When `malloc` and `memcpy` are used, it is the caller's responsibility to discard the returned buffer with a call to `free`.
It is the responsibility of the calling application to free the buffer whose address is returned in `buf_ptr_ptr`. This can be accomplished with `free` if file image callbacks have not been set (see [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.")) or with the appropriate method if file image callbacks have been set. 

See also
     [H5LTopen_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae2834cca05d6c4787f51933cc521f8fb "Opens an HDF5 file image in memory."), [H5Fget_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadc53f4e76b1199cb5d2a8cb7fbb114ad "Retrieves a copy of the image of an existing, open file."), [H5Pset_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer."), [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images."), [H5Pget_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e "Retrieves callback routines for working with file images."), [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html), [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd), [HDF5 File Image](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_i_m__u_g.html). 

Since
    1.8.9 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e)H5Pget_file_image_callbacks()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_file_image_callbacks  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) * |  _callbacks_ptr_ )  
Retrieves callback routines for working with file images.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in,out] | callbacks_ptr | Pointer to the instance of the [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) struct in which the callback routines are to be returned  
Struct fields must be initialized to NULL before the call is made.  
Struct field contents upon return will match those passed in in the last [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") call for the file access property list `fapl_id`.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e "Retrieves callback routines for working with file images.") retrieves the callback routines set for working with file images opened with the file access property list `fapl_id`.
The callbacks must have been previously set with [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") in the file access property list.
Upon the successful return of [H5Pget_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e "Retrieves callback routines for working with file images."), the fields in the instance of the [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) struct pointed to by `callbacks_ptr` will contain the same values as were passed in the most recent [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") call for the file access property list `fapl_id`. 

See also
     [H5LTopen_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae2834cca05d6c4787f51933cc521f8fb "Opens an HDF5 file image in memory."), [H5Fget_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadc53f4e76b1199cb5d2a8cb7fbb114ad "Retrieves a copy of the image of an existing, open file."), [H5Pset_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer."), [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images."), [H5Pget_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e "Retrieves callback routines for working with file images."), [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html), [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd), [HDF5 File Image](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_i_m__u_g.html). 

Since
    1.8.9 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6f68d03de151117cefa29b571865a394)H5Pget_file_locking()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_file_locking  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | bool * |  _use_file_locking_ ,   
|  | bool * |  _ignore_when_disabled_ )  
Retrieves the file locking property values.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | use_file_locking | File locking flag   
[out] | ignore_when_disabled | Ignore when disabled flag  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_file_locking()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6f68d03de151117cefa29b571865a394 "Retrieves the file locking property values.") retrieves the file locking property values for the file access property list specified by `fapl_id`. 

Since
    1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa81d8427b419d80eff6e1d216d99b71)H5Pget_gc_references()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_gc_references  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | unsigned * |  _gc_ref_ )  
Returns garbage collecting references setting.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | gc_ref | Flag returning the state of reference garbage collection. A returned value of 1 indicates that garbage collection is on while 0 indicates that garbage collection is off. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_gc_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa81d8427b419d80eff6e1d216d99b71 "Returns garbage collecting references setting.") returns the current setting for the garbage collection references property from the specified file access property list. The garbage collection references property is set by [H5Pset_gc_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga61f01a12d5392ccf1321168f3c28f36f "Sets garbage collecting references flag."). 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad5d7e671c3a06bcee64bc25841aaf607)H5Pget_libver_bounds()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_libver_bounds  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) * |  _low_ ,   
|  |  [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) * |  _high_ )  
Retrieves library version bounds settings that indirectly control the format versions used when creating objects.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | low | The earliest version of the library that will be used for writing objects   
[out] | high | The latest version of the library that will be used for writing objects 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_libver_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad5d7e671c3a06bcee64bc25841aaf607 "Retrieves library version bounds settings that indirectly control the format versions used when creat...") retrieves the lower and upper bounds on the HDF5 library release versions that indirectly determine the object format versions used when creating objects in the file.
This property is retrieved from the file access property list specified by the parameter `fapl_id`.
The value returned in the parameters `low` and `high` is one of the enumerated values in the [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) struct, which is defined in [H5Fpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html). 

Version
    1.10.2 Add [H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8) to the enumerated defines in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga3012f7f3310c7d25ada7617896bef1ee)H5Pget_mdc_config()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_mdc_config  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) * |  _config_ptr_ )  
Get the current initial metadata cache configuration from the provided file access property list.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in,out] | config_ptr | Pointer to the instance of [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) in which the current metadata cache configuration is to be reported  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Note
    The `in` direction applies only to the [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aad880fc4455c253781e8968f2239d56f) field. All other fields are `out` parameters.  
The fields of the [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) structure are shown below: 
typedef struct [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) {
/* general configuration fields: */
int [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aad880fc4455c253781e8968f2239d56f);
bool [rpt_fcn_enabled](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af6bef1101778934a0f7d54963db4a9f5);
bool [open_trace_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ae92fb5b442027a53da62a853b9be7636);
bool [close_trace_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ab047662ba4d5dd6eb32835b472c38fb0);
char [trace_file_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a92e7d20eb2b7b353961c64558ddac080)[[H5AC__MAX_TRACE_FILE_NAME_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a_cpublic_8h.html#a717f1f3545cfc3d1b2208c96cc0c3bd3) + 1];
bool [evictions_enabled](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a1b7d261ea3c2cd28446253e8f3b67e0a);
bool [set_initial_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a5bfa4ced2bee62567910bb14dd28265b);
size_t [initial_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a649236e7dd714855a50f122aa5caca9f);
double [min_clean_fraction](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#abd805b98f873c1720f34a0ce937838fd);
size_t [max_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af4728438dee601cb2554d9bf18d78a43);
size_t [min_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af99ca22b80e05fd5b3603806348ab647);
long int [epoch_length](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ac998e51b01e0eef09d9a29c43f97e4bf);
/* size increase control fields: */
enum [H5C_cache_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#a040d488146ff1ca0a82209e9af3918fa) [incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ae825aaf759060239e92170d20eb97d26);
double [lower_hr_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a20f323fcb4747fc7228d2d74bb965586);
double [increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ac504dff76b24ab9f15536c51aec9fbbb);
bool [apply_max_increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aef1ae6503a0ae34abacce69d2996628e);
size_t [max_increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ad5a729f1d611f2780679a35b3524052c);
enum [H5C_cache_flash_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#aaaa13ca7756d135b7df6d5a6779ee908) [flash_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a0e25a1dc2c695bea335df0e23ed6363c);
double [flash_multiple](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a77b1812e0407c9122db524462a5c9633);
double [flash_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a95fb1e03a77ef5c109d0c851416ced55);
/* size decrease control fields: */
enum [H5C_cache_decr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#a4f8534794ad9a977185a5d608c0af04f) [decr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a5df68196b281c19d8ab7da0788566aec);
double [upper_hr_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a84a5ff4ac69196aa27c14f6f796db596);
double [decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a54007d3f2afb718b437f499a5c8b46d9);
bool [apply_max_decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a7ef00e6aab4e781e1e9b1344b3e23460);
size_t [max_decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a75e875a61c9da7f82482d0f6fe6e7152);
int [epochs_before_eviction](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a7ac41e345300bdecd9943e855d55b71b);
bool [apply_empty_reserve](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a14cfe890a5a58b493084730c988aa4ec);
double [empty_reserve](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a9c1ae995513b55737aad09e11beff733);
/* parallel configuration fields: */
size_t [dirty_bytes_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a8e3c2a2d300b7a8f8d3705fc5e59a3c1);
int [metadata_write_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a83a536128dbb7785b2553c294f33d1fe);
} [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html);
(Click on a enumerator, field, or type for more information.)
[H5Pget_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga3012f7f3310c7d25ada7617896bef1ee "Get the current initial metadata cache configuration from the provided file access property list.") gets the initial metadata cache configuration contained in a file access property list and loads it into the instance of [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) pointed to by the `config_ptr` parameter. This configuration is used when the file is opened.
Note that the version field of `*config_ptr` must be initialized; this allows the library to support earlier versions of the [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) structure.
See the overview of the metadata cache in the special topics section of the user guide for details on the configuration data returned. If you haven't read and understood that documentation, the results of this call will not make much sense. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa18d59ee9efb12626410b1638f76f00)H5Pget_mdc_image_config()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_mdc_image_config  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html) * |  _config_ptr_ )  
Retrieves the metadata cache image configuration values for a file access property list.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | config_ptr | Pointer to metadata cache image configuration values  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_mdc_image_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaaa18d59ee9efb12626410b1638f76f00 "Retrieves the metadata cache image configuration values for a file access property list.") retrieves the metadata cache image values into `config_ptr` for the file access property list specified in `plist_id`.
[H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html) is defined as follows: 
typedef struct [H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html) {
int [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aad880fc4455c253781e8968f2239d56f);
bool [generate_image](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aa1b45b71d5b59abae823123df7678fc7);
bool [save_resize_status](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aa8b1987de42d9297f927db576c72271f);
int [entry_ageout](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aec92d40c46311615f2155573aca27ec4);
} [H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html);
(Click on a enumerator, field, or type for more information.) 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374)H5Pget_mdc_log_options()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_mdc_log_options  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | bool * |  _is_enabled_ ,   
|  | char * |  _location_ ,   
|  | size_t * |  _location_size_ ,   
|  | bool * |  _start_on_access_ )  
Gets metadata cache logging options.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | is_enabled | Flag whether logging is enabled   
[out] | location | Location of log in UTF-8/ASCII (file path/name) (On Windows, this must be ASCII)   
[out] | location_size | Size in bytes of the location string   
[out] | start_on_access | Whether the logging begins as soon as the file is opened or created  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The metadata cache is a central part of the HDF5 library through which all file metadata reads and writes take place. File metadata is normally invisible to the user and is used by the library for purposes such as locating and indexing data. File metadata should not be confused with user metadata, which consists of attributes created by users and attached to HDF5 objects such as datasets via [Attributes (H5A)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html) API calls.
Due to the complexity of the cache, a trace/logging feature has been created that can be used by HDF5 developers for debugging and performance analysis. The functions that control this functionality will normally be of use to a very limited number of developers outside of The HDF Group. The functions have been documented to help users create logs that can be sent with bug reports.
Control of the log functionality is straightforward. Logging is enabled via the [H5Pset_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.") function, which will modify the file access property list used to open or create a file. This function has a flag that determines whether logging begins at file open or starts in a paused state. Log messages can then be controlled via the [H5Fstart_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67 "Starts logging metadata cache events if logging was previously enabled.") / [H5Fstop_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234 "Stops logging metadata cache events if logging was previously enabled and is currently ongoing.") functions. [H5Pget_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374 "Gets metadata cache logging options.") can be used to examine a file access property list, and [H5Fget_mdc_logging_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6 "Gets the current metadata cache logging status.") will return the current state of the logging flags.
The log format is described in the [Design: Metadata Cache Logging](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/Design-MetadataCache-Logging-THG20140224-v4.pdf) document. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac17861181246af0209c0da5209305461)H5Pget_meta_block_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_meta_block_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _size_ )  
Returns the current metadata block size setting.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | size | Minimum size, in bytes, of metadata block allocations 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Returns the current minimum size, in bytes, of new metadata block allocations. This setting is retrieved from the file access property list `fapl_id`.
This value is set by [H5Pset_meta_block_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1 "Sets the minimum metadata block size.") and is retrieved from the file access property list `fapl_id`. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)H5Pget_metadata_read_attempts()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_metadata_read_attempts  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned * |  _attempts_ )  
Retrieves the number of read attempts from a file access property list.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | attempts | The number of read attempts 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_metadata_read_attempts()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e "Retrieves the number of read attempts from a file access property list.") retrieves the number of read attempts that is set in the file access property list `plist_id`.
For a default file access property list, the value retrieved will depend on whether the user sets the number of attempts via [H5Pset_metadata_read_attempts()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5 "Sets the number of read attempts in a file access property list."):
  * If the number of attempts is set to N, the value returned will be N. 
  * If the number of attempts is not set, the value returned will be the default for non-SWMR access (1). SWMR is short for single-writer/multiple-reader. 


For the file access property list of a specified HDF5 file, the value retrieved will depend on how the file is opened and whether the user sets the number of read attempts via [H5Pset_metadata_read_attempts()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5 "Sets the number of read attempts in a file access property list."):
  * For a file opened with SWMR access:
    * If the number of attempts is set to N, the value returned will be N. 
    * If the number of attempts is not set, the value returned will be the default for SWMR access (100). 
  * For a file opened without SWMR access, the value retrieved will always be the default for non-SWMR access (1). The value set via [H5Pset_metadata_read_attempts()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5 "Sets the number of read attempts in a file access property list.") does not have any effect on non-SWMR access. 



Failure Modes
    
When the input property list is not a file access property list.
When the library is unable to retrieve the number of read attempts from the file access property list. 

Examples
    
The first example illustrates the two cases for retrieving the number of read attempts from a default file access property list.
/* Get a copy of file access property list */
fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
/* Retrieve the # of read attempts from the file access property list */
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)(fapl, &attempts);
/*
* The value returned in "attempts" will be 1 (default for non-SWMR access).
*/
/* Set the # of read attempts to 20 */
[H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5)(fapl, 20);
/* Retrieve the # of read attempts from the file access property list */
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)(fapl, &attempts);
/*
* The value returned in "attempts" will be 20 as set.
*/
/* Close the property list */
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
[H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e)
#define H5P_FILE_ACCESS
**Definition** H5Ppublic.h:56
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)
herr_t H5Pget_metadata_read_attempts(hid_t plist_id, unsigned *attempts)
Retrieves the number of read attempts from a file access property list.
[H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5)
herr_t H5Pset_metadata_read_attempts(hid_t plist_id, unsigned attempts)
Sets the number of read attempts in a file access property list.
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)
herr_t H5Pclose(hid_t plist_id)
Terminates access to a property list.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
The second example illustrates the two cases for retrieving the number of read attempts from the file access property list of a file opened with SWMR access.
/* Open the file with SWMR access and default file access property list */
fid = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILE, ([H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) | [H5F_ACC_SWMR_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a22b12837bca0dba6689096a370d73402)), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
/* Get the file's file access property list */
file_fapl = [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4)(fid);
/* Retrieve the # of read attempts from the file's file access property list */
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)(file_fapl, &attempts);
/*
* The value returned in "attempts" will be 100 (default for SWMR access).
*/
/* Close the property list */
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(file_fapl);
/* Close the file */
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(fid);
/* Create a copy of file access property list */
fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
/* Set the # of read attempts */
[H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5)(fapl, 20);
/* Open the file with SWMR access and the non-default file access property list */
fid = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILE, ([H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) | [H5F_ACC_SWMR_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a22b12837bca0dba6689096a370d73402)), fapl);
/* Get the file's file access property list */
file_fapl = [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4)(fid);
/* Retrieve the # of read attempts from the file's file access property list */
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)(file_fapl, &attempts);
/*
* The value returned in "attempts" will be 20.
*/
/* Close the property lists */
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(file_fapl);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
/* Close the file */
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(fid);
[H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394)
#define H5F_ACC_RDONLY
**Definition** H5Fpublic.h:28
[H5F_ACC_SWMR_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a22b12837bca0dba6689096a370d73402)
#define H5F_ACC_SWMR_READ
**Definition** H5Fpublic.h:49
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4)
hid_t H5Fget_access_plist(hid_t file_id)
Returns a file access property list identifier.
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
The third example illustrates the two cases for retrieving the number of read attempts from the file access property list of a file opened with non-SWMR access.
/* Open the file with non-SWMR access and default file access property list */
fid = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILE, [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
/* Get the file's file access property list */
file_fapl = [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4)(fid);
/* Retrieve the # of read attempts from the file's file access property list */
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)(file_fapl, &attempts);
/*
* The value returned in "attempts" will be 1 (default for non-SWMR access).
*/
/* Close the property list */
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(file_fapl);
/* Close the file */
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(fid);
/* Create a copy of file access property list */
fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
/* Set the # of read attempts */
[H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5)(fapl, 20);
/* Open the file with non-SWMR access and the non-default file access property list */
fid = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILE, [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), fapl);
/* Get the file's file access property list */
file_fapl = [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4)(fid);
/* Retrieve the # of read attempts from the file's file access property list */
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)(file_fapl, &attempts);
/*
* The value returned in "attempts" will be 1 (default for non-SWMR access).
*/
/* Close the property lists */
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(file_fapl);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
/* Close the file */
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(fid); 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5554cf0775f9d7ac3b0cd844533d4486)H5Pget_mpi_params()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_mpi_params  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | MPI_Comm * |  _comm_ ,   
|  | MPI_Info * |  _info_ )  
Get the MPI communicator and info.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | comm | MPI communicator   
[out] | info | MPI info object  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_mpi_params()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5554cf0775f9d7ac3b0cd844533d4486 "Get the MPI communicator and info.") gets the MPI communicator and info stored in the file access property list `fapl_id`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga251515e9fee4641037b4866a4f7c49fe)H5Pget_multi_type()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_multi_type  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) * |  _type_ )  
Retrieves type of data property for MULTI driver.  

Parameters
     [in] | fapl_id | File access property list or data transfer property list identifier   
---|---|---  
[out] | type | Type of data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_multi_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga251515e9fee4641037b4866a4f7c49fe "Retrieves type of data property for MULTI driver.") retrieves the type of data setting from the file access or data transfer property list `fapl_id`. This enables a user application to specify the type of data the application wishes to access so that the application can retrieve a file handle for low-level access to the particular member of a set of MULTI files in which that type of data is stored. The file handle is retrieved with a separate call to [H5Fget_vfd_handle()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae4020a66fb8da0586e3b74c81ffccea4 "Returns pointer to the file handle from the virtual file driver.") (or, in special circumstances, to [H5FDget_vfd_handle()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a9fcfb5d6708f4c3f5d319b801ac252bc); see the Virtual File Layer documentation for more information.
The type of data returned in `type` will be one of those listed in the discussion of the `type` parameter in the description of the function [H5Pset_multi_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga507341f31848c57008a3225bff3fe128 "Specifies type of data to be accessed via the MULTI driver, enabling more direct access.").
Use of this function is only appropriate for an HDF5 file written as a set of files with the MULTI file driver. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadb66d434fd8d2f600213b0eec539564e)H5Pget_object_flush_cb()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_object_flush_cb  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [H5F_flush_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a07cc80d29d745646218aa8cb068cf944) * |  _func_ ,   
|  | void ** |  _udata_ )  
Retrieves the object flush property values from the file access property list.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | func | The user-defined callback function   
[in] | udata | The user-defined input data for the callback function 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_object_flush_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadb66d434fd8d2f600213b0eec539564e "Retrieves the object flush property values from the file access property list.") gets the user-defined callback function that is set in the file access property list `fapl_id` and stored in the parameter `func`. The callback is invoked whenever an object flush occurs in the file. This routine also obtains the user-defined input data that is passed along to the callback function in the parameter `udata`. 

Example
    
The example below illustrates the usage of this routine to obtain the object flush property values.
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id;
unsigned counter;
H5F_object_flush_t *ret_cb;
unsigned *ret_counter;
/* Create a copy of the file access property list */
fapl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
/* Set up the object flush property values */
/* flush_cb: callback function to invoke when an object flushes (see below) */
/* counter: user data to pass along to the callback function */
[H5Pset_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19)(fapl_id, flush_cb, &counter);
/* Open the file */
file_id = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILE, [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
/* Get the file access property list for the file */
fapl = [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4)(file_id);
/* Retrieve the object flush property values for the file */
[H5Pget_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadb66d434fd8d2f600213b0eec539564e)(fapl, &ret_cb, &ret_counter);
/* ret_cb will point to flush_cb() */
/* ret_counter will point to counter */
/*
.
.
.
.
.
.
*/
/* The callback function for the object flush property */
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
flush_cb([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, void *_udata)
{
unsigned *flush_ct = (unsigned *)_udata;
++(*flush_ct);
return 0;
}
[H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4)
#define H5F_ACC_RDWR
**Definition** H5Fpublic.h:29
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Pset_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19)
herr_t H5Pset_object_flush_cb(hid_t plist_id, H5F_flush_cb_t func, void *udata)
Sets a callback function to invoke when an object flush occurs in the file.
[H5Pget_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadb66d434fd8d2f600213b0eec539564e)
herr_t H5Pget_object_flush_cb(hid_t plist_id, H5F_flush_cb_t *func, void **udata)
Retrieves the object flush property values from the file access property list. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0da11baf31cf424d053aa7952c933d98)H5Pget_page_buffer_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_page_buffer_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | size_t * |  _buf_size_ ,   
|  | unsigned * |  _min_meta_perc_ ,   
|  | unsigned * |  _min_raw_perc_ )  
Retrieves the maximum size for the page buffer and the minimum percentage for metadata and raw data pages.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | buf_size | Maximum size, in bytes, of the page buffer   
[out] | min_meta_perc | Minimum metadata percentage to keep in the page buffer before allowing pages containing metadata to be evicted  
[out] | min_raw_perc | Minimum raw data percentage to keep in the page buffer before allowing pages containing raw data to be evicted 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_page_buffer_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0da11baf31cf424d053aa7952c933d98 "Retrieves the maximum size for the page buffer and the minimum percentage for metadata and raw data p...") retrieves `buf_size`, the maximum size in bytes of the page buffer, `min_meta_perc`, the minimum metadata percentage, and `min_raw_perc`, the minimum raw data percentage. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab1e57751a5f34c1a7cb6dd0194877d3e)H5Pget_relax_file_integrity_checks()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_relax_file_integrity_checks  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | uint64_t * |  _flags_ )  
Retrieve relaxed file integrity check flags.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | flags | Relaxed file integrity check flags 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_relax_file_integrity_checks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab1e57751a5f34c1a7cb6dd0194877d3e "Retrieve relaxed file integrity check flags.") retrieves the relaxed file integrity check value into `flags` for the file access property list specified in `plist_id`. 

Since
    1.14.4 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac2321d0c34bb2b3cf33cd7bf02ca8e66)H5Pget_sieve_buf_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_sieve_buf_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | size_t * |  _size_ )  
Returns maximum data sieve buffer size.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | size | Maximum size, in bytes, of data sieve buffer 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_sieve_buf_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac2321d0c34bb2b3cf33cd7bf02ca8e66 "Returns maximum data sieve buffer size.") retrieves, size, the current maximum size of the data sieve buffer.
This value is set by [H5Pset_sieve_buf_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24fd737955839194bf5605d5f47928ee "Sets the maximum size of the data sieve buffer.") and is retrieved from the file access property list fapl_id. 

Version
    1.6.0 The `size` parameter has changed from type `hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)` to `size_t` 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6896bea06d7744b56e22347f572f5470)H5Pget_small_data_block_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_small_data_block_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _size_ )  
Retrieves the current small data block size setting.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[out] | size | Maximum size, in bytes, of the small data block 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_small_data_block_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6896bea06d7744b56e22347f572f5470 "Retrieves the current small data block size setting.") retrieves the current setting for the size of the small data block.
If the returned value is zero (0), the small data block mechanism has been disabled for the file. 

Since
    1.4.4 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2ad4dc5c6e5e4271334a7b1c6ee0777d)H5Pget_vol_cap_flags()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_vol_cap_flags  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | uint64_t * |  _cap_flags_ )  
Query the capability flags for the VOL connector that will be used with this file access property list (FAPL).  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | cap_flags | Flags that indicate the VOL connector capabilities 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_vol_cap_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga2ad4dc5c6e5e4271334a7b1c6ee0777d "Query the capability flags for the VOL connector that will be used with this file access property lis...") queries the current VOL connector information for a FAPL to retrieve the capability flags for the VOL connector stack, as will be used by a file open or create operation that uses this FAPL. 

Note
    This routine supports the use of the HDF5_VOL_CONNECTOR environment variable to override the VOL connector set programmatically for the FAPL (with H5Pset_vol).      The H5VL_CAP_FLAG_ASYNC flag can be checked to see if asynchronous operations are supported by the VOL connector stack. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5f133bdf09ca5a32622688d1ba5cc838)H5Pget_vol_id()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_vol_id  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) * |  _vol_id_ )  
Returns the identifier of the current VOL connector.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | vol_id | Current VOL connector identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_vol_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5f133bdf09ca5a32622688d1ba5cc838 "Returns the identifier of the current VOL connector.") returns the VOL connector identifier `vol_id` for the file access property list `plist_id`. This identifier should be closed with [H5VLclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html#gaa3324ac7aedf9362b498226903288094 "Closes a VOL connector identifier."). 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafc58db23c257cdcf2f0c1c3ae911ab0f)H5Pget_vol_info()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_vol_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | void ** |  _vol_info_ )  
Returns a copy of the VOL information for a connector.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | vol_info | The VOL information for a connector 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_vol_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafc58db23c257cdcf2f0c1c3ae911ab0f "Returns a copy of the VOL information for a connector.") returns a copy of the VOL information `vol_info` for a connector specified by the file access property list `plist_id`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a)H5Pset_alignment()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_alignment  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _threshold_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _alignment_ )  
Sets alignment properties of a file access property list.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | threshold | Threshold value. Note that setting the threshold value to 0 (zero) has the effect of a special case, forcing everything to be aligned   
[in] | alignment | Alignment value 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_alignment()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.") sets the alignment properties of a file access property list so that any file object greater than or equal in size to `threshold` bytes will be aligned on an address that is a multiple of `alignment`. The addresses are relative to the end of the user block; the alignment is calculated by subtracting the user block size from the absolute file address and then adjusting the address to be a multiple of `alignment`.
Default values for `threshold` and `alignment` are one, implying no alignment. Generally the default values will result in the best performance for single-process access to the file. For MPI IO and other parallel systems, choose an alignment that is a multiple of the disk block size.
If the file space handling strategy is set to [H5F_FSPACE_STRATEGY_PAGE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a9cc492c4b5c936e48716a8dab3691bccacd625bd864903e71132c9098929f5a0a), then the alignment set via this routine is ignored. The file space handling strategy is set by [H5Pset_file_space_strategy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l..."). 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89)H5Pset_cache()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_cache  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | int |  _mdc_nelmts_ ,   
|  | size_t |  _rdcc_nslots_ ,   
|  | size_t |  _rdcc_nbytes_ ,   
|  | double |  _rdcc_w0_ )  
Sets the raw data chunk cache parameters.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | mdc_nelmts | No longer used; any value passed is ignored   
[in] | rdcc_nslots | The number of chunk slots in the raw data chunk cache for this dataset. Increasing this value reduces the number of cache collisions, but slightly increases the memory used. Due to the hashing strategy, this value should ideally be a prime number. As a rule of thumb, this value should be at least 10 times the number of chunks that can fit in `rdcc_nbytes` bytes. For maximum performance, this value should be set approximately 100 times that number of chunks. The default value is 8191.   
[in] | rdcc_nbytes | Total size of the raw data chunk cache in bytes. The default size is 8 MiB per dataset.   
[in] | rdcc_w0 | The chunk preemption policy for all datasets. This must be between 0 and 1 inclusive and indicates the weighting according to which chunks which have been fully read or written are penalized when determining which chunks to flush from cache. A value of 0 means fully read or written chunks are treated no differently than other chunks (the preemption is strictly LRU), while a value of 1 means fully read or written chunks are always preempted before other chunks. If your application only reads or writes data once, this can be safely set to 1. Otherwise, this should be set lower depending on how often you re-read or re-write the same data. The default value is 0.75. If the value passed is [H5D_CHUNK_CACHE_W0_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a4a01949e0020fa9471811cc8ed271ee7), then the property will not be set on the dataset access property list, and the parameter will come from the file access property list. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.") sets the number of elements, the total number of bytes, and the preemption policy value for all datasets in a file on the file's file access property list.
The raw data chunk cache inserts chunks into the cache by first computing a hash value using the address of a chunk and then by using that hash value as the chunk's index into the table of cached chunks. In other words, the size of this hash table and the number of possible hash values are determined by the `rdcc_nslots` parameter. If a different chunk in the cache has the same hash value, a collision will occur, which will reduce efficiency. If inserting the chunk into the cache would cause the cache to be too big, then the cache will be pruned according to the `rdcc_w0` parameter.
The `mdc_nelmts` parameter is no longer used; any value passed in that parameter will be ignored.
**Motivation:** Setting raw data chunk cache parameters can be done with [H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters."), [H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters."), or a combination of both. [H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.") is used to adjust the chunk cache parameters for all datasets via a global setting for the file, and [H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters.") is used to adjust the chunk cache parameters for individual datasets. When both are used, parameters set with [H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters.") will override any parameters set with [H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters."). 

Note
    Optimum chunk cache parameters may vary widely depending on different data layout and access patterns. For datasets with low performance requirements for example, changing the cache settings can save memory.      Note: Raw dataset chunk caching is not currently supported when using the MPI I/O and MPI POSIX file drivers in read/write mode; see [H5Pset_fapl_mpio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list."). When using this file driver, all calls to [H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") and [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") will access the disk directly, and [H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.") will have no effect on performance.      Raw dataset chunk caching is supported when these drivers are used in read-only mode. 

Version
    1.8.0 The use of the `mdc_nelmts` parameter was discontinued. Metadata cache configuration is managed with [H5Pset_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaf234199ad4cf9c708f45893f7f9cd4d3 "Set the initial metadata cache configuration in the indicated File Access Property List to the suppli...") and [H5Pget_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga3012f7f3310c7d25ada7617896bef1ee "Get the current initial metadata cache configuration from the provided file access property list.").       1.6.0 The `rdcc_nbytes` and `rdcc_nelmts` parameters changed from type int to size_t.  

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca)H5Pset_coll_metadata_write()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_coll_metadata_write  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | bool |  _is_collective_ )  
Sets metadata write mode to be collective or independent (default)  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | is_collective | Boolean value indicating whether metadata writes are collective (`>0`) or independent (`0`). _Default mode:_ Independent (`0`)  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_coll_metadata_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca "Sets metadata write mode to be collective or independent \(default\)") tells the HDF5 library whether to perform metadata writes collectively (1) or independently (0).
If collective access is selected, then on a flush of the metadata cache, all processes will divide the metadata cache entries to be flushed evenly among themselves and issue a single MPI-IO collective write operation. This is the preferred method when the size of the metadata created by the application is large.
If independent access is selected, the library uses the default method for doing metadata I/O either from process zero or independently from each process. 

See also
    
  * [H5Pget_all_coll_metadata_ops()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gaa278583dd1a989fa35a915def6518126 "Retrieves metadata read mode setting.")
  * [H5Pget_coll_metadata_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga565428cd6c090a8c7836b7adf7720403 "Retrieves metadata write mode setting.")
  * [H5Pset_all_coll_metadata_ops()](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gac45976cb82e4d3f085f191e5872d0316 "Sets metadata I/O mode for read operations to be collective or independent \(default\)")
  * [H5Pset_coll_metadata_write()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca "Sets metadata write mode to be collective or independent \(default\)")
  * [Functions with No Access Property List Parameter that May Generate Metadata Reads](https://support.hdfgroup.org/documentation/hdf5/latest/maybe_metadata_reads.html)



Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gada5811359151b195aa5e1172becd0250)H5Pset_core_write_tracking()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_core_write_tracking  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | bool |  _is_enabled_ ,   
|  | size_t |  _page_size_ )  
Sets write tracking information for core driver, [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a).  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | is_enabled | Boolean value specifying whether feature is enabled   
[in] | page_size | Positive integer specifying size, in bytes, of write aggregation pages Value of 1 (one) enables tracking with no paging. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
When a file is created or opened for writing using the core virtual file driver (VFD) with the backing store option turned on, the core driver can be configured to track changes to the file and write out only the modified bytes.
This write tracking feature is enabled and disabled with `is_enabled`. The default setting is that write tracking is disabled, or off.
To avoid a large number of small writes, changes can be aggregated into pages of a user-specified size, `page_size`.
Setting `page_size` to 1 enables tracking with no page aggregation.
The backing store option is set via the function H5Pset_fapl_core. 

Attention
    
This function is only for use with the core VFD and must be used after the call to [H5Pset_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver."). It is an error to use this function with any other VFD.
It is an error to use this function when the backing store flag has not been set using [H5Pset_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.").
This function only applies to the backing store write operation which typically occurs when the file is flushed or closed. This function has no relationship to the increment parameter passed to [H5Pset_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.").
For optimum performance, the `page_size` parameter should be a power of two.
It is an error to set the page size to 0.  

Version
    1.8.14 C function modified in this release to return error if `page_size` is set to 0 (zero).  

Since
    1.8.13 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e)H5Pset_driver()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_driver  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _driver_id_ ,   
|  | const void * |  _driver_info_ )  
Sets a file driver.  

Parameters
     [in] | plist_id | Property list identifier   
---|---|---  
[in] | driver_id | The new driver identifier   
[in] | driver_info | Optional struct containing driver properties 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_driver()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e "Sets a file driver.") sets the file driver, driver_id, for a file access or data transfer property list, `plist_id`, and supplies an optional struct containing the driver-specific properties, `driver_info`.
The driver properties will be copied into the property list and the reference count on the driver will be incremented, allowing the caller to close the driver identifier but still use the property list. 

Version
    1.8.2 Function publicized in this release; previous releases described this function only in the virtual file driver documentation. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga156702db27ece40d21b37be5fe5e8b15)H5Pset_driver_by_name()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_driver_by_name  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | const char * |  _driver_name_ ,   
|  | const char * |  _driver_config_ )  
Sets a file driver according to a given driver name.  

Parameters
     [in] | plist_id | Property list identifier   
---|---|---  
[in] | driver_name | The new driver name   
[in] | driver_config | Optional string containing driver properties 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_driver_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga156702db27ece40d21b37be5fe5e8b15 "Sets a file driver according to a given driver name.") sets the file driver, by the name driver_name, for a file access or data transfer property list, `plist_id`, and supplies an optional string containing the driver-specific properties, `driver_config`. The driver properties string will be copied into the property list.
If the driver specified by `driver_name` is not currently registered, an attempt will be made to load the driver as a plugin. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac4426b1d36aa8766fbe2deaf67a18c06)H5Pset_driver_by_value()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_driver_by_value  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | [H5FD_class_value_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a17ff64731b589ac818c2309d0d0ce8fb) |  _driver_value_ ,   
|  | const char * |  _driver_config_ )  
Sets a file driver according to a given driver value (ID).  

Parameters
     [in] | plist_id | Property list identifier   
---|---|---  
[in] | driver_value | The new driver value (ID)   
[in] | driver_config | Optional string containing driver properties 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_driver_by_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac4426b1d36aa8766fbe2deaf67a18c06 "Sets a file driver according to a given driver value \(ID\).") sets the file driver, by the value driver_value, for a file access or data transfer property list, `plist_id`, and supplies an optional string containing the driver-specific properties, `driver_config`. The driver properties string will be copied into the property list.
If the driver specified by `driver_value` is not currently registered, an attempt will be made to load the driver as a plugin. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325)H5Pset_elink_file_cache_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_elink_file_cache_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned |  _efc_size_ )  
Sets the number of files that can be held open in an external link open file cache.  

Motivation
    
The _external link open file cache_ holds files open after they have been accessed via an external link. This cache reduces the number of times such files are opened when external links are accessed repeatedly and can significantly improves performance in certain heavy-use situations and when low-level file opens or closes are expensive.
[H5Pset_elink_file_cache_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325 "Sets the number of files that can be held open in an external link open file cache.") sets the number of files that will be held open in an external link open file cache. [H5Pget_elink_file_cache_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4c9bcfff90f48bfefa2c25e551485923 "Retrieves the size of the external link open file cache.") retrieves the size of an existing cache; and [H5Fclear_elink_file_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gafcc153d8606829d4401e93305e5246d7 "Clears the external link open file cache.") clears an existing cache without closing it.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | efc_size | External link open file cache size in number of files _Default setting is 0 (zero)._ 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_elink_file_cache_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325 "Sets the number of files that can be held open in an external link open file cache.") specifies the number of files that will be held open in an external link open file cache.
The default external link open file cache size is 0 (zero), meaning that files accessed via an external link are not held open. Setting the cache size to a positive integer turns on the cache; setting the size back to zero turns it off.
With this property set, files are placed in the external link open file cache cache when they are opened via an external link. Files are then held open until either they are evicted from the cache or the parent file is closed. This property setting can improve performance when external links are repeatedly accessed.
When the cache is full, files will be evicted using a least recently used (LRU) scheme; the file which has gone the longest time without being accessed through the parent file will be evicted and closed if nothing else is holding that file open.
Files opened through external links inherit the parent file's file access property list by default, and therefore inherit the parent file's external link open file cache setting.
When child files contain external links of their own, the caches can form a graph of cached external files. Closing the last external reference to such a graph will recursively close all files in the graph, even if cycles are present.  

Example
    
The following code sets up an external link open file cache that will hold open up to 8 files reached through external links:
status = [H5Pset_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325)(fapl_id, 8);
[H5Pset_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325)
herr_t H5Pset_elink_file_cache_size(hid_t plist_id, unsigned efc_size)
Sets the number of files that can be held open in an external link open file cache. 

Since
    1.8.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaca1f462af9b441a0fdc6d0da6d4e17de)H5Pset_evict_on_close()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_evict_on_close  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | bool |  _evict_on_close_ )  
Controls the library's behavior of evicting metadata associated with a closed object.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | evict_on_close | Whether the HDF5 object should be evicted on close 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The library's metadata cache is fairly conservative about holding on to HDF5 object metadata(object headers, chunk index structures, etc.), which can cause the cache size to grow, resulting in memory pressure on an application or system. When enabled, the "evict on close" property will cause all metadata for an object to be evicted from the cache as long as metadata is not referenced by any other open object.
This function only applies to file access property lists.
The default library behavior is to not evict on object or file close.
When applied to a file access property list, any subsequently opened object will inherit the "evict on close" property and will have its metadata evicted when the object is closed. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6b24e6daf4816bbfb89b63bab40aa982)H5Pset_family_offset()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_family_offset  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _offset_ )  
Sets offset property for low-level access to a file in a family of files.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | offset | Offset in bytes within the HDF5 file 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_family_offset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6b24e6daf4816bbfb89b63bab40aa982 "Sets offset property for low-level access to a file in a family of files.") sets the offset property in the file access property list `fapl_id` so that the user application can retrieve a file handle for low-level access to a particular member of a family of files. The file handle is retrieved with a separate call to [H5Fget_vfd_handle()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae4020a66fb8da0586e3b74c81ffccea4 "Returns pointer to the file handle from the virtual file driver.") (or, in special circumstances, to [H5FDget_vfd_handle()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a9fcfb5d6708f4c3f5d319b801ac252bc); see [HDF5 Virtual File Layer](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html)).
The value of `offset` is an offset in bytes from the beginning of the HDF5 file, identifying a user-determined location within the HDF5 file. The file handle the user application is seeking is for the specific member-file in the associated family of files to which this offset is mapped.
Use of this function is only appropriate for an HDF5 file written as a family of files with the `FAMILY` file driver. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f)H5Pset_fapl_core()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_core  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | size_t |  _increment_ ,   
|  | bool |  _backing_store_ )  
Modifies the file access property list to use the [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | increment | Size, in bytes, of memory increments   
[in] | backing_store | Boolean flag indicating whether to write the file contents to disk when the file is closed  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_core()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.") modifies the file access property list to use the [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) driver.
The [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) driver enables an application to work with a file in memory, speeding reads and writes as no disk access is made. File contents are stored only in memory until the file is closed. The `backing_store` parameter determines whether file contents are ever written to disk.
`increment` specifies the increment by which allocated memory is to be increased each time more memory is required.
While using [H5Fcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") to create a core file, if the `backing_store` is set to 1 (true), the file contents are flushed to a file with the same name as this core file when the file is closed or access to the file is terminated in memory.
The application is allowed to open an existing file with [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) driver. While using [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") to open an existing file, if the `backing_store` is set to 1 (true) and the `flags` for [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") is set to [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), any change to the file contents are saved to the file when the file is closed. If `backing_store` is set to 0 (false) and the `flags` for [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") is set to [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), any change to the file contents will be lost when the file is closed. If the flags for [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") is set to [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), no change to the file is allowed either in memory or on file. 

Note
    Currently this driver cannot create or open family or multi files. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c)H5Pset_fapl_direct()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_direct  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | size_t |  _alignment_ ,   
|  | size_t |  _block_size_ ,   
|  | size_t |  _cbuf_size_ )  
Sets up use of the direct I/O driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | alignment | Required memory alignment boundary   
[in] | block_size | File system block size   
[in] | cbuf_size | Copy buffer size  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_direct()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.") sets the file access property list, `fapl_id`, to use the direct I/O driver, [H5FD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a99213f218f9ab0c51f9c679228a1e436). With this driver, data is written to or read from the file synchronously without being cached by the system.
File systems usually require the data address in memory, the file address, and the size of the data to be aligned. The HDF5 library's direct I/O driver is able to handle unaligned data, though that will consume some additional memory resources and may slow performance. To get better performance, use the system function `posix_memalign` to align the data buffer in memory and the HDF5 function [H5Pset_alignment()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.") to align the data in the file. Be aware, however, that aligned data I/O may cause the HDF5 file to be bigger than the actual data size would otherwise require because the alignment may leave some holes in the file.
`alignment` specifies the required alignment boundary in memory.
`block_size` specifies the file system block size. A value of 0 (zero) means to use HDF5 library's default value of 4KB.
`cbuf_size` specifies the copy buffer size. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d)H5Pset_fapl_family()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_family  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _memb_size_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _memb_fapl_id_ )  
Sets the file access property list to use the family driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | memb_size | Size in bytes of each file member   
[in] | memb_fapl_id | Identifier of file access property list for each family member  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_family()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver.") sets the file access property list identifier, `fapl_id`, to use the family driver.
`memb_size` is the size in bytes of each file member. This size will be saved in file when the property list `fapl_id` is used to create a new file. If `fapl_id` is used to open an existing file, `memb_size` has to be equal to the original size saved in file. A failure with an error message indicating the correct member size will be returned if `memb_size` does not match the size saved. If any user does not know the original size, [H5F_FAMILY_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aeae74ee757cb3e381abf8e3c480c06cc) can be passed in. The library will retrieve the saved size.
`memb_fapl_id` is the identifier of the file access property list to be used for each family member.
The family file driver uses `snprintf` to generate the member file names, passing the member number as an unsigned int. The file name used with the family file driver must therefore contain a single `printf` format specifier that indicates a variable of the correct width and produces unique strings for each member number passed as a parameter to snprintf. For example one might insert `%06d` into the file name string. There must be no other format specifiers in the string.
If this file driver is for the source file of a virtual dataset (VDS) `printf`-style mapping, special care must be taken. In this case the VDS code expands the file name with `snprintf` first, then the family driver second. This means that, while the format specifier for the VDS block number is inserted normally, the format specifier for the family file driver member number must be escaped such that it is only recognized as a format specifier the second time it is run through `snprintf`. As an example one may use `%%06d` as the member file number format specifier in the source file name. 

Version
    1.8.0 Behavior of the `memb_size` parameter was changed.  

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931)H5Pset_fapl_hdfs()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_hdfs  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_hdfs_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html) * |  _fa_ )  
Modifies the file access property list to use the [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | fa | Pointer to [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) driver configuration structure 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_hdfs()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931 "Modifies the file access property list to use the H5FD_HDFS driver.") modifies the file access property list to use the [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) driver. 

Since
    1.10.6 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0)H5Pset_fapl_ioc()
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_ioc  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html) * |  _vfd_config_ )  
Modifies the specified File Access Property List to use the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | vfd_config | Pointer to [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver configuration structure. May be NULL.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0 "Modifies the specified File Access Property List to use the H5FD_IOC driver.") modifies the File Access Property List to use the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver.
The [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver is a reference implementation of an "I/O concentrator" file driver that works in conjunction with the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver and provides the I/O backend for servicing I/O requests to subfiles.
Typically, an HDF5 application won't need to call this routine directly. The [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver is usually set up as a side effect of an HDF5 application using the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver, but this routine is provided in case the application wishes to manually configure the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver. 

Note
    The `vfd_config` parameter may be NULL. In this case, the driver will be setup with default settings. Note that in this case, it is assumed the parent [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver was also setup with default settings. If the two drivers differ in configuration settings, application behavior may not be as expected. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d)H5Pset_fapl_log()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_log  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | const char * |  _logfile_ ,   
|  | unsigned long long |  _flags_ ,   
|  | size_t |  _buf_size_ )  
Sets up the logging virtual file driver ([H5FD_LOG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a027aaf28f5104c77c4f51ecd29a5f7f4)) for use.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | logfile | Name of the log file   
[in] | flags | Flags specifying the types of logging activity   
[in] | buf_size | The size of the logging buffers, in bytes (see description)  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_log()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d "Sets up the logging virtual file driver \(H5FD_LOG\) for use.") modifies the file access property list to use the logging driver, [H5FD_LOG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a027aaf28f5104c77c4f51ecd29a5f7f4). The logging virtual file driver (VFD) is a clone of the standard SEC2 ([H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606)) driver with additional facilities for logging VFD metrics and activity to a file.
`logfile` is the name of the file in which the logging entries are to be recorded.
The actions to be logged are specified in the parameter `flags` using the pre-defined constants described in the following table. Multiple flags can be set through the use of a logical `OR` contained in parentheses. For example, logging read and write locations would be specified as `(H5FD_LOG_LOC_READ|H5FD_LOG_LOC_WRITE)`.
Table1: Logging Flags [H5FD_LOG_LOC_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a120669aefa2b196e1654adc89ec573d1) | Track the location and length of every read, write, or seek operation.   
---|---  
[H5FD_LOG_LOC_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#aad9b1373eda57a82cf4e5d671e1840c1)  
[H5FD_LOG_LOC_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a80fe0af18d00636c3f7b98e9d65ffd21)  
[H5FD_LOG_LOC_IO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a7367b1298e0b4c87fa9465dd28827dca) | Track all I/O locations and lengths. The logical equivalent of the following: `(H5FD_LOG_LOC_READ[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a120669aefa2b196e1654adc89ec573d1) | H5FD_LOG_LOC_WRITE[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#aad9b1373eda57a82cf4e5d671e1840c1) | H5FD_LOG_LOC_SEEK[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a80fe0af18d00636c3f7b98e9d65ffd21))`  
[H5FD_LOG_FILE_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad6950042cea2cd02909798ef461a9684) | Track the number of times each byte is read or written.   
[H5FD_LOG_FILE_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a3953c5550fec9361e233c3d1aac41144)  
[H5FD_LOG_FILE_IO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a494058d10b26c3059d499320f78528b3) | Track the number of times each byte is read and written. The logical equivalent of the following: `(H5FD_LOG_FILE_READ[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad6950042cea2cd02909798ef461a9684) | H5FD_LOG_FILE_WRITE[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a3953c5550fec9361e233c3d1aac41144))`  
[H5FD_LOG_FLAVOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad5592d1612b80582c4440cc2352d29a6) | Track the type, or flavor, of information stored at each byte.   
[H5FD_LOG_NUM_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a98f658e5d600b477b458e2d48dc04cfe) | Track the total number of read, write, seek, or truncate operations that occur.   
[H5FD_LOG_NUM_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a4bb9fb65b42da293cb06b108e09b922a)  
[H5FD_LOG_NUM_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a73258d8eba3fa5819869650dbbd31338)  
[H5FD_LOG_NUM_TRUNCATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a4c5f66077075477d56ad8424b171cbd9)  
[H5FD_LOG_NUM_IO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a73baff79dc4819dc5733aed41a2db676) | Track the total number of all types of I/O operations. The logical equivalent of the following: `(H5FD_LOG_NUM_READ[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a98f658e5d600b477b458e2d48dc04cfe) | H5FD_LOG_NUM_WRITE[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a4bb9fb65b42da293cb06b108e09b922a) | H5FD_LOG_NUM_SEEK[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a73258d8eba3fa5819869650dbbd31338) | H5FD_LOG_NUM_TRUNCATE[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a4c5f66077075477d56ad8424b171cbd9))`  
[H5FD_LOG_TIME_OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ac253b663d693a3fa154a9f70de382e3e) | Track the time spent in open, stat, read, write, seek, or close operations.   
[H5FD_LOG_TIME_STAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a47fd3387519a8fc710cd0797e76fcef9)  
[H5FD_LOG_TIME_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ae243e77f6140dd939c7def71c2d9e1e3)  
[H5FD_LOG_TIME_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a7e3f0ecd0d5065d9031e8da9446442d6)  
[H5FD_LOG_TIME_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a96b366d078d5b93cd5a3b5c25ff3811f)  
[H5FD_LOG_TIME_CLOSE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a1fc445a926ae7cd9af6546166c0af552)  
[H5FD_LOG_TIME_IO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ab2abf6744e67751f6f3236ee6214bfe4) | Track the time spent in each of the above operations. The logical equivalent of the following: `(H5FD_LOG_TIME_OPEN[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ac253b663d693a3fa154a9f70de382e3e) | H5FD_LOG_TIME_STAT[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a47fd3387519a8fc710cd0797e76fcef9) | H5FD_LOG_TIME_READ[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ae243e77f6140dd939c7def71c2d9e1e3) | H5FD_LOG_TIME_WRITE[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a7e3f0ecd0d5065d9031e8da9446442d6) | H5FD_LOG_TIME_SEEK[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a96b366d078d5b93cd5a3b5c25ff3811f) | H5FD_LOG_TIME_CLOSE[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a1fc445a926ae7cd9af6546166c0af552))`  
[H5FD_LOG_ALLOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#afcf6971d3787ddfe911726db7385f6ac) | Track the allocation of space in the file.   
[H5FD_LOG_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ab3ad7b41fb549b5e8e5cd009b3999113) | Track everything. The logical equivalent of the following: `(H5FD_LOG_ALLOC[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#afcf6971d3787ddfe911726db7385f6ac) | H5FD_LOG_TIME_IO[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ab2abf6744e67751f6f3236ee6214bfe4) | H5FD_LOG_NUM_IO[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a73baff79dc4819dc5733aed41a2db676) | H5FD_LOG_FLAVOR[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad5592d1612b80582c4440cc2352d29a6) | H5FD_LOG_FILE_IO[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a494058d10b26c3059d499320f78528b3) | H5FD_LOG_LOC_IO[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a7367b1298e0b4c87fa9465dd28827dca))`  
The logging driver can track the number of times each byte in the file is read from or written to (using [H5FD_LOG_FILE_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad6950042cea2cd02909798ef461a9684) and [H5FD_LOG_FILE_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a3953c5550fec9361e233c3d1aac41144)) and what kind of data is at that location (e.g., metadata, raw data; using [H5FD_LOG_FLAVOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad5592d1612b80582c4440cc2352d29a6)). This information is tracked in internal buffers of size buf_size, which must be at least the maximum size in bytes of the file to be logged while the log driver is in use.  
One buffer of size buf_size will be created for each of [H5FD_LOG_FILE_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad6950042cea2cd02909798ef461a9684), [H5FD_LOG_FILE_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a3953c5550fec9361e233c3d1aac41144) and [H5FD_LOG_FLAVOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad5592d1612b80582c4440cc2352d29a6) when those flags are set; these buffers will not grow as the file increases in size. 

Output:
    This section describes the logging driver (LOG VFD) output.  
The table, immediately below, describes output of the various logging driver flags and function calls. A list of valid flavor values, describing the type of data stored, follows the table.  Table2: Logging Output Flag | VFD Call | Output and Comments   
---|---|---  
[H5FD_LOG_LOC_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a120669aefa2b196e1654adc89ec573d1) | Read  |  `%10a-%10a (%10Zu bytes) (s) Read`  
  
Start position  
End position  
Number of bytes  
Flavor of read  
  
Adds `(%f s)` and seek time if [H5FD_LOG_TIME_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a96b366d078d5b93cd5a3b5c25ff3811f) is also set.   
[H5FD_LOG_LOC_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a120669aefa2b196e1654adc89ec573d1) | Read Error  |  `Error! Reading: %10a-%10a (%10Zu bytes)`  
  
Same parameters as non-error entry.   
[H5FD_LOG_LOC_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#aad9b1373eda57a82cf4e5d671e1840c1) | Write  |  `%10a-%10a (%10Zu bytes) (s) Written`  
  
Start position  
End position  
Number of bytes  
Flavor of write  
  
Adds `(%f s)` and seek time if [H5FD_LOG_TIME_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a96b366d078d5b93cd5a3b5c25ff3811f) is also set.   
[H5FD_LOG_LOC_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#aad9b1373eda57a82cf4e5d671e1840c1) | Write Error  |  `Error! Writing: %10a-%10a (%10Zu bytes)`  
  
Same parameters as non-error entry.   
[H5FD_LOG_LOC_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a80fe0af18d00636c3f7b98e9d65ffd21) | Read, Write  |  `Seek: From %10a-%10a`  
  
Start position  
End position  
  
Adds `(%f s)` and seek time if [H5FD_LOG_TIME_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a96b366d078d5b93cd5a3b5c25ff3811f) is also set.   
[H5FD_LOG_FILE_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad6950042cea2cd02909798ef461a9684) | Close  | Begins with:  
Dumping read I/O information  
  
Then, for each range of identical values, there is this line:  
`Addr %10-%10 (%10lu bytes) read from %3d times`  
  
Start address  
End address  
Number of bytes  
Number of times read  
  
Note: The data buffer is scanned and each range of identical values gets one entry in the log file to save space and make it easier to read.   
[H5FD_LOG_FILE_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a3953c5550fec9361e233c3d1aac41144) | Close  | Begins with:  
Dumping read I/O information  
  
Then, for each range of identical values, there is this line:  
`Addr %10-%10 (%10lu bytes) written to %3d times`  
  
Start address  
End address  
Number of bytes  
Number of times written  
  
Note: The data buffer is scanned and each range of identical values gets one entry in the log file to save space and make it easier to read.   
[H5FD_LOG_FLAVOR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ad5592d1612b80582c4440cc2352d29a6) | Close  | Begins with:  
Dumping I/O flavor information  
  
Then, for each range of identical values, there is this line:  
`Addr %10-%10 (%10lu bytes) flavor is s`  
  
Start address  
End address  
Number of bytes  
Flavor  
  
Note: The data buffer is scanned and each range of identical values gets one entry in the log file to save space and make it easier to read.   
[H5FD_LOG_NUM_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a98f658e5d600b477b458e2d48dc04cfe) | Close  | Total number of read operations: `%11u`  
[H5FD_LOG_NUM_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a4bb9fb65b42da293cb06b108e09b922a) | Close  | Total number of write operations: `%11u`  
[H5FD_LOG_NUM_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a73258d8eba3fa5819869650dbbd31338) | Close  | Total number of seek operations: `%11u`  
[H5FD_LOG_NUM_TRUNCATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a4c5f66077075477d56ad8424b171cbd9) | Close  | Total number of truncate operations: `%11u`  
[H5FD_LOG_TIME_OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ac253b663d693a3fa154a9f70de382e3e) | Open  | Open took: `(%f s)`  
[H5FD_LOG_TIME_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ae243e77f6140dd939c7def71c2d9e1e3) | Close, Read  | Total time in read operations: `%f s`  
  
See also: [H5FD_LOG_LOC_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a120669aefa2b196e1654adc89ec573d1)  
[H5FD_LOG_TIME_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a7e3f0ecd0d5065d9031e8da9446442d6) | Close, Write  | Total time in write operations: `%f s`  
  
See also: [H5FD_LOG_LOC_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#aad9b1373eda57a82cf4e5d671e1840c1)  
[H5FD_LOG_TIME_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a96b366d078d5b93cd5a3b5c25ff3811f) | Close, Read, Write  | Total time in write operations: `%f s`  
  
See also: [H5FD_LOG_LOC_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a80fe0af18d00636c3f7b98e9d65ffd21) or [H5FD_LOG_LOC_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#aad9b1373eda57a82cf4e5d671e1840c1)  
[H5FD_LOG_TIME_CLOSE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a1fc445a926ae7cd9af6546166c0af552) | Close  | Close took: `(%f s)`  
[H5FD_LOG_TIME_STAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a47fd3387519a8fc710cd0797e76fcef9) | Open  | Stat took: `(%f s)`  
[H5FD_LOG_ALLOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#afcf6971d3787ddfe911726db7385f6ac) | Alloc  |  `%10-%10 (%10Hu bytes) (%s) Allocated`  
  
Start of address space  
End of address space  
Total size allocation  
Flavor of allocation  

Flavors:
    The _flavor_ describes the type of stored information. The following table lists the flavors that appear in log output and briefly describes each. These terms are provided here to aid in the construction of log message parsers; a full description is beyond the scope of this document.  Table3: Flavors of logged data Flavor | Description   
---|---  
[H5FD_MEM_NOLIST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a06b02f5b50dcf7e78a94acada33979bb) | Error value   
[H5FD_MEM_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a137331d00cf5b0c84ef7dfa725429f90) | Value not yet set.  
May also be a datatype set in a larger allocation that will be suballocated by the library.   
[H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce) | Superblock data   
[H5FD_MEM_BTREE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a29b8528e16990fbe265682559b917fa3) | B-tree data   
[H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e) | Raw data (for example, contents of a dataset)   
[H5FD_MEM_GHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a020d6245f874e8262058c3278fefe58e) | Global heap data   
[H5FD_MEM_LHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae7536174d3ae2a842a71d6c192b43a13) | Local heap data   
[H5FD_MEM_OHDR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a4337f7056fb57717e82fa1081f496d75) | Object header data  

Version
    1.8.7 The flags parameter has been changed from `unsigned int` to `unsigned long long`. The implementation of the [H5FD_LOG_TIME_OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ac253b663d693a3fa154a9f70de382e3e), [H5FD_LOG_TIME_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#ae243e77f6140dd939c7def71c2d9e1e3), [H5FD_LOG_TIME_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a7e3f0ecd0d5065d9031e8da9446442d6), and [H5FD_LOG_TIME_SEEK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a96b366d078d5b93cd5a3b5c25ff3811f) flags has been finished. New flags were added: [H5FD_LOG_NUM_TRUNCATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a4c5f66077075477d56ad8424b171cbd9) and [H5FD_LOG_TIME_STAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dlog_8h.html#a47fd3387519a8fc710cd0797e76fcef9).       1.6.0 The `verbosity` parameter has been removed. Two new parameters have been added: `flags` of type `unsigned` and `buf_size` of type `size_t`.  

Since
    1.4.0   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf)H5Pset_fapl_mirror()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_mirror  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html) * |  _fa_ )  
Modifies the file access property list to use the [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | fa | Pointer to [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) driver configuration structure 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_mirror()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf "Modifies the file access property list to use the H5FD_MIRROR driver.") modifies the file access property list to use the [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) driver. 

Since
    1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1)H5Pset_fapl_mpio()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_mpio  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | MPI_Comm |  _comm_ ,   
|  | MPI_Info |  _info_ )  
Stores MPI IO communicator information to the file access property list.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | comm | MPI communicator   
[in] | info | MPI info object  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_mpio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.") stores the user-supplied MPI IO parameters `comm`, for communicator, and `info`, for information, in the file access property list `fapl_id`. That property list can then be used to create and/or open a file.
[H5Pset_fapl_mpio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.") is available only in the parallel HDF5 library and is not a collective function.
`comm` is the MPI communicator to be used for file open, as defined in `MPI_File_open` of MPI. This function makes a duplicate of the communicator, so modifications to `comm` after this function call returns have no effect on the file access property list.
`info` is the MPI Info object to be used for file open, as defined in MPI_File_open() of MPI. This function makes a duplicate copy of the Info object, so modifications to the Info object after this function call returns will have no effect on the file access property list.
If the file access property list already contains previously-set communicator and Info values, those values will be replaced and the old communicator and Info object will be freed. 

Note
    Raw dataset chunk caching is not currently supported when using this file driver in read/write mode. All calls to [H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") and [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") will access the disk directly, and [H5Pset_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga034a5fc54d9b05296555544d8dd9fe89 "Sets the raw data chunk cache parameters.") and [H5Pset_chunk_cache()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html#ga104d00442c31714ee073dee518f661f1 "Sets the raw data chunk cache parameters.") will have no effect on performance.  
Raw dataset chunk caching is supported when this driver is used in read-only mode. 

Version
    1.4.5 Handling of the MPI Communicator and Info object changed at this release. A duplicate of each of these is now stored in the property list instead of pointers to each.  

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396)H5Pset_fapl_multi()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_multi  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | const [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) * |  _memb_map_ ,   
|  | const [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) * |  _memb_fapl_ ,   
|  | const char *const * |  _memb_name_ ,   
|  | const [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) * |  _memb_addr_ ,   
|  | bool |  _relax_ )  
Sets up use of the multi-file driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | memb_map | Maps memory usage types to other memory usage types   
[in] | memb_fapl | Property list for each memory usage type   
[in] | memb_name | Name generator for names of member files   
[in] | memb_addr | The offsets within the virtual address space, from 0 (zero) to [HADDR_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a49742d33813ee38ef58eca9fbeda6b86), at which each type of data storage begins   
[in] | relax | Allows read-only access to incomplete file sets when `true` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.") sets the file access property list `fapl_id` to use the multi-file driver.
The multi-file driver enables different types of HDF5 data and metadata to be written to separate files. These files are viewed by the HDF5 library and the application as a single virtual HDF5 file with a single HDF5 file address space. The types of data that can be broken out into separate files include raw data, the superblock, B-tree data, global heap data, local heap data, and object headers. At the programmer's discretion, two or more types of data can be written to the same file while other types of data are written to separate files.
The array `memb_map` maps memory usage types to other memory usage types and is the mechanism that allows the caller to specify how many files are created. The array contains [H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566) entries, which are either the value [H5FD_MEM_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a137331d00cf5b0c84ef7dfa725429f90) or a memory usage type. The number of unique values determines the number of files that are opened.
The array `memb_fapl` contains a property list for each memory usage type that will be associated with a file.
The array `memb_name` should be a name generator (a `printf`-style format with a `%s` which will be replaced with the name passed to [H5FDopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a36c22438fe64e0f9e4dc13aec57da21f), usually from [H5Fcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.")). There must be no other format specifiers in the string.
The array `memb_addr` specifies the offsets within the virtual address space, from 0 (zero) to [HADDR_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a49742d33813ee38ef58eca9fbeda6b86), at which each type of data storage begins.
If `relax` is set to 1 (true), then opening an existing file for read-only access will not fail if some file members are missing. This allows a file to be accessed in a limited sense if just the meta data is available.
Default values for each of the optional arguments are as follows: 
`memb_map` | The default member map contains the value [H5FD_MEM_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a137331d00cf5b0c84ef7dfa725429f90) for each element.   
---|---  
`memb_fapl` | The default value is [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) for each element.   
`memb_name` | The default string is `%s-X.h5` where `X` is one of the following letters:
  * `s` for [H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce)
  * `b` for [H5FD_MEM_BTREE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a29b8528e16990fbe265682559b917fa3)
  * `r` for [H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e)
  * `g` for [H5FD_MEM_GHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a020d6245f874e8262058c3278fefe58e)
  * `l` for [H5FD_MEM_LHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae7536174d3ae2a842a71d6c192b43a13)
  * `o` for [H5FD_MEM_OHDR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a4337f7056fb57717e82fa1081f496d75)

  
`memb_addr` | The default setting is that the address space is equally divided among all of the elements:
  * [H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce) `-> 0 * (HADDR_MAX/6)`
  * [H5FD_MEM_BTREE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a29b8528e16990fbe265682559b917fa3) `-> 1 * (HADDR_MAX/6)`
  * [H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e) `-> 2 * (HADDR_MAX/6)`
  * [H5FD_MEM_GHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a020d6245f874e8262058c3278fefe58e) `-> 3 * (HADDR_MAX/6)`
  * [H5FD_MEM_LHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae7536174d3ae2a842a71d6c192b43a13) `-> 4 * (HADDR_MAX/6)`
  * [H5FD_MEM_OHDR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a4337f7056fb57717e82fa1081f496d75) `-> 5 * (HADDR_MAX/6)`



Example:
    The following code sample sets up a multi-file access property list that partitions data into meta and raw files, each being one-half of the address:  
  
[H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) mt, memb_map[[H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566)];
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) memb_fapl[[H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566)];
const char *memb[[H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566)];
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) memb_addr[[H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566)];
// The mapping...
for (mt=0; mt<[H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566); mt++) {
memb_map[mt] = [H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce);
}
memb_map[[H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e)] = [H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e);
// Member information
memb_fapl[[H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce)] = [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f);
memb_name[[H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce)] = "%s.meta";
memb_addr[[H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce)] = 0;
memb_fapl[[H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e)] = [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f);
memb_name[[H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e)] = "%s.raw";
memb_addr[[H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e)] = [HADDR_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a49742d33813ee38ef58eca9fbeda6b86)/2;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
[H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396)(fapl, memb_map, memb_fapl,
memb_name, memb_addr, true);
[H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b)
enum H5F_mem_t H5FD_mem_t
**Definition** H5FDpublic.h:273
[H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566)
@ H5FD_MEM_NTYPES
**Definition** H5Fpublic.h:163
[H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e)
@ H5FD_MEM_DRAW
**Definition** H5Fpublic.h:158
[H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce)
@ H5FD_MEM_SUPER
**Definition** H5Fpublic.h:156
[HADDR_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a49742d33813ee38ef58eca9fbeda6b86)
#define HADDR_MAX
**Definition** H5public.h:369
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f)
uint64_t haddr_t
**Definition** H5public.h:355
[H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396)
herr_t H5Pset_fapl_multi(hid_t fapl_id, const H5FD_mem_t *memb_map, const hid_t *memb_fapl, const char *const *memb_name, const haddr_t *memb_addr, bool relax)
Sets up use of the multi-file driver. 

Version
    1.6.3 `memb_name` parameter type changed to `const char* const*`.  

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762)H5Pset_fapl_onion()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_onion  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | const [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) * |  _fa_ )  
set the onion info for the file access property list 
* * * 

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | fa | The pointer to the structure [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_onion()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762 "set the onion info for the file access property list") sets the structure [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) for the file access property list that is set for the onion VFD driver. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f)H5Pset_fapl_ros3()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_ros3  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | const [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html) * |  _fa_ )  
Modifies the specified File Access Property List to use the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | fa | Pointer to [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver configuration structure.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

Note
    As of HDF5 2.0.0, as a side effect of calling this function, if the page buffer size has not been set on fapl_id, it is set to 64 MiB. To set a different page buffer size, simply call H5Pset_page_buffer() with fapl_id and your desired page buffer size. To disable the page buffer, call H5Pset_page_buffer() with a size of  
  1. Disabling the page buffer with the ROS3 driver may cause severe performance degradation. 


##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga316f9801b3b0a02916fd4a721532aeda)H5Pset_fapl_ros3_endpoint()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_ros3_endpoint  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | const char * |  _endpoint_ )  
Modifies the specified File Access Property List to use an alternative endpoint URL when opening files with the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | endpoint | Alternate endpoint url.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_ros3_endpoint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga316f9801b3b0a02916fd4a721532aeda "Modifies the specified File Access Property List to use an alternative endpoint URL when opening file...") modifies an existing File Access Property List which is used by [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver by adding or updating the endpoint url of the property list. When not specified, the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver uses the standard s3.<region-code>.amazonaws.com, unless an alternative endpoint URL is specified in the AWS_ENDPOINT_URL_S3 or AWS_ENDPOINT_URL environment variable. Be aware, to set the endpoint first you need to create a proper File Access Property List using H5Pset_fapl_ros() and use this list as input argument of the function [H5Pset_fapl_ros3_endpoint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga316f9801b3b0a02916fd4a721532aeda "Modifies the specified File Access Property List to use an alternative endpoint URL when opening file...").
Note, the endpoint url is only needed when you want to access a S3 bucket using an alternate url. For example, this can be useful when using s3:// object URIs to access files which are located somewhere other than the standard s3.<region-code>.amazonaws.com. 

Since
    2.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga74d2ae62171700a1a50c9fd76713cf4d)H5Pset_fapl_ros3_token()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_ros3_token  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | const char * |  _token_ )  
Modifies the specified File Access Property List to use the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver by adding the specified session/security token.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | token | Session/security token.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_ros3_token()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga74d2ae62171700a1a50c9fd76713cf4d "Modifies the specified File Access Property List to use the H5FD_ROS3 driver by adding the specified ...") modifies an existing File Access Property List which is used by [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver by adding or updating the session/security token of the property list. Be aware, to set the token first you need to create a proper File Access Property List using H5Pset_fapl_ros() and use this list as input argument of the function [H5Pset_fapl_ros3_token()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga74d2ae62171700a1a50c9fd76713cf4d "Modifies the specified File Access Property List to use the H5FD_ROS3 driver by adding the specified ...").
Note, the session token is only needed when you want to access a S3 bucket using temporary security credentials. 

Since
    1.14.2 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db)H5Pset_fapl_sec2()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_sec2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _fapl_id_ | ) |   
---|---|---|---|---|---  
Modifies the file access property list to use the [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) driver.  

Parameters
     [in] | fapl_id | File access property list identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_sec2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db "Modifies the file access property list to use the H5FD_SEC2 driver.") modifies the file access property list to use the [H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) driver. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175)H5Pset_fapl_split()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_split  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_ ,   
---|---|---|---  
|  | const char * |  _meta_ext_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _meta_plist_id_ ,   
|  | const char * |  _raw_ext_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _raw_plist_id_ )  
Emulates the old split file driver.  

Parameters
     [in] | fapl | File access property list identifier   
---|---|---  
[in] | meta_ext | Metadata filename extension   
[in] | meta_plist_id | File access property list identifier for the metadata file   
[in] | raw_ext | Raw data filename extension   
[in] | raw_plist_id |  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_split()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver.") is a compatibility function that enables the multi-file driver to emulate the split driver from HDF5 Releases 1.0 and 1.2. The split file driver stored metadata and raw data in separate files but provided no mechanism for separating types of metadata.
`fapl` is a file access property list identifier.
`meta_ext` is the filename extension for the metadata file. The extension is appended to the name passed to [H5FDopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a36c22438fe64e0f9e4dc13aec57da21f), usually from [H5Fcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file."), to form the name of the metadata file. If the string `%s` is used in the extension, it works like the name generator as in [H5Pset_fapl_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.").
`meta_plist_id` is the file access property list identifier for the metadata file.
`raw_ext` is the filename extension for the raw data file. The extension is appended to the name passed to [H5FDopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a36c22438fe64e0f9e4dc13aec57da21f), usually from [H5Fcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file."), to form the name of the raw data file. If the string `%s` is used in the extension, it works like the name generator as in [H5Pset_fapl_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver."). There must be no other format specifiers in the string.
`raw_plist_id` is the file access property list identifier for the raw data file.
If a user wishes to check to see whether this driver is in use, the user must call [H5Pget_driver()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") and compare the returned value to the string [H5FD_MULTI](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmulti_8h.html#a754e05ae5e0f2d86f64002b338c0fd5c). A positive match will confirm that the multi driver is in use; HDF5 provides no mechanism to determine whether it was called as the special case invoked by [H5Pset_fapl_split()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver."). 

Example:
    
// Example 1: Both metadata and raw data files are in the same
// directory. Use Station1-m.h5 and Station1-r.h5 as
// the metadata and raw data files.
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl, fid;
fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
[H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175)(fapl, "-m.h5", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), "-r.h5", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
fid=[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("Station1",[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a),[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f),fapl);
// Example 2: metadata and raw data files are in different
// directories. Use PointA-m.h5 and /pfs/PointA-r.h5 as
// the metadata and raw data files.
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl, fid;
fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
[H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175)(fapl, "-m.h5", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), "/pfs/%s-r.h5", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
fid=[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("PointA",[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a),[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f),fapl);
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175)
herr_t H5Pset_fapl_split(hid_t fapl, const char *meta_ext, hid_t meta_plist_id, const char *raw_ext, hid_t raw_plist_id)
Emulates the old split file driver.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5)H5Pset_fapl_splitter()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_splitter  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_splitter_vfd_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__splitter__vfd__config__t.html) * |  _config_ptr_ )  
Sets the file access property list to use the splitter driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | config_ptr | Configuration options for the VFD  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_splitter()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5 "Sets the file access property list to use the splitter driver.") sets the file access property list identifier, `fapl_id`, to use the splitter driver.
The splitter VFD echoes file manipulation (e.g. create, truncate) and write calls to a second, write-only file. 

Note
    The splitter VFD should not be confused with the split VFD, which is a simplification of the multi VFD and creates separate files for metadata and data. 

Since
    1.12.1, back-ported to 1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052)H5Pset_fapl_stdio()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_stdio  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _fapl_id_ | ) |   
---|---|---|---|---|---  
Sets the standard I/O driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_stdio()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052 "Sets the standard I/O driver.") modifies the file access property list to use the stdio VFD, which uses I/O calls from stdio.h. 

Note
    This VFD was designed to be a "demo" VFD that shows how to write your own VFD. Most applications should not use this VFD and should instead use the POSIX I/O VFD (sec2). 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2)H5Pset_fapl_subfiling()
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_subfiling  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | const [H5FD_subfiling_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__subfiling__config__t.html) * |  _vfd_config_ )  
Modifies the specified File Access Property List to use the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | vfd_config | Pointer to [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver configuration structure. May be NULL.  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_subfiling()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver.") modifies the File Access Property List to use the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver.
The [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver is an MPI-based file driver that allows an HDF5 application to distribute a logical HDF5 file across a collection of "subfiles" in equal-sized data segment "stripes". I/O to the logical HDF5 file is then directed to the appropriate "subfile" according to the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) configuration and a system of I/O concentrators, which are MPI ranks operating worker threads.
By allowing a configurable stripe size, number of I/O concentrators and method for selecting MPI ranks as I/O concentrators, the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver aims to enable an HDF5 application to find a middle ground between the single shared file and file-per-process approaches to parallel file I/O for the particular machine the application is running on. In general, the goal is to avoid some of the complexity of the file-per-process approach while also minimizing the locking issues of the single shared file approach on a parallel file system. 

Note
    Since the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver is an MPI-based file driver, the HDF5 application should ensure that [H5Pset_mpi_params()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6daceb4a9e51fca7cb198f964b67baf0 "Set the MPI communicator and info.") is called before this routine so that the appropriate MPI communicator and info objects will be setup for use by the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) and [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) drivers.      The current architecture of the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver requires that the HDF5 application must have been initialized with MPI_Init_thread() using a value of MPI_THREAD_MULTIPLE for the thread support level.      The `vfd_config` parameter may be NULL. In this case, the reference implementation I/O concentrator VFD will be used with the default settings of one I/O concentrator per node and a stripe size of 32MiB. Refer to the [H5FD_subfiling_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__subfiling__config__t.html "Configuration structure for H5Pset_fapl_subfiling\(\) / H5Pget_fapl_subfiling\(\)") documentation for information about configuration for the [H5FD_SUBFILING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsubfiling_8h.html#a070ba7b51cfe718ba4da75b308066a9d) driver. 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e)H5Pset_fapl_windows()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fapl_windows  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _fapl_id_ | ) |   
---|---|---|---|---|---  
Sets the Windows I/O driver.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fapl_windows()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e "Sets the Windows I/O driver.") sets the default HDF5 Windows I/O driver on Windows systems.
Since the HDF5 library uses this driver, [H5FD_WINDOWS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dwindows_8h.html#ab5173993ddefd103bfb3d37c2837a9a4), by default on Windows systems, it is not normally necessary for a user application to call [H5Pset_fapl_windows()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e "Sets the Windows I/O driver."). While it is not recommended, there may be times when a user chooses to set a different HDF5 driver, such as the standard I/O driver ([H5FD_STDIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dstdio_8h.html#a030a03b96a9f6e46035ce64e25389085)) or the sec2 driver ([H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606)), in a Windows application. [H5Pset_fapl_windows()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gad25acfdce018e6f8459d890b3725062e "Sets the Windows I/O driver.") is provided so that the application can return to the Windows I/O driver when the time comes.
Only the Windows driver is tested on Windows systems; other drivers are used at the application's and the user's risk.
Furthermore, the Windows driver is tested and available only on Windows systems; it is not available on non-Windows systems. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c)H5Pset_fclose_degree()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_fclose_degree  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | [H5F_close_degree_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475f) |  _degree_ )  
Sets the file close degree.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | degree | Pointer to a location containing the file close degree property, the value of `degree` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_fclose_degree()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c "Sets the file close degree.") sets the file close degree property `degree` in the file access property list `fapl_id`.
The value of `degree` determines how aggressively [H5Fclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") deals with objects within a file that remain open when [H5Fclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") is called to close that file. `degree` can have any one of four valid values:
Degree name  | H5Fclose behavior with no open object in file  | H5Fclose behavior with open object(s) in file   
---|---|---  
[H5F_CLOSE_WEAK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475faea1127311a219b44e4af3cb12609035f) | Actual file is closed.  | Access to file identifier is terminated; actual file close is delayed until all objects in file are closed   
[H5F_CLOSE_SEMI](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475fa2d0bd1af3f7a3e287b42d773a2c01001) | Actual file is closed.  | Function returns FAILURE   
[H5F_CLOSE_STRONG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475fae6af53249bfe320745828497f28b6390) | Actual file is closed.  | All open objects remaining in the file are closed then file is closed   
[H5F_CLOSE_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475fa8f4acef5a05a854c636ce72c3dc244c7) | The VFL driver chooses the behavior. Currently, all VFL drivers set this value to [H5F_CLOSE_WEAK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475faea1127311a219b44e4af3cb12609035f), except for the MPI-I/O driver, which sets it to [H5F_CLOSE_SEMI](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475fa2d0bd1af3f7a3e287b42d773a2c01001).  |  

Warning
    If a file is opened multiple times without being closed, each open operation must use the same file close degree setting. For example, if a file is already open with [H5F_CLOSE_WEAK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475faea1127311a219b44e4af3cb12609035f), an [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") call with [H5F_CLOSE_STRONG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475fae6af53249bfe320745828497f28b6390) will fail. 

Since
    1.6.0   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2)H5Pset_file_image()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_file_image  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | void * |  _buf_ptr_ ,   
|  | size_t |  _buf_len_ )  
Sets an initial file image in a memory buffer.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | buf_ptr | Pointer to the initial file image, or NULL if no initial file image is desired   
[in] | buf_len | Size of the supplied buffer, or 0 (zero) if no initial image is desired 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer.") allows an application to provide a file image to be used as the initial contents of a file. Calling [H5Pset_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer.")makes a copy of the buffer specified in `buf_ptr` of size `buf_len`. 

Motivation:
     [H5Pset_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer.") and other elements of HDF5 are used to load an image of an HDF5 file into system memory and open that image as a regular HDF5 file. An application can then use the file without the overhead of disk I/O. 

Recommended Reading:
    This function is part of the file image operations feature set. It is highly recommended to study the guide [HDF5 File Image](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_i_m__u_g.html) before using this feature set. See the “See Also” section below for links to other elements of HDF5 file image operations. 

See also
    
  * [H5LTopen_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae2834cca05d6c4787f51933cc521f8fb "Opens an HDF5 file image in memory.")
  * [H5Fget_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gadc53f4e76b1199cb5d2a8cb7fbb114ad "Retrieves a copy of the image of an existing, open file.")
  * [H5Pget_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga337626cc516d5d1e3303ea6bc350e56b "Retrieves a copy of the file image designated as the initial content and structure of a file.")
  * [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.")
  * [H5Pget_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gae17e38082dfdbadd75c897f1e6a9096e "Retrieves callback routines for working with file images.")


  * [HDF5 File Image](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_i_m__u_g.html)


  * Within [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images."): 
  * Callback [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html)
  * Callback [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd)



Version
    1.8.13 Fortran subroutine added in this release.  

Since
    1.8.9 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c)H5Pset_file_image_callbacks()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_file_image_callbacks  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  |  [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) * |  _callbacks_ptr_ )  
Sets the callbacks for working with file images.  

Note
     **Motivation:** [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") and other elements of HDF5 are used to load an image of an HDF5 file into system memory and open that image as a regular HDF5 file. An application can then use the file without the overhead of disk I/O.  
**Recommended Reading:** This function is part of the file image operations feature set. It is highly recommended to study the guide [HDF5 File Image](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_i_m__u_g.html) before using this feature set. See the “See Also” section below for links to other elements of HDF5 file image operations. 

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in,out] | callbacks_ptr | Pointer to the instance of the [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) structure 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.   
**Failure Modes** : Due to interactions between this function and [H5Pset_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga31d0299f6ad287e013b2a02a8ccc1fa2 "Sets an initial file image in a memory buffer.") and [H5Pget_file_image()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga337626cc516d5d1e3303ea6bc350e56b "Retrieves a copy of the file image designated as the initial content and structure of a file."), [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") will fail if a file image has already been set in the target file access property list, `fapl_id`.  
[H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") sets callback functions for working with file images in memory.
[H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") allows an application to control the management of file image buffers through user defined callbacks. These callbacks can be used in the management of file image buffers in property lists and with certain file drivers.
[H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") must be used before any file image has been set in the file access property list. Once a file image has been set, the function will fail.
The callback routines set up by [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") are invoked when a new file image buffer is allocated, when an existing file image buffer is copied or resized, or when a file image buffer is released from use.
Some file drivers allow the use of user-defined callback functions for allocating, freeing, and copying the driver's internal buffer, potentially allowing optimizations such as avoiding large `malloc` and `memcpy` operations, or to perform detailed logging.
From the perspective of the HDF5 library, the operations of the [image_malloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a81d31063c72da7bfcb2e75aeda9cad32), [image_memcpy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4562078dc2326b0315c410d733d38030), [image_realloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4b550f819daa4e0553c2272de4d40f5d), and [image_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a7e81e4662ea07b93e9380f5f78fee890) callbacks must be identical to those of the corresponding C standard library calls (`malloc`, `memcpy`, `realloc`, and `free`). While the operations must be identical, the file image callbacks have more parameters. The return values of [image_malloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a81d31063c72da7bfcb2e75aeda9cad32) and [image_realloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4b550f819daa4e0553c2272de4d40f5d) are identical to the return values of `malloc` and `realloc`. The return values of [image_malloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a81d31063c72da7bfcb2e75aeda9cad32) and [image_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a7e81e4662ea07b93e9380f5f78fee890) differ from the return values of `memcpy` and `free` in that the return values of [image_memcpy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4562078dc2326b0315c410d733d38030) and [image_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a7e81e4662ea07b93e9380f5f78fee890) can also indicate failure.
The callbacks and their parameters, along with a struct and an `ENUM` required for their use, are described below.
**Callback struct and`ENUM:`**
The callback functions set up by [H5Pset_file_image_callbacks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga14ea3598215afd078b964b672b40d63c "Sets the callbacks for working with file images.") use a struct and an `ENUM` that are defined as follows
The struct [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) serves as a container for the callback functions and a pointer to user-supplied data. The struct is defined as follows: 
typedef struct {
void *(*image_malloc)(size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *udata);
void *(*image_memcpy)(void *dest, const void *src, size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op,
void *udata);
void *(*image_realloc)(void *ptr, size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *udata);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*image_free)(void *ptr, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *udata);
void *(*udata_copy)(void *udata);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*udata_free)(void *udata);
void *udata;
} [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html);
Elements of the [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) are used by the callbacks to invoke certain operations on file images. The ENUM is defined as follows: 
typedef enum {
[H5FD_FILE_IMAGE_OP_NO_OP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dda275b53cff617478e11e382e9059353e5),
[H5FD_FILE_IMAGE_OP_PROPERTY_LIST_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008ddad3f1a08d78d24f99705bc5fc8a249123),
[H5FD_FILE_IMAGE_OP_PROPERTY_LIST_COPY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008ddacb33fad1a4edf0f8e84d79fe026dcccc),
[H5FD_FILE_IMAGE_OP_PROPERTY_LIST_GET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dda96e2ba60483056e7723abc467ce247c7),
[H5FD_FILE_IMAGE_OP_PROPERTY_LIST_CLOSE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dda17c03c08f4185a5a2c40be82d9795356),
[H5FD_FILE_IMAGE_OP_FILE_OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dda4938a7bef146b2185c711e7fbb8df2cc),
[H5FD_FILE_IMAGE_OP_FILE_RESIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dda5f979f4e4af545e6aba40f9b2af14caf),
[H5FD_FILE_IMAGE_OP_FILE_CLOSE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dda9dd8cfa5ee60537d396c98e8d0646f65)
} [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd);
The elements of the [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) are used in the following callbacks:
  * The [image_malloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a81d31063c72da7bfcb2e75aeda9cad32) callback contains a pointer to a function that must appear to HDF5 to have functionality identical to that of the standard C library `malloc()` call.
  * Signature in [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html): 
void *(*image_malloc)(size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *udata);
  

  * The [image_memcpy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4562078dc2326b0315c410d733d38030) callback contains a pointer to a function that must appear to HDF5 to have functionality identical to that of the standard C library `memcopy()` call, except that it returns a `NULL` on failure. (The `memcpy` C Library routine is defined to return the `dest` parameter in all cases.)
  * Setting [image_memcpy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4562078dc2326b0315c410d733d38030) to `NULL` indicates that HDF5 should invoke the standard C library `memcpy()` routine when copying buffers.
  * Signature in [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html): 
void *(*image_memcpy)(void *dest, const void *src, size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op,
void *udata);
  

  * The [image_realloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4b550f819daa4e0553c2272de4d40f5d) callback contains a pointer to a function that must appear to HDF5 to have functionality identical to that of the standard C library `realloc()` call.
  * Setting [image_realloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4b550f819daa4e0553c2272de4d40f5d) to `NULL` indicates that HDF5 should invoke the standard C library `realloc()` routine when resizing file image buffers.
  * Signature in [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html): 
void *(*image_realloc)(void *ptr, size_t size, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *udata);
  

  * The [image_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a7e81e4662ea07b93e9380f5f78fee890) callback contains a pointer to a function that must appear to HDF5 to have functionality identical to that of the standard C library `free()` call, except that it will return `0` (`SUCCEED`) on success and `-1` (`FAIL`) on failure.
  * Setting [image_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a7e81e4662ea07b93e9380f5f78fee890) to `NULL` indicates that HDF5 should invoke the standard C library `free()` routine when releasing file image buffers.
  * Signature in [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html): 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*image_free)(void *ptr, [H5FD_file_image_op_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#aa95ee1806ea4db9f035cd53844c008dd) file_image_op, void *udata);
  

  * The [udata_copy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a20cdfcd242ef041c748f47f8de39774d) callback contains a pointer to a function that, from the perspective of HDF5, allocates a buffer of suitable size, copies the contents of the supplied `udata` into the new buffer, and returns the address of the new buffer. The function returns NULL on failure. This function is necessary if a non-NULL `udata` parameter is supplied, so that property lists containing the image callbacks can be copied. If the `udata` parameter below is `NULL`, then this parameter should be `NULL` as well.
  * Signature in [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html): 
void *(*udata_copy)(void *udata);
  

  * The [udata_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a72907de159ae0b35408bbfe75f7c3359) callback contains a pointer to a function that, from the perspective of HDF5, frees a user data block. This function is necessary if a non-NULL udata parameter is supplied so that property lists containing image callbacks can be discarded without a memory leak. If the udata parameter below is `NULL`, this parameter should be `NULL` as well.
  * Signature in [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html): 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*udata_free)(void *udata);
  * `**udata**`, the final field in the [H5FD_file_image_callbacks_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html) struct, provides a pointer to user-defined data. This pointer will be passed to the [image_malloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a81d31063c72da7bfcb2e75aeda9cad32), [image_memcpy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4562078dc2326b0315c410d733d38030), [image_realloc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a4b550f819daa4e0553c2272de4d40f5d), and [image_free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__file__image__callbacks__t.html#a7e81e4662ea07b93e9380f5f78fee890) callbacks. Define udata as `NULL` if no user-defined data is provided.



Since
    1.8.9 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaae4de80c533a38da9f269c02b8e71de2)H5Pset_file_locking()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_file_locking  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | bool |  _use_file_locking_ ,   
|  | bool |  _ignore_when_disabled_ )  
Sets the file locking property values.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | use_file_locking | Toggle to specify file locking (or not)   
[in] | ignore_when_disabled | Toggle to ignore when disabled (or not) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_file_locking()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaae4de80c533a38da9f269c02b8e71de2 "Sets the file locking property values.") overrides the default file locking flag setting that was set when the library was configured.
This setting can be overridden by the `HDF5_USE_FILE_LOCKING` environment variable.
File locking is used when creating/opening a file to prevent problematic file accesses. 

Since
    1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga61f01a12d5392ccf1321168f3c28f36f)H5Pset_gc_references()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_gc_references  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | unsigned |  _gc_ref_ )  
Sets garbage collecting references flag.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | gc_ref | Flag setting reference garbage collection to on (1) or off (0) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_gc_references()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga61f01a12d5392ccf1321168f3c28f36f "Sets garbage collecting references flag.") sets the flag for garbage collecting references for the file.
Dataset region references and other reference types use space in an HDF5 file's global heap. If garbage collection is on and the user passes in an uninitialized value in a reference structure, the heap might get corrupted. When garbage collection is off, however, and the user reuses a reference, the previous heap block will be orphaned and not returned to the free heap space.
When garbage collection is on, the user must initialize the reference structures to 0 or risk heap corruption.
The default value for garbage collecting references is off. 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)H5Pset_libver_bounds()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_libver_bounds  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) |  _low_ ,   
|  | [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) |  _high_ )  
Controls the range of library release versions used when creating objects in a file.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | low | The earliest version of the library that will be used for writing objects   
[in] | high | The latest version of the library that will be used for writing objects 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_libver_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.") controls the range of library release versions that will be used when creating objects in a file. The object format versions are determined indirectly from the library release versions specified in the call.
This property is set in the file access property list specified by the parameter `fapl_id`.
The parameter `low` sets the earliest possible format versions that the library will use when creating objects in the file. Note that earliest possible is different from earliest, as some features introduced in library versions later than 1.0.0 resulted in updates to object formats. The parameter `high` sets the latest format versions that the library will be allowed to use when creating objects in the file.
The parameters `low` and `high` must be one of the enumerated values in the [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2) struct, which is defined in [H5Fpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html).
[H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0) is equivalent to the highest explicitly numbered API value in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2), indicating that this is currently the latest format available.
The library supports the following pairs of (`low`, `high`) combinations as derived from the values in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2):
Value of `low` and `high` | Result   
---|---  
`low=H5F_LIBVER_EARLIEST[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2abed98059b4a02d048b1eb3985fba5fa1)`   
`high=<any` other version but not [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)> | 
  * The library will create objects with the earliest possible format versions. 
  * The library will allow objects to be created with the latest format versions available to library release specified in the `high` value. 
  * API calls that create objects or features that are available to versions of the library greater than the specified version in `high` will fail. 

  
`low=H5F_LIBVER_V18[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8)`   
`high=<any` version higher than `low` but not [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)> | 
  * The library will create objects with the latest format versions available to library release 1.8.x. 
  * The library will allow objects to be created with the latest format versions available to library release specified in the `high` value. 
  * API calls that create objects or features that are available to versions of the library greater than the specified version in `high` will fail. 
  * Earlier versions of the library may not be able to access objects created with this setting. 

  
`low=H5F_LIBVER_V110[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a33c6cdc401a3a32dbf63d74019fad4b3)`   
`high=<any` version higher than `low` but not [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)> | 
  * The library will create objects with the latest format versions available to library release 1.10.x. 
  * The library will allow objects to be created with the latest format versions available to library release specified in the `high` value. 
  * API calls that create objects or features that are available to versions of the library greater than version specified in `high` will fail. 
  * Earlier versions of the library may not be able to access objects created with this setting. 

  
`low=H5F_LIBVER_V112[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2ac1b80c0874a79934aef1f7f44863853b)`   
`high=<any` version higher than `low` but not [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)> | 
  * The library will create objects with the latest format versions available to library release 1.12.x. 
  * The library will allow objects to be created with the latest format versions available to library release specified in the `high` value. 
  * API calls that create objects or features that are available to versions of the library greater than version specified in `high` will fail. 
  * Earlier versions of the library may not be able to access objects created with this setting. 

  
`low=H5F_LIBVER_V114[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a9ecfa96a6f297e218c21c715312d74de)`   
`high=<any` version higher than `low` but not [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)> | 
  * The library will create objects with the latest format versions available to library release 1.14.x. 
  * The library will allow objects to be created with the latest format versions available to library release specified in the `high` value. 
  * API calls that create objects or features that are available to versions of the library greater than version specified in `high` will fail. 
  * Earlier versions of the library may not be able to access objects created with this setting. 

  
`low=H5F_LIBVER_V200[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a7e8e06822de9c57c31c7dd404520974d)`   
`high=<any` version higher than `low` but not [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)> | 
  * The library will create objects with the latest format versions available to library release 2.0.x. 
  * The library will allow objects to be created with the latest format versions available to library release specified in the `high` value. 
  * API calls that create objects or features that are available to versions of the library greater than version specified in `high` will fail. 
  * Earlier versions of the library may not be able to access objects created with this setting. 

  
`low=high` | 
  * The library will create objects with the latest format versions available to library release specified. 
  * The objects written with this setting may be accessible to a smaller range of library versions than would be the case if low is set to [H5F_LIBVER_EARLIEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2abed98059b4a02d048b1eb3985fba5fa1). 
  * API calls that create objects or features that are available to versions of the library greater than the specified release will fail. 
  * Earlier versions of the library may not be able to access objects created with this setting. 

  
`low=H5F_LIBVER_EARLIEST[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2abed98059b4a02d048b1eb3985fba5fa1)`   
`high=H5F_LIBVER_LATEST[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)` | 
  * The library will create objects with the earliest possible format versions. 
  * The library will allow objects to be created with the latest format versions available to the latest release. See note [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0) below the table. 
  * This is the library default setting and provides the greatest format compatibility. 

  
`low=<any` version lower than `high>`   
`high=H5F_LIBVER_LATEST[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)` | 
  * The library will create objects with the latest format versions available to library release `low`. 
  * The library will allow objects to be created with the latest format versions available to the latest release. See note _H5F_LIBVER_LATEST_ below the table. 
  * This setting allows users to take advantage of the latest features and performance enhancements in the library. 
  * Earlier versions of the library may not be able to access objects created with this setting. 

  
`low=H5F_LIBVER_LATEST[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)`   
`high=H5F_LIBVER_LATEST[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)` | 
  * The library will create objects with the latest format versions available to the latest release. 
  * The library will allow objects to be created with the latest format versions available to the latest release. See note _H5F_LIBVER_LATEST_ below the table. 
  * This setting allows users to take advantage of the latest features and performance enhancements in the library. However, objects written with this setting may be accessible to a smaller range of library versions than would be the case if low is set to [H5F_LIBVER_EARLIEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2abed98059b4a02d048b1eb3985fba5fa1). 
  * Earlier versions of the library may not be able to access objects created with this setting. 

  
The default settings are `low=H5F_LIBVER_V18[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8)`, `high=H5F_LIBVER_LATEST[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)`. 

Note
     _H5F_LIBVER_LATEST_ :  
Since 2.0.x is also [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0), there is no upper limit on the format versions to use. That is, if a newer format version is required to support a feature in 2.0.x series, this setting will allow the object to be created. 

Version
    2.0.0 Default setting for `low` changed to [H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8)      1.10.2 [H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8) added to the enumerated defines in [H5F_libver_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaf234199ad4cf9c708f45893f7f9cd4d3)H5Pset_mdc_config()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_mdc_config  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) * |  _config_ptr_ )  
Set the initial metadata cache configuration in the indicated File Access Property List to the supplied value.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | config_ptr | Pointer to the instance of `H5AC_cache_config_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html)` containing the desired configuration  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The fields of the [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) structure are shown below: 
typedef struct [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html) {
/* general configuration fields: */
int [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aad880fc4455c253781e8968f2239d56f);
bool [rpt_fcn_enabled](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af6bef1101778934a0f7d54963db4a9f5);
bool [open_trace_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ae92fb5b442027a53da62a853b9be7636);
bool [close_trace_file](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ab047662ba4d5dd6eb32835b472c38fb0);
char [trace_file_name](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a92e7d20eb2b7b353961c64558ddac080)[[H5AC__MAX_TRACE_FILE_NAME_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a_cpublic_8h.html#a717f1f3545cfc3d1b2208c96cc0c3bd3) + 1];
bool [evictions_enabled](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a1b7d261ea3c2cd28446253e8f3b67e0a);
bool [set_initial_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a5bfa4ced2bee62567910bb14dd28265b);
size_t [initial_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a649236e7dd714855a50f122aa5caca9f);
double [min_clean_fraction](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#abd805b98f873c1720f34a0ce937838fd);
size_t [max_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af4728438dee601cb2554d9bf18d78a43);
size_t [min_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#af99ca22b80e05fd5b3603806348ab647);
long int [epoch_length](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ac998e51b01e0eef09d9a29c43f97e4bf);
/* size increase control fields: */
enum [H5C_cache_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#a040d488146ff1ca0a82209e9af3918fa) [incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ae825aaf759060239e92170d20eb97d26);
double [lower_hr_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a20f323fcb4747fc7228d2d74bb965586);
double [increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ac504dff76b24ab9f15536c51aec9fbbb);
bool [apply_max_increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#aef1ae6503a0ae34abacce69d2996628e);
size_t [max_increment](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#ad5a729f1d611f2780679a35b3524052c);
enum [H5C_cache_flash_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#aaaa13ca7756d135b7df6d5a6779ee908) [flash_incr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a0e25a1dc2c695bea335df0e23ed6363c);
double [flash_multiple](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a77b1812e0407c9122db524462a5c9633);
double [flash_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a95fb1e03a77ef5c109d0c851416ced55);
/* size decrease control fields: */
enum [H5C_cache_decr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_cpublic_8h.html#a4f8534794ad9a977185a5d608c0af04f) [decr_mode](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a5df68196b281c19d8ab7da0788566aec);
double [upper_hr_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a84a5ff4ac69196aa27c14f6f796db596);
double [decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a54007d3f2afb718b437f499a5c8b46d9);
bool [apply_max_decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a7ef00e6aab4e781e1e9b1344b3e23460);
size_t [max_decrement](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a75e875a61c9da7f82482d0f6fe6e7152);
int [epochs_before_eviction](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a7ac41e345300bdecd9943e855d55b71b);
bool [apply_empty_reserve](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a14cfe890a5a58b493084730c988aa4ec);
double [empty_reserve](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a9c1ae995513b55737aad09e11beff733);
/* parallel configuration fields: */
size_t [dirty_bytes_threshold](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a8e3c2a2d300b7a8f8d3705fc5e59a3c1);
int [metadata_write_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html#a83a536128dbb7785b2553c294f33d1fe);
} [H5AC_cache_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__config__t.html);
(Click on a enumerator, field, or type for more information.)
[H5Pset_mdc_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaf234199ad4cf9c708f45893f7f9cd4d3 "Set the initial metadata cache configuration in the indicated File Access Property List to the suppli...") attempts to set the initial metadata cache configuration to the supplied value. It will fail if an invalid configuration is detected. This configuration is used when the file is opened.
See the overview of the metadata cache in the special topics section of the user manual for details on what is being configured. If you have not read and understood that documentation, you really should not be using this API call. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65cf9fea33d1324009efc2d5db848434)H5Pset_mdc_image_config()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_mdc_image_config  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html) * |  _config_ptr_ )  
Sets the metadata cache image option for a file access property list.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[out] | config_ptr | Pointer to metadata cache image configuration values  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_mdc_image_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65cf9fea33d1324009efc2d5db848434 "Sets the metadata cache image option for a file access property list.") sets the metadata cache image option with configuration values specified by `config_ptr` for the file access property list specified in `plist_id`.
[H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html) is defined as follows: 
typedef struct [H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html) {
int [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aad880fc4455c253781e8968f2239d56f);
bool [generate_image](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aa1b45b71d5b59abae823123df7678fc7);
bool [save_resize_status](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aa8b1987de42d9297f927db576c72271f);
int [entry_ageout](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aec92d40c46311615f2155573aca27ec4);
} [H5AC_cache_image_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html);
(Click on a enumerator, field, or type for more information.) 

Limitations: While it is an obvious error to request a cache image when
    opening the file read only, it is not in general possible to test for this error in the [H5Pset_mdc_image_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65cf9fea33d1324009efc2d5db848434 "Sets the metadata cache image option for a file access property list.") call. Rather than fail the subsequent file open, the library silently ignores the file image request in this case.  
It is also an error to request a cache image on a file that does not support superblock extension messages (i.e. a superblock version less than 2). As above, it is not always possible to detect this error in the [H5Pset_mdc_image_config()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65cf9fea33d1324009efc2d5db848434 "Sets the metadata cache image option for a file access property list.") call, and thus the request for a cache image will fail silently in this case as well.  
Creation of cache images is currently disabled in parallel – as above, any request for a cache image in this context will fail silently.  
Files with cache images may be read in parallel applications, but note that the load of the cache image is a collective operation triggered by the first operation that accesses metadata after file open (or, if persistent free space managers are enabled, on the first allocation or deallocation of file space, or read of file space manager status, whichever comes first). Thus the parallel process may deadlock if any process does not participate in this access.  
In long sequences of file closes and opens, infrequently accessed metadata can accumulate in the cache image to the point where the cost of storing and restoring this metadata exceeds the benefit of retaining frequently used metadata in the cache image. When implemented, the [H5AC_cache_image_config_t::entry_ageout](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a_c__cache__image__config__t.html#aec92d40c46311615f2155573aca27ec4) should address this problem. In the interim, not requesting a cache image every n file close/open cycles may be an acceptable work around. The choice of `n` will be driven by application behavior, but `n = 10` seems a good starting point. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef)H5Pset_mdc_log_options()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_mdc_log_options  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | bool |  _is_enabled_ ,   
|  | const char * |  _location_ ,   
|  | bool |  _start_on_access_ )  
Sets metadata cache logging options.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | is_enabled | Whether logging is enabled   
[in] | location | Location of log in UTF-8/ASCII (file path/name) (On Windows, this must be ASCII)   
[in] | start_on_access | Whether the logging will begin as soon as the file is opened or created 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The metadata cache is a central part of the HDF5 library through which all file metadata reads and writes take place. File metadata is normally invisible to the user and is used by the library for purposes such as locating and indexing data. File metadata should not be confused with user metadata, which consists of attributes created by users and attached to HDF5 objects such as datasets via H5A API calls.
Due to the complexity of the cache, a trace/logging feature has been created that can be used by HDF5 developers for debugging and performance analysis. The functions that control this functionality will normally be of use to a very limited number of developers outside of The HDF Group. The functions have been documented to help users create logs that can be sent with bug reports.
Control of the log functionality is straightforward. Logging is enabled via the [H5Pset_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga33ef5eda9cd47bfe827ea16d835a88ef "Sets metadata cache logging options.") function, which will modify the file access property list used to open or create a file. This function has a flag that determines whether logging begins at file open or starts in a paused state. Log messages can then be controlled via the [H5Fstart_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga378fb5863071278b47070cf205f53e67 "Starts logging metadata cache events if logging was previously enabled.") and [H5Fstop_mdc_logging()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga78627b23010f82002b837f4d312bf234 "Stops logging metadata cache events if logging was previously enabled and is currently ongoing.") function.
[H5Pget_mdc_log_options()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5da8039847bcbf091e92138b84c97374 "Gets metadata cache logging options.") can be used to examine a file access property list, and [H5Fget_mdc_logging_status()](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_d_c.html#ga5d1d9a5b7df6e6fb92700bacb4c917d6 "Gets the current metadata cache logging status.") will return the current state of the logging flags.
The log format is described in [_Metadata Cache Logging_] (<https://support.hdfgroup.org/releases/hdf5/documentation/advanced_topics/FineTuningMetadataCache.md>). 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1)H5Pset_meta_block_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_meta_block_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _size_ )  
Sets the minimum metadata block size.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | size | Minimum size, in bytes, of metadata block allocations 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_meta_block_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8822e3dedc8e1414f20871a87d533cb1 "Sets the minimum metadata block size.") sets the minimum size, in bytes, of metadata block allocations when [H5FD_FEAT_AGGREGATE_METADATA](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a679a56f7b84eba3ce4cf116666997e97) is set by a VFL driver.
Each raw metadata block is initially allocated to be of the given size. Specific metadata objects (e.g., object headers, local heaps, B-trees) are then sub-allocated from this block.
The default setting is 2048 bytes, meaning that the library will attempt to aggregate metadata in at least 2K blocks in the file. Setting the value to zero (`0`) with this function will turn off metadata aggregation, even if the VFL driver attempts to use the metadata aggregation strategy.
Metadata aggregation reduces the number of small data objects in the file that would otherwise be required for metadata. The aggregated block of metadata is usually written in a single write action and always in a contiguous block, potentially significantly improving library and application performance. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5)H5Pset_metadata_read_attempts()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_metadata_read_attempts  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | unsigned |  _attempts_ )  
Sets the number of read attempts in a file access property list.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | attempts | The number of read attempts. Must be a value greater than `0` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.      Failure Modes:  
  * When the user sets the number of read attempts to `0`.
  * When the input property list is not a file access property list.
  * When the library is unable to set the number of read attempts in the file access property list.


[H5Pset_metadata_read_attempts()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5 "Sets the number of read attempts in a file access property list.") sets the number of reads that the library will try when reading checksummed metadata in an HDF5 file opened with SWMR access. When reading such metadata, the library will compare the checksum computed for the metadata just read with the checksum stored within the piece of checksum. When performing SWMR operations on a file, the checksum check might fail when the library reads data on a system that is not atomic. To remedy such situations, the library will repeatedly read the piece of metadata until the check passes or finally fails the read when the allowed number of attempts is reached.
The number of read attempts used by the library will depend on how the file is opened and whether the user sets the number of read attempts via this routine:
  * For a file opened with SWMR access:
    * If the user sets the number of attempts to `N`, the library will use `N`.
    * If the user does not set the number of attempts, the library will use the default for SWMR access (`100`).
  * For a file opened with non-SWMR access, the library will always use the default for non-SWMR access (`1`). The value set via this routine does not have any effect during non-SWMR access.


**Example:** The first example illustrates the case in setting the number of read attempts for a file opened with SWMR access.
/* Create a copy of file access property list */
fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
/* Set the # of read attempts */
[H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5)(fapl, 20);
/* Open the file with SWMR access and the non-default file access property list */
fid = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILE, ([H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) | [H5F_ACC_SWMR_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a22b12837bca0dba6689096a370d73402)), fapl);
/* Get the file's file access property list */
file_fapl = [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4)(fid);
/* Retrieve the # of read attempts from the file's file access property list */
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)(file_fapl, &attempts);
/*
* The value returned in "attempts" will be 20.
* The library will use 20 as the number of read attempts
* when reading checksummed metadata in the file
*/
/* Close the property list */
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(file_fapl);
/* Close the file */
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(fid);
**Example:** The second example illustrates the case in setting the number of read attempts for a file opened with non-SWMR access. The value set in the file access property list does not have any effect.
/* Create a copy of file access property list */
fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
/* Set the # of read attempts */
[H5Pset_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab827cef16ec569c87cec94a8b3f350c5)(fapl, 20);
/* Open the file with SWMR access and the non-default file access property list */
fid = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILE, [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), fapl);
/* Get the file's file access property list */
file_fapl = [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4)(fid);
/* Retrieve the # of read attempts from the file's file access property list */
[H5Pget_metadata_read_attempts](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga670948d56435920f1e1c2e88b823935e)(file_fapl, &attempts);
/*
* The value returned in "attempts" will be 1 (default for non-SWMR access).
* The library will use 1 as the number of read attempts
* when reading checksummed metadata in the file
*/
/* Close the property lists */
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(file_fapl);
/* Close the file */
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(fid); 

Note
     **Motivation:** On a system that is not atomic, the library might possibly read inconsistent metadata with checksum when performing single-writer/multiple-reader (SWMR) operations for an HDF5 file. Upon encountering such situations, the library will try reading the metadata again to obtain consistent data. This routine provides the means to set the number of read attempts other than the library default. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6daceb4a9e51fca7cb198f964b67baf0)H5Pset_mpi_params()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_mpi_params  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | MPI_Comm |  _comm_ ,   
|  | MPI_Info |  _info_ )  
Set the MPI communicator and info.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | comm | MPI communicator   
[in] | info | MPI info object  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_mpi_params()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6daceb4a9e51fca7cb198f964b67baf0 "Set the MPI communicator and info.") sets the MPI communicator and info stored in the file access property list `fapl_id`. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga507341f31848c57008a3225bff3fe128)H5Pset_multi_type()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_multi_type  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) |  _type_ )  
Specifies type of data to be accessed via the `MULTI` driver, enabling more direct access.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | type | Type of data to be accessed 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_multi_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga507341f31848c57008a3225bff3fe128 "Specifies type of data to be accessed via the MULTI driver, enabling more direct access.") sets the _type of data_ property in the file access property list `fapl_id`. This setting enables a user application to specify the type of data the application wishes to access so that the application can retrieve a file handle for low-level access to the particular member of a set of `MULTI` files in which that type of data is stored. The file handle is retrieved with a separate call to [H5Fget_vfd_handle()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae4020a66fb8da0586e3b74c81ffccea4 "Returns pointer to the file handle from the virtual file driver.") (or, in special circumstances, to [H5FDget_vfd_handle()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a9fcfb5d6708f4c3f5d319b801ac252bc); see [HDF5 Virtual File Layer](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html).
The type of data specified in `type` may be one of the following:
[H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce) | Super block data   
---|---  
[H5FD_MEM_BTREE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a29b8528e16990fbe265682559b917fa3) | B-tree data   
[H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e) | Dataset raw data   
[H5FD_MEM_GHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a020d6245f874e8262058c3278fefe58e) | Global heap data   
[H5FD_MEM_LHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae7536174d3ae2a842a71d6c192b43a13) | Local Heap data   
[H5FD_MEM_OHDR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a4337f7056fb57717e82fa1081f496d75) | Object header data   
This function is for use only when accessing an HDF5 file written as a set of files with the `MULTI` file driver. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19)H5Pset_object_flush_cb()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_object_flush_cb  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | [H5F_flush_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a07cc80d29d745646218aa8cb068cf944) |  _func_ ,   
|  | void * |  _udata_ )  
Sets a callback function to invoke when an object flush occurs in the file.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | func | Callback function   
[in] | udata | User-defined callback function context 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_object_flush_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19 "Sets a callback function to invoke when an object flush occurs in the file.") sets the callback function to invoke in the file access property list `plist_id` whenever an object flush occurs in the file. Library objects are group, dataset, and committed datatype.
The callback function `func` must conform to the prototype defined below: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5F_flush_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a07cc80d29d745646218aa8cb068cf944))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) object_id, void *user_data)
[H5F_flush_cb_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a07cc80d29d745646218aa8cb068cf944)
herr_t(* H5F_flush_cb_t)(hid_t object_id, void *udata)
**Definition** H5Fpublic.h:238
The parameters of the callback function, per the above prototypes, are defined as follows:
  * `object_id` is the identifier of the object which has just been flushed.
  * `user_data` is the user-defined input data for the callback function.


**Example:** The example below illustrates the usage of this routine to set the callback function to invoke when an object flush occurs.
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, fapl_id;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dataset_id, dapl_id;
unsigned counter;
/* Create a copy of the file access property list */
fapl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
/* Set up the object flush property values */
/* flush_cb: callback function to invoke when an object flushes (see below) */
/* counter: user data to pass along to the callback function */
[H5Pset_object_flush_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab4a4a788af5b6e88381dda0df2efbf19)(fapl_id, flush_cb, &counter);
/* Open the file */
file_id = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILE, [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
/* Create a group */
gid = [H5Gcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga86d93295965f750ef25dea2505a711d9)(fid, “group”, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), H5P_DEFAULT_H5P_DEFAULT);
/* Open a dataset */
dataset_id = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file_id, DATASET, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
/* The flush will invoke flush_cb() with counter */
[H5Dflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7)(dataset_id);
/* counter will be equal to 1 */
/* ... */
/* The flush will invoke flush_cb() with counter */
[H5Gflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga1d55dbf931f8003bb329c4340b8fe4d6)(gid);
/* counter will be equal to 2 */
/* ... */
/* The callback function for object flush property */
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
flush_cb([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, void *_udata)
{
unsigned *flush_ct = (unsigned *)_udata;
++(*flush_ct);
return 0;
}
[H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)
hid_t H5Dopen2(hid_t loc_id, const char *name, hid_t dapl_id)
Opens an existing dataset.
[H5Dflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga4a2175a62baa1e35ad2467bb1fdff1f7)
herr_t H5Dflush(hid_t dset_id)
Flushes all buffers associated with a dataset to disk.
[H5Gflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga1d55dbf931f8003bb329c4340b8fe4d6)
herr_t H5Gflush(hid_t group_id)
Flushes all buffers associated with a group to disk.
[H5Gcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga86d93295965f750ef25dea2505a711d9)
hid_t H5Gcreate2(hid_t loc_id, const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id)
Creates a new group and links it into the file. 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8008cddafa81bd1ddada23f6d9a161ca)H5Pset_page_buffer_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_page_buffer_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | size_t |  _buf_size_ ,   
|  | unsigned |  _min_meta_per_ ,   
|  | unsigned |  _min_raw_per_ )  
Sets the maximum size for the page buffer and the minimum percentage for metadata and raw data pages.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | buf_size | Maximum size, in bytes, of the page buffer   
[in] | min_meta_per | Minimum metadata percentage to keep in the page buffer before allowing pages containing metadata to be evicted (Default is 0)   
[in] | min_raw_per | Minimum raw data percentage to keep in the page buffer before allowing pages containing raw data to be evicted (Default is 0)  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_page_buffer_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8008cddafa81bd1ddada23f6d9a161ca "Sets the maximum size for the page buffer and the minimum percentage for metadata and raw data pages.") sets buf_size, the maximum size in bytes of the page buffer. The default value is zero, meaning that page buffering is disabled. When a non-zero page buffer size is set, the library will enable page buffering if that size is larger or equal than a single page size if a paged file space strategy is enabled using the functions [H5Pset_file_space_strategy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l...") and [H5Pset_file_space_page_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gad012d7f3c2f1e1999eb1770aae3a4963 "Sets the file space page size for a file creation property list.").
The page buffer layer captures all I/O requests before they are issued to the VFD and "caches" them in fixed sized pages. Once the total number of pages exceeds the page buffer size, the library evicts pages from the page buffer by writing them to the VFD. At file close, the page buffer is flushed writing all the pages to the file.
If a non-zero page buffer size is set, and the file space strategy is not set to paged or the page size for the file space strategy is larger than the page buffer size, the subsequent call to [H5Fcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") using the `plist_id` will fail. 

Note
    As of HDF5 1.14.4, this property will be ignored when an existing file is being opened and the file space strategy stored in the file isn't paged. This was previously a failure.      As of HDF5 1.14.4, if a file with a paged file space strategy is opened with a page size that is smaller than the file's page size, the page cache size will be rounded up to the file's page size. This was previously a failure.
The function also allows setting the minimum percentage of pages for metadata and raw data to prevent a certain type of data to evict hot data of the other type. 

Note
    As of HDF5 2.0.0, the default page buffer size (0) may be overridden in some circumstances, such as when using the ROS3 file driver. To forcibly disable the page buffer, call this function with buf_size set to 0. To return this setting to the overridable default, call this function with buf_size set to H5F_PAGE_BUFFER_SIZE_DEFAULT. This macro is only available in HDF5 2.0.0 and later. 

Since
    1.10.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafa8e677af3200e155e9208522f8e05c0)H5Pset_relax_file_integrity_checks()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_relax_file_integrity_checks  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | uint64_t |  _flags_ )  
Relax file integrity checks that may issue errors for some valid files.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | flags | Relaxed integrity checks flag. Valid values are: 
  * [H5F_RFIC_UNUSUAL_NUM_UNUSED_NUMERIC_BITS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aef18fabdccce8db11b6dd3e231028028) suppresses integrity checks for detecting unusually high values for the number of unused bits in numeric datatype classes (H5T_INTEGER, H5T_FLOAT, and H5T_BITFIELD). Integrity checks are triggered when the precision for a datatype (i.e. the number of bits containing actual data) is less than half of the datatype's size and the datatype is greater than 1 byte in size. For example, a datatype with a precision of 15 bits and a size of 4 bytes (i.e. 32 bits) will issue an error, but a datatype with 17 bits of precision and a size of 4 bytes will not issue an error, nor will a datatype with a precision of 1, 2, or 3 bits and a size of 1 byte issue an error. 
  * [H5F_RFIC_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a86e0864b6be8816c2820610bd52e049a) relaxes all integrity checks above.



Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Incorrectly encoded or corrupted metadata in a native HDF5 format file can cause incorrect library behavior when the metadata has no checksum. Integrity checks within the library detect these circumstances and issue errors when incorrect metadata is found. Unfortunately, some of the integrity checks for detecting these circumstances may incorrectly issue an error for a valid HDF5 file that was intentionally created with these configurations. Setting the appropriate flag(s) with this routine will relax the file integrity checks for these valid files and suppress errors when accessing objects with these configurations.
The library will also issue errors when these configurations are used to create objects, preventing applications from unintentionally creating them. Setting the appropriate flag with this routine will also suppress those errors on creation, although using this routine and the appropriate flag(s) will still be required when accessing files created with these configurations.
A more complete solution that avoids errors on both object creation and access is to use the H5Pset_libver_bounds routine with a low bound of at least [H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8) when creating objects with these configurations. This will cause the library to checksum a file's metadata, allowing accidental data corruption to be correctly detected and errors correctly issued without ambiguity. 

Since
    1.14.4 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24fd737955839194bf5605d5f47928ee)H5Pset_sieve_buf_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_sieve_buf_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | size_t |  _size_ )  
Sets the maximum size of the data sieve buffer.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | size | Maximum size, in bytes, of data sieve buffer 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_sieve_buf_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga24fd737955839194bf5605d5f47928ee "Sets the maximum size of the data sieve buffer.") sets `size`, the maximum size in bytes of the data sieve buffer, which is used by file drivers that are capable of using data sieving.
The data sieve buffer is used when performing I/O on datasets in the file. Using a buffer which is large enough to hold several pieces of the dataset being read in for hyperslab selections boosts performance by quite a bit.
The default value is set to 64KB, indicating that file I/O for raw data reads and writes will occur in at least 64KB blocks. Setting the value to zero (`0`) with this API function will turn off the data sieving, even if the VFL driver attempts to use that strategy.
Internally, the library checks the storage sizes of the datasets in the file. It picks the smaller one between the size from the file access property and the size of the dataset to allocate the sieve buffer for the dataset in order to save memory usage. 

Version
    1.6.0 The `size` parameter has changed from type `hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)` to `size_t`. 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5a99962a79412814b79be830f14c23dd)H5Pset_small_data_block_size()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_small_data_block_size  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _size_ )  
Sets the size of a contiguous block reserved for small data.  

Parameters
     [in] | fapl_id | File access property list identifier   
---|---|---  
[in] | size | Maximum size, in bytes, of the small data block. The default size is `2048`. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_small_data_block_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5a99962a79412814b79be830f14c23dd "Sets the size of a contiguous block reserved for small data.") reserves blocks of `size` bytes for the contiguous storage of the raw data portion of _small_ datasets. The HDF5 library then writes the raw data from small datasets to this reserved space, thus reducing unnecessary discontinuities within blocks of meta data and improving I/O performance.
A small data block is actually allocated the first time a qualifying small dataset is written to the file. Space for the raw data portion of this small dataset is suballocated within the small data block. The raw data from each subsequent small dataset is also written to the small data block until it is filled; additional small data blocks are allocated as required.
The HDF5 library employs an algorithm that determines whether I/O performance is likely to benefit from the use of this mechanism with each dataset as storage space is allocated in the file. A larger `size` will result in this mechanism being employed with larger datasets.
The small data block size is set as an allocation property in the file access property list identified by `fapl_id`.
Setting `size` to zero (`0`) disables the small data block mechanism. 

Since
    1.4.4 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0)H5Pset_vol()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_vol  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _new_vol_id_ ,   
|  | const void * |  _new_vol_info_ )  
Set the file VOL connector for a file access property list.  

Parameters
     [in] | plist_id | File access property list identifier   
---|---|---  
[in] | new_vol_id | VOL connector identifier   
[in] | new_vol_info | Optional VOL information 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_vol()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8aaa97e70b2544c3d95d908e1ae5b0f0 "Set the file VOL connector for a file access property list.") sets the VOL connector `new_vol_id` for a file access property list `plist_id` using the (optional) VOL information in `new_vol_info`. 

Since
    1.12.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


