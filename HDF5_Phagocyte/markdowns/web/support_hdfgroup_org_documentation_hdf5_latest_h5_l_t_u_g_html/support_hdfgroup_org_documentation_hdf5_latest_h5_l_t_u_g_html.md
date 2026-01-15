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
  * [ HDF5 Lite APIs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_t__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_t__u_g.html#title1 "H2")
  * [ Dataset Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_t__u_g.html#title2 "H2")
  * [ Attribute Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_t__u_g.html#title3 "H2")
  * [ Utility Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_t__u_g.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 High Level Lite
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 Lite APIs
##  Introduction
The HDF5 Lite API (H5LT) provides simplified functions for common HDF5 operations, combining multiple low-level calls into single high-level functions. This reduces code complexity and makes HDF5 more accessible for straightforward data storage tasks. 

See also
     [HDF5 Lite APIs (H5LT,H5LD)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html) Reference Manual
##  Dataset Operations
Create and access datasets with single function calls using type-specific functions like [H5LTmake_dataset_int](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga28891f2fbec407534394a8148c3f0908 "Creates and writes a dataset."), [H5LTmake_dataset_float](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga08fa486a415b87a78d32fbff6a74cbfb "Creates and writes a dataset."), [H5LTread_dataset_int](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga27dff339f933e51aaf0cf9169bed3256 "Reads a dataset from disk."), etc. Query dataset properties with [H5LTget_dataset_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gaf93283cb49cccae513181caf15e8ed28 "Gets the dimensionality of a dataset."), [H5LTget_dataset_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga608f1a76153de414722b75cf7831cca8 "Retrieves information about a dataset."), and [H5LTfind_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga4523d5d0a683ed022a184209d5ce2927 "Determines whether a dataset exists.").
##  Attribute Operations
Set and read attributes using type-specific functions such as [H5LTset_attribute_int](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga9b2c7d5eef5a1d3c90c6857ca337f737 "Creates and writes an attribute."), [H5LTset_attribute_string](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae60e8867e35c255dc5b5f3de134df80c "Creates and writes a string attribute."), [H5LTget_attribute_int](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga26785bc328ff30f2217547c17fe2f6de "Reads an attribute from disk."), etc. Query attribute information with [H5LTget_attribute_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad20a9cf20f1de48602a441c6a7dcc8d7 "Gets the dimensionality of an attribute.") and [H5LTget_attribute_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga8111f14c59f7b72c73803442b1b42ff7 "Gets information about an attribute.").
##  Utility Functions
Convert between datatypes and text representations using [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f "Creates an HDF5 datatype given a text description.") and [H5LTdtype_to_text](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga155a5e59441ed05ca1b91e6404314862 "Creates a text description of an HDF5 datatype."). Validate paths with [H5LTpath_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga39038c92a8dcbe7bb542661572112089 "Determines whether an HDF5 path is valid and, optionally, whether the path resolves to an HDF5 object..."), and work with file images using [H5LTopen_file_image](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae2834cca05d6c4787f51933cc521f8fb "Opens an HDF5 file image in memory.").
Previous Chapter [HDF5 Images](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i_m__u_g.html#sec_hl_images) - Next Chapter [HDF5 Table APIs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_b__u_g.html#sec_hl_table_api)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


