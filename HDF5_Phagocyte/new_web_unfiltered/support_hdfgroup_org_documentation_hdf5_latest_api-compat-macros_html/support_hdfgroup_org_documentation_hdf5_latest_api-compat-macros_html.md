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
  * [ Audience](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title0 "H1")
  * [ Compatibility Issues](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title1 "H1")
  * [ Summary and Motivation](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title2 "H1")
  * [ Understanding and Using the Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title3 "H1")
  * [ Compatibility Macro Mapping Options](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title4 "H1")
  * [ Library Mapping Options](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title5 "H2")
  * [ Application Mapping Options](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title6 "H2")
  * [ Function Mapping Options](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title7 "H2")
  * [ Compatibility Macros in HDF5 1.6.8 and Later](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title8 "H1")
  * [ Common Use Case](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html#title9 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
API Compatibility Macros
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Audience
The target audience for this document has existing applications that use the HDF5 library, and is considering moving to the latest HDF5 release to take advantage of the latest library features and enhancements.
#  Compatibility Issues
With each major release of HDF5, such as 1.12 or 1.10, certain compatibility issues must be considered when migrating applications from an earlier major release.
This document describes the approach taken by The HDF Group to help existing users of HDF5 address compatibility issues in the HDF5 API.
#  Summary and Motivation
In response to new and evolving requirements for the library and data format, several basic functions have changed since HDF5 was first released. To allow existing applications to continue to compile and run properly, all versions of these functions have been retained in the later releases of the HDF5 library.
Given the scope of changes available with each major release of HDF5, and recognizing the potentially time-consuming task of editing all the affected calls in user applications, The HDF Group has created a set of macros that can be used to flexibly and easily map existing API calls to previous release functions. We refer to these as the _API compatibility macros_.
The HDF Group generally encourages users to update applications to work with the latest HDF5 library release so that all new features and enhancements are available to them. At the same time, The HDF Group understands that, under some circumstances, updating applications may not be feasible or necessary. The API compatibility macros, described in this document, provide a bridge from old APIs to new and can be particularly helpful in situations such as these: 
  * Source code is not available - only binaries are available; updating the application is not feasible. 
  * Source code is available, but there are no resources to update it. 
  * Source code is available, as are resources to update it, but the old version works quite well so updates are not a priority. At the same time, it is desirable to take advantage of certain efficiencies in the newer HDF5 release that do not require code changes. 
  * Source code is available, as are resources to update it, but the applications are large or complex, and must continue to run while the code updates are carried out. 


#  Understanding and Using the Macros
As part of latest HDF5 release, several functions that existed in previous versions of the library were updated with new calling parameters and given new names. The updated versions of the functions have a number (for, e.g., '2') at the end of the original function name. The original versions of these functions were retained and renamed to have an earlier number (for, e.g., '1') at the end of the original function name.
For example, consider the function `H5Lvisit` in HDF5 release 1.10 as compared with 1.12:
Original function name and signature in 1.10.0  |  `herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lvisit(hid_t grp_id, H5_index_t idx_type, H5_iter_order_t order, H5L_iterate_t op, void *op_data)[](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69)`  
---|---  
Updated function and signature, introduced in release 1.12.0  |  `herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lvisit2(hid_t group_id, H5_index_t idx_type, H5_iter_order_t order, H5L_iterate2_t op, void *op_data)[](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gae1c6f963892a5f4e8922a66fbe338f66 "Recursively visits all links starting from a specified group.")`  
Original function and signature, renamed in release 1.12.0  |  `herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lvisit1(hid_t group_id, H5_index_t idx_type, H5_iter_order_t order, H5L_iterate1_t op, void *op_data)[](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga5424ef7043c82147490d027a0e8a59ef "Recursively visits all links starting from a specified group.")`  
API compatibility macro, introduced in release 1.12.0  |  `H5Lvisit` The macro, `H5Lvisit`, will be mapped to either `H5Lvisit1` or `H5Lvisit2`. The mapping is determined by a combination of the configuration options use to build the HDF5 library and compile-time options used to build the application. The calling parameters used with the `H5Lvisit` compatibility macro should match the number and type of the function the macros will be mapped to (`H5Lvisit1` or `H5Lvisit2`).  The function names ending in '1' or '2' are referred to as _versioned names_ , and the corresponding functions are referred to as _versioned functions_. For new code development, The HDF Group recommends use of the compatibility macro mapped to the latest version of the function. The original version of the function should be considered deprecated and, in general, should not be used when developing new code.   
#  Compatibility Macro Mapping Options
To determine the mapping for a given API compatibility macro in a given application, a combination of user-controlled selections, collectively referred to as the _compatibility macro mapping options_ , is considered in the following sequence:
  1. What compatibility macro configuration option was used to build the HDF5 library? We refer to this selection as the _library mapping_. 
  2. Was a compatibility macro global compile-time option specified when the application was built? We refer to this (optional) selection as the _application mapping_. If an application mapping exists, it overrides the library mapping. _(See adjacent notes.)_
  3. Were any compatibility macro function-level compile-time options specified when the application was built? We refer to these (optional) selections as _function mappings_. If function mappings exist, they override library and application mappings for the relevant API compatibility macros. _(See adjacent notes.)_

|  |  **Notes:** An application mapping can map APIs to the same version or to a version older than the configured library mapping. When the application attempts to map APIs to a newer version of the API than the library was configured with, it will fail to "upgrade" the mapping (and may fail silently). When it is necessary to "upgrade" the macro mappings from those set in the library mapping, it must be done at the per-function level, using the function-level mappings. As long as one does not try to map a function to a version that was compiled out in the library mapping, individual functions can be upgraded or downgraded freely.   
---  
##  Library Mapping Options
When the HDF5 library is built, `configure` flags can be used to control the API compatibility macro mapping behavior exhibited by the library. This behavior can be overridden by application and function mappings. One configure flag excludes deprecated functions from the HDF5 library, making them unavailable to applications linked with the library.
Table 1: Library Mapping [Options](https://support.hdfgroup.org/documentation/hdf5/latest/struct_options.html) `configure` flag  | Macros map to release   
(versioned function; `H5Lvisit` shown)  | Deprecated functions available?   
(`H5Lvisit1`)   
---|---|---  
`-DHDF5_DEFAULT_API_VERSION:STRING="v200"`   
(the default in 2.0.0)  | 2.0.x (`H5Lvisit2`)  | yes   
`-DHDF5_DEFAULT_API_VERSION:STRING="v114"` | 1.14.x (`H5Lvisit2`)  | yes   
`-DHDF5_DEFAULT_API_VERSION:STRING="v112"` | 1.12.x (`H5Lvisit2`)  | yes   
`-DHDF5_DEFAULT_API_VERSION:STRING="v110"` | 1.10.x (`H5Lvisit1`)  | yes   
`-DHDF5_DEFAULT_API_VERSION:STRING="v18"` | 1.8.x (`H5Lvisit1`)  | yes   
`-DHDF5_DEFAULT_API_VERSION:STRING="v16"` | 1.6.x (`H5Lvisit1`)  | yes   
`-DHDF5_ENABLE_DEPRECATED_SYMBOLS:BOOL=OFF` | 1.12.x (`H5Lvisit2`)  | no   
Refer to the file `libhdf5.settings` in the directory where the HDF5 library is installed to determine the `configure` flags used to build the library. In particular, look for the two lines shown here under _Features_ :
`Default API mapping: v200`
`With deprecated public symbols: yes`
##  Application Mapping Options
When an application using HDF5 APIs is built and linked with the HDF5 library, compile-time options to `h5cc` can be used to control the API compatibility macro mapping behavior exhibited by the application. The application mapping overrides the behavior specified by the library mapping, and can be overridden on a function-by-function basis by the function mappings.
If the HDF5 library was configured with the `-DHDF5_ENABLE_DEPRECATED_SYMBOLS:BOOL=OFF` flag, then the deprecated functions will not be available, regardless of the application mapping options.
Table 2: Application Mapping [Options](https://support.hdfgroup.org/documentation/hdf5/latest/struct_options.html) `h5cc` option  | Macros map to release   
(versioned function; `H5Lvisit` shown)  | Deprecated functions available?   
(`H5Lvisit1`)   
---|---|---  
`-DH5_USE_200_API`   
_(Default behavior if no option specified.)_ | 2.0.x (`H5Lvisit2`)  | yes*   
_*if available in library_  
`-DH5_USE_114_API` | 1.14.x (`H5Lvisit2`)  | yes*   
_*if available in library_  
`-DH5_USE_112_API` | 1.12.x (`H5Lvisit2`)  | yes*   
_*if available in library_  
`-DH5_USE_110_API` | 1.10.x (`H5Lvisit1`)  | yes*   
_*if available in library_  
`-DH5_USE_18_API` | 1.8.x (`H5Lvisit1`)  | yes*   
_*if available in library_  
`-DH5_USE_16_API` | 1.6.x (`H5Lvisit1`)  | yes*   
_*if available in library_  
`-DH5_NO_DEPRECATED_SYMBOLS` | 1.10.x (`H5Lvisit1`)  | no   
##  Function Mapping Options
Function mappings are specified when the application is built. These mappings can be used to control the mapping of the API compatibility macros to underlying functions on a function-by-function basis. The function mappings override the library and application mappings discussed earlier.
If the HDF5 library was configured with the `-DHDF5_ENABLE_DEPRECATED_SYMBOLS:BOOL=OFF` flag, or `-DH5_NO_DEPRECATED_SYMBOLS` is used to compile the application, then the deprecated functions will not be available, regardless of the function mapping options.
For every function with multiple available versions, a compile-time version flag can be defined to selectively map the function macro to the desired versioned function. The function mapping consists of the function name followed by "`_vers`" which is mapped by number to a specific function or struct: 
Macro  | Function Mapping  | Mapped to function or struct   
---|---|---  
`H5xxx` |  `H5xxx_vers=1` |  `H5xxx1`  
|  `H5xxx_vers=2` |  `H5xxx2`  
For example, in version 1.10 the `H5Rreference` macro can be mapped to either `H5Rreference1` or `H5Rreference2`. When used, the value of the `H5Rreference_vers` compile-time version flag determines which function will be called:
  * When `H5Rreference_vers` is set to `1`, the macro `H5Rreference` will be mapped to `H5Rreference1`.   
`H5cc ... -DH5Rreference_vers=1 ...`
  * When `H5Rdereference_vers` is set to `2`, the macro `H5Rdereference` will be mapped to `H5Rdereference2`.   
`h5cc ... -DH5Rreference_vers=2 ...`
  * When `H5Rreference_vers` is not set, the macro `H5Rreference` will be mapped to either `H5Rreference1` or `H5Rreference2`, based on the application mapping, if one was specified, or on the library mapping.   
`h5cc ... `



Warning
    Please be aware that some function mappings use mapped structures, as well. If compiling an application with a function mapping that uses a mapped structure, you must include each function and mapped structure plus EVERY function that uses the mapped structure, whether or not that function is used in the application. _In 1.12, mappings of structures are used by the H5L and H5O function mappings._  
  
For example, an application `application.c` only calls `H5Lvisit`, `H5Ovisit`, and `H5Oget_info_by_name`. To compile this application with 1.10 APIs in 1.12 with the function specific mappings, then not only must `H5Lvisit_vers`, `H5Ovisit_vers`, and `H5Oget_info_by_name_vers` be specified on the command line, but the mapped structures and every function that uses the mapped structures must be included, as well. The full compile line is shown below: 
h5cc -DH5Lvisit_vers=1 -DH5Ovisit_vers=1 -DH5Oget_info_by_name_vers=1 \
-DH5Lvisit_by_name_vers=1 -DH5Literate_vers=1 \
-DH5Literate_by_name_vers= -DH5O_info_t_vers=1 -DH5L_info_t_vers=1 \
-DH5L_iterate_t_vers=1 -DH5Lget_info_by_idx_vers=1 \
-DH5Lget_info_vers=1 application.c
###  Function Mapping Options in Releases 2.0.x
Macro   
(`H5xxx`)  | Default function used if no macro specified 
  * Function mapping: `H5xxx_vers=N`

| Function used for backwards compatibility with 1.12 or 1.14 
  * Function mapping: `H5xxx_vers=1`

  
---|---|---  
[H5Dread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e) |  [H5Dread_chunk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga217c9d411dfd4c0732213c7cbd98164c "Reads a raw data chunk directly from a dataset in a file into a buffer.")
  * Function mapping:`H5Dread_chunk_vers=2`

|  [H5Dread_chunk1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad3b9b3024ca4ab7eb7fe6872088398f3 "Reads a raw data chunk directly from a dataset in a file into a buffer.")
  * Function mapping:`H5Dread_chunk_vers=1`

  
[H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab) |  [H5Iregister_type2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gad698dd84ea83ade847ffd6c49bc865e2 "Creates and returns a new ID type.")
  * Function mapping:`H5Iregister_type_vers=2`

|  [H5Iregister_type1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#gab57559f14a16d4375815e45054abad16 "Creates and returns a new ID type.")
  * Function mapping:`H5Iregister_type_vers=1`

  
[H5Tdecode()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac90f7722dacad861c0a22507d3adf0dd) |  [H5Tdecode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga485114ecc0c366fda0d88833dd1083a1 "Decodes a binary object description of datatype and returns a new object handle.")
  * Function mapping:`H5Tdecode_vers=2`

|  [H5Tdecode1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gacf3174a0433c2768ca23a1047164c05e "Decodes a binary object description of datatype and returns a new object handle.")
  * Function mapping:`H5Tdecode_vers=1`

  
###  Function Mapping Options in Releases 1.14.x
No versioned functions were added in the 1.14.x releases.
###  Function Mapping Options in Releases 1.12.x
Macro   
(`H5xxx`)  | Default function used if no macro specified 
  * Function/struct mapping:`H5xxx_vers=N`

| Function used if specifying 1.10 
  * Function/struct mapping: `H5xxx_vers=1`

  
---|---|---  
[H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) |  [H5Lget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400 "Returns information about a link.")
  * Function mapping:`H5Lget_info_vers=2`
  * Struct mapping:`H5L_info_t_vers=2`

|  [H5Lget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gacc2ad7f2b402c4bf9bb122d7f43b98dc "Returns information about a link.")
  * Function mapping `H5Lget_info_vers=1`
  * Struct mapping: `H5L_info_t_vers=1`

  
[H5Lget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67) |  [H5Lget_info_by_idx2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9 "Retrieves metadata for a link in a group, according to the order within a field or index.")
  * Function mapping: `H5Lget_info_by_idx_vers=2`
  * Struct mapping: `H5L_info_t_vers=2`

|  [H5Lget_info_by_idx1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga7ed207f47e0e0f768f0d540c73e37e2a "Retrieves metadata for a link in a group, according to the order within a field or index.")
  * Function mapping: `H5Lget_info_by_idx_vers=1`
  * Struct mapping: `H5L_info_t_vers=1`

  
[H5Literate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) |  [H5Literate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73 "Iterates over links in a group, with user callback routine, according to the order within an index.")
  * Function mapping: `H5Literate_vers=2`
  * Struct mapping: `H5L_iterate_t_vers=2`

|  [H5Literate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1e7c0a8cf17699563c02e128f27042f1 "Iterates over links in a group, with user callback routine, according to the order within an index.")
  * Function mapping: `H5Literate_vers=1`
  * Struct mapping: `H5L_iterate_t_vers=1`

  
[H5Literate_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga655b002428e0176c2fa23a0315fbbcc2) |  [H5Literate_by_name2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga745a65eb516ce40a3be43490aaeb5c5e "Iterates through links in a group.")
  * Function mapping: `H5Literate_by_name_vers=2`
  * Struct mapping: `H5L_iterate_t_vers=2`

|  [H5Literate_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga87e036da0c8d1146a073f3ee08e0fedc "Iterates through links in a group by its name.")
  * Function mapping: `H5Literate_by_name_vers=1`
  * Struct mapping: `H5L_iterate_t_vers=1`

  
[H5Lvisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) |  [H5Lvisit2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gae1c6f963892a5f4e8922a66fbe338f66 "Recursively visits all links starting from a specified group.")
  * Function mapping: `H5Lvisit_vers=2`
  * Struct mapping: `H5L_iterate_t_vers=2`

|  [H5Lvisit1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga5424ef7043c82147490d027a0e8a59ef "Recursively visits all links starting from a specified group.")
  * Function mapping: `H5Lvisit_vers=1`
  * Struct mapping: `H5L_iterate_t_vers=1`

  
[H5Lvisit_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga138405315e233673741893e4e250f055) |  [H5Lvisit_by_name2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gafee93792c7e27a7e78b1ec221876b173 "Recursively visits all links starting from a specified group.")
  * Function mapping: `H5Lvisit_by_name_vers=2`
  * Struct mapping: `H5L_iterate_t_vers=2`

|  [H5Lvisit_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1f1ba1bb4d44f2c111990024809417ac "Recursively visits all links starting from a specified group.")
  * Function mapping: `H5Lvisit_by_name_vers=1`
  * Struct mapping: `H5L_iterate_t_vers=1`

  
[H5Oget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) |  [H5Oget_info3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0fbf7d780a1eefce920facadb198013 "Retrieves the metadata for an object specified by an identifier.")
  * Function mapping: `H5Oget_info_vers=3`
  * Struct mapping: `H5O_info_t_vers=2`

|  [H5Oget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf3751684a6706e3ba49b863406011f80 "Retrieves the metadata for an object specified by an identifier.")
  * Function mapping: `H5Oget_info_vers=1`
  * Struct mapping: `H5O_info_t_vers=1`

  
[H5Oget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafe764884e1530f86079288dd5895a7bd) |  [H5Oget_info_by_idx3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa2f8884f7d3e7fd9b8549f5b59fd9eb "Retrieves the metadata for an object, identifying the object by an index position.")
  * Function mapping: `H5Oget_info_by_idx_vers=3`
  * Struct mapping: `H5O_info_t_vers=2`

|  [H5Oget_info_by_idx1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga7208d2cf198dcfc875603323841bffae "Retrieves the metadata for an object, identifying the object by an index position.")
  * Function mapping: `H5Oget_info_by_idx_vers=1`
  * Struct mapping: `H5O_info_t_vers=1`

  
[H5Oget_info_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga16d8ac07f9244cfccb42b5f309ca6b3c) |  [H5Oget_info_by_name3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gabb69c962999e027cef0079bbb1282199 "Retrieves the metadata for an object, identifying the object by location and relative name.")
  * Function mapping: `H5O_get_info_by_name_vers=3`
  * Struct mapping: `H5O_info_t_vers=2`

|  [H5Oget_info_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga96ce408ffda805210844246904da2842 "Retrieves the metadata for an object, identifying the object by location and relative name.")
  * Function mapping: `H5O_get_info_by_name_vers=1`
  * Struct mapping: `H5O_info_t_vers=1`

  
[H5Ovisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) |  [H5Ovisit3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6d03115ae0e5e5b516bbf35bb492266a "Recursively visits all objects accessible from a specified object.")
  * Function mapping: `H5Ovisit_vers=3`
  * Struct mapping: `H5O_iterate_t_vers=2`

|  [H5Ovisit1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6efdb2a0a9fe9fe46695cc0f7bd993e7 "Recursively visits all objects accessible from a specified object.")
  * Function mapping: `H5Ovisit_vers=1`
  * Struct mapping: `H5O_iterate_t_vers=1`

  
[H5Ovisit_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gab02a69e88b11404e7fd61f55344b186c) |  [H5Ovisit_by_name3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga34815400b01df59c4dac19436124885a "Recursively visits all objects accessible from a specified object.")
  * Function mapping: `H5Ovisit_by_name_vers=3`
  * Struct mapping: `H5O_iterate_t_vers=2`

|  [H5Ovisit_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaffacf3bd66f4fe074099eae1c80914f2 "Recursively visits all objects starting from a specified object.")
  * Function mapping: `H5Ovisit_by_name_vers=1`
  * Struct mapping: `H5O_iterate_t_vers=1`

  
[H5Pencode()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af1a9ff52a69251d57ffa686102f162a8) |  [H5Pencode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa "Encodes the property values in a property list into a binary buffer.")
  * Function mapping: `H5Pencode_vers=2`

|  [H5Pencode1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gaf40518cb161ee9508da4b9c0d34553bf "Encodes the property values in a property list into a binary buffer.")
  * Function mapping: `H5Pencode_vers=1`

  
[H5Sencode()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga35e289baf490229e233296929fbf5208) |  [H5Sencode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga178ec7b8769ad5704a170d9bd3421074 "Encodes a data space object description into a binary buffer.")
  * Function mapping: `H5Sencode_vers=2`

|  [H5Sencode1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga82cf9f6af03ad89be21c36922e03baea "Encodes a data space object description into a binary buffer.")
  * Function mapping: `H5Sencode_vers=1`

  
###  Function Mapping Options in Releases 1.10.x
Macro  | Default function used   
(if no macro specified) | Introduced in  |  `h5cc` version flag and value  | Mapped to function or struct   
---|---|---|---|---  
[H5Rdereference()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a) |  [H5Rdereference2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga9b09586f7b6ec708434dd8f95f58a9b7 "Opens the HDF5 object referenced.") | HDF5-1.10.0  |  `-DH5Rdereference_vers=1` |  [H5Rdereference1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gafe7bbbc168c4836949c4c0b092654c45 "Opens the HDF5 object referenced.")  
`-DH5Rdereference_vers=2` |  [H5Rdereference2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga9b09586f7b6ec708434dd8f95f58a9b7 "Opens the HDF5 object referenced.")  
[H5Fget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae17036b3e36a8777328204e8bf073144) |  [H5Fget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaced8c09c1559636a9c3f33dff3f4520e "Retrieves global file information.") | HDF5-1.10.0  |  `-DH5Fget_info_vers=1` |  [H5Fget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga660153029322fa6b77f5473cedc2d72f "Retrieves global file information.") with struct [H5F_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__info1__t.html)  
`-DH5Fget_info_vers=2` |  [H5Fget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaced8c09c1559636a9c3f33dff3f4520e "Retrieves global file information.") with struct [H5F_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f__info2__t.html)  
[H5Oget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) |  [H5Oget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf3751684a6706e3ba49b863406011f80 "Retrieves the metadata for an object specified by an identifier.") | HDF5-1.10.3  |  `-DH5Oget_info_vers=1` |  [H5Oget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf3751684a6706e3ba49b863406011f80 "Retrieves the metadata for an object specified by an identifier.")  
`-DH5Oget_info_vers=2` |  [H5Oget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga06f896e14fe4fa940fbc2bc235e0cf74 "Retrieves the metadata for an object specified by an identifier.")  
[H5Oget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafe764884e1530f86079288dd5895a7bd) |  [H5Oget_info_by_idx1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga7208d2cf198dcfc875603323841bffae "Retrieves the metadata for an object, identifying the object by an index position.") | HDF5-1.10.3  |  `-DH5Oget_info_by_idx_vers=1` |  [H5Oget_info_by_idx1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga7208d2cf198dcfc875603323841bffae "Retrieves the metadata for an object, identifying the object by an index position.")  
`-DH5Oget_info_by_idx_vers=2` |  [H5Oget_info_by_idx2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga85e15e65922874111da1a5efd5dd7bed "Retrieves the metadata for an object, identifying the object by an index position.")  
[H5Oget_info_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga16d8ac07f9244cfccb42b5f309ca6b3c) |  [H5Oget_info_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga96ce408ffda805210844246904da2842 "Retrieves the metadata for an object, identifying the object by location and relative name.") | HDF5-1.10.3  |  `-DH5Oget_info_by_name_vers=1` |  [H5Oget_info_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga96ce408ffda805210844246904da2842 "Retrieves the metadata for an object, identifying the object by location and relative name.")  
`-DH5Oget_info_by_name_vers=2` |  [H5Oget_info_by_name2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga0090da86c086c1c63a5acfaed39a035e "Retrieves the metadata for an object, identifying the object by location and relative name.")  
[H5Ovisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) |  [H5Ovisit1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6efdb2a0a9fe9fe46695cc0f7bd993e7 "Recursively visits all objects accessible from a specified object.") | HDF5-1.10.3  |  `-DH5Ovisit_vers=1` |  [H5Ovisit1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga6efdb2a0a9fe9fe46695cc0f7bd993e7 "Recursively visits all objects accessible from a specified object.")  
`-DH5Ovisit_vers=2` |  [H5Ovisit2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa4ab542f581f4fc9a4eaa95debb29c9e "Recursively visits all objects accessible from a specified object.")  
[H5Ovisit_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gab02a69e88b11404e7fd61f55344b186c) |  [H5Ovisit_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaffacf3bd66f4fe074099eae1c80914f2 "Recursively visits all objects starting from a specified object.") | HDF5-1.10.3  |  `-DH5Ovisit_by_name_vers=1` |  [H5Ovisit_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaffacf3bd66f4fe074099eae1c80914f2 "Recursively visits all objects starting from a specified object.")  
`-DH5Ovisit_by_name_vers=2` |  [H5Ovisit_by_name2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga9c155caf5499405fe403e1eb27b5beb6 "Recursively visits all objects starting from a specified object.")  
###  Function Mapping Options in Releases 1.8.x
At release 1.8.0, the API compatibility macros, function mapping compile-time version flags and values, and corresponding versioned functions listed in the following table were introduced. If the application being compiled to run with any 1.10.x release was written to use any 1.6.x release of HDF5, you must also consider these macros and mapping options.
Table 5: Function Mapping [Options](https://support.hdfgroup.org/documentation/hdf5/latest/struct_options.html) in Releases 1.8.x  Macro  |  `h5cc` version flag and value  | Mapped to function   
or struct   
---|---|---  
[H5Acreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24) |  `DH5Acreate_vers=1` |  [H5Acreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaa30f5f6c277d6c46f8aa31e89cdba085 "Creates an attribute attached to a specified object.")  
`DH5Acreate_vers=2` |  [H5Acreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308 "Creates an attribute attached to a specified object.")  
[H5Aiterate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) |  `DH5Aiterate_vers=1` |  [H5Aiterate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabdb2cf7368eec0ad998cbe6a3f61aa41 "Calls a user's function for each attribute on an object.")   
with struct [H5A_operator1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#ae42c937252ed79a1ad4672f04adba750)  
`DH5Aiterate_vers=2` |  [H5Aiterate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9315a22b60468b6e996559b1b8a77251 "Calls a user-defined function for each attribute on an object.")   
with struct [H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74)  
[H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) |  `DH5Dcreate_vers=1` |  [H5Dcreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga6b86f2683ae6a78d48d33c45257744a2 "Creates a dataset at the specified location.")  
`DH5Dcreate_vers=2` |  [H5Dcreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df "Creates a new dataset and links it into the file.")  
[H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) |  `DH5Dopen_vers=1` |  [H5Dopen1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabaf03a683e1da2c8dad6ba1010d55b81 "Opens an existing dataset.")  
`DH5Dopen_vers=2` |  [H5Dopen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2 "Opens an existing dataset.")  
[H5Eclear()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac4c79bee16d0ffe8a6017bfdb66c9916) |  `DH5Eclear_vers=1` |  [H5Eclear1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga0f2ee26cbe35c5dde49d615fc31ea2f6 "Clears the error stack for the current thread.")  
`DH5Eclear_vers=2` |  [H5Eclear2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gac9d90679c7879f3c4ebce858aaa9dfb2 "Clears the specified error stack or the error stack for the current thread.")  
[H5Eprint()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaa7e93cb96b399e5853721258872435a8) |  `DH5Eprint_vers=1` |  [H5Eprint1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga9c71eb8e5b7261668e2e8926f1822365 "Prints the current error stack in a default manner.")  
`DH5Eprint_vers=2` |  [H5Eprint2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gae86ab32e85028412c731cf0f2b8d1f56 "Prints the specified error stack in a default manner.")  
[H5Epush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4a7d9ca6b4f7bf521d29c85bbc5b7941) |  `DH5Epush_vers=1` |  [H5Epush1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga7e2d223ad3bf68fe35f343b97edf0e92 "Pushes a new error record onto the error stack.")  
`DH5Epush_vers=2` |  [H5Epush2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga17c2790837a1c1ea7e56b65d3c00a920 "Pushes a new error record onto an error stack.")  
[H5Eset_auto()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga245383d63fba5c4282ce8e9ef8488702) |  `DH5Eset_auto_vers=1` |  [H5Eset_auto1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gab7e1c2db4a0811b70227833b5462eea8 "Turns automatic error printing on or off.")  
`DH5Eset_auto_vers=2` |  [H5Eset_auto2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaf0d6b18cd5160517fe5165b9a8443c69 "Turns automatic error printing on or off.")  
[H5Eget_auto()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaa088620ef0b6c4ac2abcf57d61c8cdb8) |  `DH5Eget_auto_vers=1` |  [H5Eget_auto1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga0ca4dd7ed560882a7da176a0e2325707 "Returns the current settings for the automatic error stack traversal function and its data.")  
`DH5Eget_auto_vers=2` |  [H5Eget_auto2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga2eda33cbadd9be5bfddbaa91e863c936 "Returns the settings for the automatic error stack traversal function and its data.")  
[H5E_auto_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a553d5a9d3baca989e9cc00d369810051)   
struct for [H5Eset_auto()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga245383d63fba5c4282ce8e9ef8488702)   
and [H5Eget_auto()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#gaa088620ef0b6c4ac2abcf57d61c8cdb8) |  `DH5E_auto_t_vers=1` |  [H5E_auto1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a54e242097ba2121788448b37579a107e)  
`DH5E_auto_t_vers=2` |  [H5E_auto2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a0109c84ac574ac24abe6f7c7acab3fca)  
[H5Ewalk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga565cd6c7b7a197f8954d821419aba0d0) |  `DH5Ewalk_vers=1` |  [H5Ewalk1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga8d54a0f62f9ea625bdeab8e5e0c894c4 "Walks the current error stack, calling the specified function.")   
with callback [H5E_walk1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#a2088c033647027f76172116e2da33293 "Callback function for H5Ewalk1\(\)")   
and struct [H5E_error1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error1__t.html)  
`DH5Ewalk_vers=2` |  [H5Ewalk2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4ecc0f6a1ea5bb821373a5a7b8070655 "Walks the specified error stack, calling the specified function.")   
with callback [H5E_walk2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_epublic_8h.html#aa0fc6445c613e4159a17d28ca61be825 "Callback function for H5Ewalk2\(\)")   
and struct [H5E_error2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_e__error2__t.html)  
[H5Gcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) |  `DH5Gcreate_vers=1` |  [H5Gcreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga7397440085510728a2e2d22199e81980 "Creates a new group and links it into the file.")  
`DH5Gcreate_vers=2` |  [H5Gcreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga86d93295965f750ef25dea2505a711d9 "Creates a new group and links it into the file.")  
[H5Gopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245) |  `DH5Gopen_vers=1` |  [H5Gopen1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga163ca3eb7893d34973ee900b2da886be "Opens an existing group for modification and returns a group identifier for that group.")  
`DH5Gopen_vers=2` |  [H5Gopen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gadab91e2dd7a7e253dcc0e4fe04b81403 "Opens an existing group in a file.")  
[H5Pget_filter()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7e070dfec9cb3a3aaf9c188a987e6a15) |  `DH5Pget_filter_vers=1` |  [H5Pget_filter1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gacbd4f93aa7cd270668385440fb5873a1 "Returns information about a filter in a pipeline \(DEPRECATED\)")  
`DH5Pget_filter_vers=2` |  [H5Pget_filter2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga024d200a6a07e12f008a62c4e62d0bcc "Returns information about a filter in a pipeline.")  
[H5Pget_filter_by_id()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac7aa336e7b1b9033cea2448ba623951f) |  `DH5Pget_filter_by_id_vers=1` |  [H5Pget_filter_by_id1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga351bb4dc44dae41344f18aab177f4cf1 "Returns information about the specified filter.")  
`DH5Pget_filter_by_id_vers=2` |  [H5Pget_filter_by_id2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga2d5e9df5f0e93abae11ee5edd82fcec3 "Returns information about the specified filter.")  
[H5Pinsert()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9ccdea50538c7cfde87a9fa63ea68555) |  `DH5Pinsert_vers=1` |  [H5Pinsert1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga6ba9694c03ae97c9f514470366a909f9 "Registers a temporary property with a property list.")  
`DH5Pinsert_vers=2` |  [H5Pinsert2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga930e15d5f994e223bea80621ef3065d4 "Registers a temporary property with a property list.")  
[H5Pregister()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a334ff323dfa6653ce21d0988ae7c73ba) |  `DH5Pregister_vers=1` |  [H5Pregister1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga91799f6cda78911e9ecc2cfaaea3a3b5 "Registers a permanent property with a property list class.")  
`DH5Pregister_vers=2` |  [H5Pregister2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gaac3f957a5d3cbb4adc8b7ba2aa5f1719 "Registers a permanent property with a property list class.")  
[H5Rget_obj_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gafe413df448be0d230de922357fd7bc3b) |  `DH5Rget_obj_typevers=1` |  [H5Rget_obj_type1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaf3736b2880a58471882b079b9f03defe "Retrieves the type of object that an object reference points to.")  
`DH5Rget_obj_type_vers=2` |  [H5Rget_obj_type2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga766e39a76bcdd68dc514425353eff807 "Retrieves the type of object that an object reference points to.")  
[H5Tarray_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) |  `DH5Tarray_create_vers=1` |  [H5Tarray_create1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gaa0dc45417b2d45cc6810a1831f117e80 "Creates an array datatype object.")  
`DH5Tarray_create_vers=2` |  [H5Tarray_create2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618 "Creates an array datatype object.")  
[H5Tcommit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd) |  `DH5Tcommit_vers=1` |  [H5Tcommit1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1c00afb6dc5534778370a92c33fa2625 "Commits a transient datatype to a file, creating a newly named datatype.")  
`DH5Tcommit_vers=2` |  [H5Tcommit2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga10352b6fa9ac58a7fbd5299496f1df31 "Commits a transient datatype, linking it into the file and creating a new committed datatype.")  
[H5Tget_array_dims()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473) |  `DH5Tget_array_dims_vers=1` |  [H5Tget_array_dims1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga40dca4c9bdc5e6781a07830570a476ca "Retrieves sizes of array dimensions.")  
`DH5Tget_array_dims_vers=2` |  [H5Tget_array_dims2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga3ea18a56f03d3b9c8f3ff4091c784769 "Retrieves sizes of array dimensions.")  
[H5Topen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209) |  `DH5Topen_vers=1` |  [H5Topen1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9f76fa0dc34bc7b310e100e5bfed66fb "Opens a named datatype.")  
`DH5Topen_vers=2` |  [H5Topen2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga7e65e77634f1fb4ba38cbcdab9a59bc2 "Opens a committed \(named\) datatype.")  
[H5Z_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a3f702f77f3ab031d771be3b2b4bf4fba) struct for [H5Zregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b "Registers a new filter with the HDF5 library.") |  `DH5Z_class_t_vers=1` |  [H5Z_class1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class1__t.html)  
`DH5Z_class_t_vers=2` |  [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html)  
###  Further Information
It is possible to specify multiple function mappings for a single application build: 
h5cc ... -DH5Rdereference_vers=1 -DH5Fget_info_vers=2 ...
As a result of the function and struct mappings in this compile example, all occurrences of the macro `H5Rdereference` will be mapped to `H5Rdereference1` and all occurrences of the macro `H5Fget_info` will be mapped to `H5Fget_info2` for the application being built.
The function and struct mappings can be used to guarantee that a given API compatibility macro will be mapped to the desired underlying function or struct version regardless of the library or application mappings. In cases where an application may benefit greatly from features offered by some of the later APIs, or must continue to use some earlier API versions for compatibility reasons, this fine-grained control may be very important.
As noted earlier, the function mappings can only reference versioned functions that are included in the HDF5 library, as determined by the configure flag used to build the library. For example, if the HDF5 library being linked with the application was built with the `-DHDF5_ENABLE_DEPRECATED_SYMBOLS:BOOL=OFF` option, version 1 of the underlying functions would not be available, and the example above that defined `H5Rdereference_vers=1` would not be supported.
The function mappings do not negate any available functions. If `H5Rdereference1` is available in the installed version of the HDF5 library, and the application was not compiled with the `-DH5_NO_DEPRECATED_SYMBOLS` flag, the function `H5Rdereference1` will remain available to the application through its versioned name. Similarly, `H5Rdereference2` will remain available to the application as `H5Rdereference2`. The function mapping version flag `H5Rdereference_vers` only controls the mapping of the API compatibility macro `H5Rdereference` to one of the two available functions.
This can be especially useful in any case where the programmer does not have direct control over global macro definitions, such as when writing code meant to be copied to multiple applications or when writing code in a header file.
#  Compatibility Macros in HDF5 1.6.8 and Later
A series of similar compatibility macros were introduced into the release 1.6 series of the library, starting with release 1.6.8. These macros simply alias the '1' version functions, callbacks, and typedefs listed above to their original non-numbered names.
These macros were strictly a forward-looking feature at that time; they were not necessary for compatibility in 1.6.x. These macros were created at that time to enable writing code that could be used with any version of the library after 1.6.8 and any library compilation options except `H5_NO_DEPRECATED_SYMBOLS`, by always using the '1' version of versioned functions and types. For example, `H5Dopen1` will always be interpreted in exactly the same manner by any version of the library since 1.6.8.
#  Common Use Case
A common scenario where the API compatibility macros may be helpful is the migration of an existing application to a new HDF5 release. An incremental migration plan is outlined here: 
  1. Build the HDF5 library without specifying any library mapping `configure` flag. In this default mode, the 1.6.x, 1.8.x, and 1.10.x versions of the underlying functions are available, and the API compatibility macros will be mapped to the current HDF5 versioned functions. 
  2. Compile the application with the `-DH5_USE_NN_API` application mapping option if it was written for use with an earlier HDF5 library. Because the application mapping overrides the library mapping, the macros will all be mapped to the earlier versions of the functions. 
  3. Remap one API compatibility macro at a time (or sets of macros), to use the current HDF5 versions. At each stage, use the function mappings to map the macros being worked on to the current versions. For example, use the `-DH5Rdereference_vers=2` version flag setting to remap the `H5Rdereference` macro to `H5Rdereference2`, the 1.10.x version.
During this step, the application code will need to be modified to change the calling parameters used with the API compatibility macros to match the number and type of the 1.10.x versioned functions. The macro name, for example `H5Rdereference`, should continue to be used in the code, to allow for possible re-mappings to later versioned functions in a future release. 
  4. After all macros have been migrated to the latest versioned functions in step 3, compile the application without any application or function mappings. This build uses the library mappings set in step 1, and maps API compatibility macros to the latest versions. 
  5. Finally, compile the application with the application mapping `-DH5_NO_DEPRECATED_SYMBOLS`, and address any failures to complete the application migration process. 


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


