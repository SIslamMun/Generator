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
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#title1 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#title2 "H2")
  * [◆ H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#title3 "H2")
  * [◆ H5FD_CORE_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#func-members)
H5FDcore.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Macros  
---  
#define  |  [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_CORE_id_g)  
#define  |  [H5FD_CORE_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ad764689b321f93e219b1074ac5ae9dc1) [H5_VFD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a0b5d9e83c9aeaa531daecda4a6b91a4d)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaddc561e523035fffea356cd6eef26e50) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t *increment, bool *backing_store)  
| Queries core file driver properties.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t increment, bool backing_store)  
| Modifies the file access property list to use the [H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) driver.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a)H5FD_CORE
#define H5FD_CORE ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_CORE_id_g)  
---  
ID for the core VFD 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ad764689b321f93e219b1074ac5ae9dc1)H5FD_CORE_VALUE
#define H5FD_CORE_VALUE [H5_VFD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a0b5d9e83c9aeaa531daecda4a6b91a4d)  
---  
Identifier for the core VFD  

Since
    1.14.0 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDcore.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


