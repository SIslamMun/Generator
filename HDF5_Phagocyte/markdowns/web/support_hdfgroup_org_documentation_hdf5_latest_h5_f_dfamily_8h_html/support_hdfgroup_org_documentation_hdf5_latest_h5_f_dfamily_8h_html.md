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
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#title1 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#title2 "H2")
  * [◆ H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#title3 "H2")
  * [◆ H5FD_FAMILY_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#func-members)
H5FDfamily.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Macros  
---  
#define  |  [H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_FAMILY_id_g)  
#define  |  [H5FD_FAMILY_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#a775be690fa1777bfce3b253aa23a7dc3) [H5_VFD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a84db4e6fd3b7b135648de7ee97c92d6c)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *memb_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *memb_fapl_id)  
| Returns file access property list information.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) memb_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) memb_fapl_id)  
| Sets the file access property list to use the family driver.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6)H5FD_FAMILY
#define H5FD_FAMILY ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_FAMILY_id_g)  
---  
ID for the family VFD 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#a775be690fa1777bfce3b253aa23a7dc3)H5FD_FAMILY_VALUE
#define H5FD_FAMILY_VALUE [H5_VFD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a84db4e6fd3b7b135648de7ee97c92d6c)  
---  
Identifier for the family VFD  

Since
    1.14.0 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDfamily.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


