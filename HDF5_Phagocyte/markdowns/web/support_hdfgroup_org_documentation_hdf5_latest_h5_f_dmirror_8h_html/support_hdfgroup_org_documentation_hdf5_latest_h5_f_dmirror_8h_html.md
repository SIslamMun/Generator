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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title2 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title3 "H2")
  * [◆ H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title4 "H2")
  * [◆ H5FD_MIRROR_CURR_FAPL_T_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title5 "H2")
  * [◆ H5FD_MIRROR_FAPL_MAGIC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title6 "H2")
  * [◆ H5FD_MIRROR_MAX_IP_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title7 "H2")
  * [◆ H5FD_MIRROR_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#nested-classes) | [Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#func-members)
H5FDmirror.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Data Structures  
---  
struct  | [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html)  
| Configuration structure for [H5Pset_fapl_mirror()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf "Modifies the file access property list to use the H5FD_MIRROR driver.") / [H5Pget_fapl_mirror()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga62a0c63720675e1698d61b7321beb1c1 "Queries a File Access Property List for H5FD_MIRROR file driver properties.") [More...](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html#details)  
  
##  Macros  
---  
#define  |  [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_MIRROR_id_g)  
#define  |  [H5FD_MIRROR_CURR_FAPL_T_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#ab7fd09353683690e7104e6b38a612f8a) 1  
#define  |  [H5FD_MIRROR_FAPL_MAGIC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a7b2c9e2c19566ebdfd1611ddeef0d549) 0xF8DD514C  
#define  |  [H5FD_MIRROR_MAX_IP_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a177e28c121e8c29397ee7fe66e62ab56) 45 /* Max size of an IPv4-mapped IPv6 address */  
#define  |  [H5FD_MIRROR_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a421a9bb88872f205aa5f41a27cb46d16) [H5_VFD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a9d130b99837793fa7c003c8655737e58)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga62a0c63720675e1698d61b7321beb1c1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html) *fa_out)  
| Queries a File Access Property List for [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) file driver properties.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html) *fa)  
| Modifies the file access property list to use the [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) driver.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412)H5FD_MIRROR
#define H5FD_MIRROR ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_MIRROR_id_g)  
---  
ID for the mirror VFD  

Since
    1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#ab7fd09353683690e7104e6b38a612f8a)H5FD_MIRROR_CURR_FAPL_T_VERSION
#define H5FD_MIRROR_CURR_FAPL_T_VERSION 1  
---  
The version number of the [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html "Configuration structure for H5Pset_fapl_mirror\(\) / H5Pget_fapl_mirror\(\)") configuration structure for the [H5FD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a05b78c6f3d122b4112632080474b3412) driver 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a7b2c9e2c19566ebdfd1611ddeef0d549)H5FD_MIRROR_FAPL_MAGIC
#define H5FD_MIRROR_FAPL_MAGIC 0xF8DD514C  
---  
Magic number to identify the [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html "Configuration structure for H5Pset_fapl_mirror\(\) / H5Pget_fapl_mirror\(\)") struct 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a177e28c121e8c29397ee7fe66e62ab56)H5FD_MIRROR_MAX_IP_LEN
#define H5FD_MIRROR_MAX_IP_LEN 45 /* Max size of an IPv4-mapped IPv6 address */  
---  
Max size of the remote_ip array in [H5FD_mirror_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__mirror__fapl__t.html "Configuration structure for H5Pset_fapl_mirror\(\) / H5Pget_fapl_mirror\(\)")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html#a421a9bb88872f205aa5f41a27cb46d16)H5FD_MIRROR_VALUE
#define H5FD_MIRROR_VALUE [H5_VFD_MIRROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a9d130b99837793fa7c003c8655737e58)  
---  
Identifier for the mirror VFD 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDmirror.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmirror_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


