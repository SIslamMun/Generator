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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title2 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title3 "H2")
  * [◆ H5FD__CURR_HDFS_FAPL_T_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title4 "H2")
  * [◆ H5FD__HDFS_KERB_CACHE_PATH_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title5 "H2")
  * [◆ H5FD__HDFS_NODE_NAME_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title6 "H2")
  * [◆ H5FD__HDFS_USER_NAME_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title7 "H2")
  * [◆ H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title8 "H2")
  * [◆ H5FD_HDFS_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#nested-classes) | [Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#func-members)
H5FDhdfs.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Data Structures  
---  
struct  | [H5FD_hdfs_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html)  
| Configuration structure for [H5Pset_fapl_hdfs()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931 "Modifies the file access property list to use the H5FD_HDFS driver.") / [H5Pget_fapl_hdfs()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafb4ec2b60ab86eade5aa55f5e4f5fd35 "Queries a File Access Property List for H5FD_HDFS file driver properties.") [More...](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html#details)  
  
##  Macros  
---  
#define  |  [H5FD__CURR_HDFS_FAPL_T_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#af0f0f4be810d8a828ad1bce776bb96c6) 1  
#define  |  [H5FD__HDFS_KERB_CACHE_PATH_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#a6250cdaee6a7de597350f77ac7f519af) 128  
#define  |  [H5FD__HDFS_NODE_NAME_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ab12143da32df036a6134516b22994e69) 128  
#define  |  [H5FD__HDFS_USER_NAME_SPACE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#a1b8c5c4e12e06f03e41d42017e3b481a) 128  
#define  |  [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_HDFS_id_g)  
#define  |  [H5FD_HDFS_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#a1904d7995406b511843b9115e8d6d3ef) [H5_VFD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a934cc8573a3cc95b8d84150e60744105)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gafb4ec2b60ab86eade5aa55f5e4f5fd35) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_hdfs_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html) *fa_out)  
| Queries a File Access Property List for [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) file driver properties.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_hdfs_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html) *fa)  
| Modifies the file access property list to use the [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) driver.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#af0f0f4be810d8a828ad1bce776bb96c6)H5FD__CURR_HDFS_FAPL_T_VERSION
#define H5FD__CURR_HDFS_FAPL_T_VERSION 1  
---  
The version number of the [H5FD_hdfs_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__hdfs__fapl__t.html "Configuration structure for H5Pset_fapl_hdfs\(\) / H5Pget_fapl_hdfs\(\)") configuration structure for the [H5FD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584) driver  

Since
    1.8.22 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#a6250cdaee6a7de597350f77ac7f519af)H5FD__HDFS_KERB_CACHE_PATH_SPACE
#define H5FD__HDFS_KERB_CACHE_PATH_SPACE 128  
---  
Max size of the kerberos cache path  

Since
    1.10.6, back-ported to 1.8.22 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ab12143da32df036a6134516b22994e69)H5FD__HDFS_NODE_NAME_SPACE
#define H5FD__HDFS_NODE_NAME_SPACE 128  
---  
Max size of the node name  

Since
    1.10.6, back-ported to 1.8.22 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#a1b8c5c4e12e06f03e41d42017e3b481a)H5FD__HDFS_USER_NAME_SPACE
#define H5FD__HDFS_USER_NAME_SPACE 128  
---  
Max size of the user name  

Since
    1.10.6, back-ported to 1.8.22 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#ac3868cc2fa0e9aec4bcb52830906d584)H5FD_HDFS
#define H5FD_HDFS ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_HDFS_id_g)  
---  
ID for the HDFS VFD 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html#a1904d7995406b511843b9115e8d6d3ef)H5FD_HDFS_VALUE
#define H5FD_HDFS_VALUE [H5_VFD_HDFS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a934cc8573a3cc95b8d84150e60744105)  
---  
Identifier for the hdfs VFD  

Since
    1.14.0 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDhdfs.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dhdfs_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


