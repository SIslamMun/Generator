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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title2 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title3 "H2")
  * [◆ H5FD_CURR_SPLITTER_VFD_CONFIG_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title4 "H2")
  * [◆ H5FD_SPLITTER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title5 "H2")
  * [◆ H5FD_SPLITTER_MAGIC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title6 "H2")
  * [◆ H5FD_SPLITTER_PATH_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title7 "H2")
  * [◆ H5FD_SPLITTER_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#nested-classes) | [Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#func-members)
H5FDsplitter.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Data Structures  
---  
struct  | [H5FD_splitter_vfd_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__splitter__vfd__config__t.html)  
##  Macros  
---  
#define  |  [H5FD_CURR_SPLITTER_VFD_CONFIG_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#a6ca98b3d47a26ca06252457a2027d5fc) 1  
#define  |  [H5FD_SPLITTER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#ac6c45c6a8e1cb7f5b4400d95bf651eae) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_SPLITTER_id_g)  
#define  |  [H5FD_SPLITTER_MAGIC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#a50dbab4cdb1dd40688e6ebf1a0e6ffa3) 0x2B916880  
#define  |  [H5FD_SPLITTER_PATH_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#aa662df8b1c69b6aeec89a1eb9834a651) 4096  
#define  |  [H5FD_SPLITTER_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#a965d833a5b79795b8fc3f2a5ebc18139) [H5_VFD_SPLITTER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a58e1d1b57dc2409a1ef6adc9b9039c43)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga6e850a0319dc6839e2dd06a0522b3831) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_splitter_vfd_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__splitter__vfd__config__t.html) *config_ptr)  
| Gets splitter driver properties from the the file access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_splitter_vfd_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__splitter__vfd__config__t.html) *config_ptr)  
| Sets the file access property list to use the splitter driver.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#a6ca98b3d47a26ca06252457a2027d5fc)H5FD_CURR_SPLITTER_VFD_CONFIG_VERSION
#define H5FD_CURR_SPLITTER_VFD_CONFIG_VERSION 1  
---  
The version of the [H5FD_splitter_vfd_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__splitter__vfd__config__t.html) structure used 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#ac6c45c6a8e1cb7f5b4400d95bf651eae)H5FD_SPLITTER
#define H5FD_SPLITTER ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_SPLITTER_id_g)  
---  
ID for the splitter VFD 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#a50dbab4cdb1dd40688e6ebf1a0e6ffa3)H5FD_SPLITTER_MAGIC
#define H5FD_SPLITTER_MAGIC 0x2B916880  
---  
Semi-unique constant used to help identify structure pointers  

Since
    1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#aa662df8b1c69b6aeec89a1eb9834a651)H5FD_SPLITTER_PATH_MAX
#define H5FD_SPLITTER_PATH_MAX 4096  
---  
Maximum length of a filename/path string in the Write-Only channel, including the NULL-terminator.  

Since
    1.10.7 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html#a965d833a5b79795b8fc3f2a5ebc18139)H5FD_SPLITTER_VALUE
#define H5FD_SPLITTER_VALUE [H5_VFD_SPLITTER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a58e1d1b57dc2409a1ef6adc9b9039c43)  
---  
Identifier for the splitter VFD  

Since
    1.14.0 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDsplitter.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsplitter_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


