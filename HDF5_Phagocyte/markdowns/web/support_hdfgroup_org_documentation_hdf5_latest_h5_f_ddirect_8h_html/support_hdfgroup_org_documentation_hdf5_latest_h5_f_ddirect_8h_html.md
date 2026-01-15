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
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#title1 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#title2 "H2")
  * [◆ CBSIZE_DEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#title3 "H2")
  * [◆ FBSIZE_DEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#title4 "H2")
  * [◆ H5FD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#title5 "H2")
  * [◆ H5FD_DIRECT_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#title6 "H2")
  * [◆ MBOUNDARY_DEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#func-members)
H5FDdirect.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Macros  
---  
#define  |  [CBSIZE_DEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a06f85ce88b1cb4032fc2343fbfce8190) (16 * 1024 * 1024)  
#define  |  [FBSIZE_DEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a9133a19b80fa2105189cf2241c6aa98f) 4096  
#define  |  [H5FD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a99213f218f9ab0c51f9c679228a1e436) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_DIRECT_id_g)  
#define  |  [H5FD_DIRECT_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#ab8bd8b277af48acc156d0f122e977a63) [H5_VFD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#af360414fdad9d3109a40887f6dd64046)  
#define  |  [MBOUNDARY_DEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#ab417064ff567fd14a8b9bb24164134dd) 4096  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4d94952796e789d7922f039d9659d34d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t *boundary, size_t *block_size, size_t *cbuf_size)  
| Retrieves direct I/O driver settings.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, size_t alignment, size_t block_size, size_t cbuf_size)  
| Sets up use of the direct I/O driver.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a06f85ce88b1cb4032fc2343fbfce8190)CBSIZE_DEF
#define CBSIZE_DEF (16 * 1024 * 1024)  
---  
Default value for maximum copy buffer size 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a9133a19b80fa2105189cf2241c6aa98f)FBSIZE_DEF
#define FBSIZE_DEF 4096  
---  
Default value for file block size 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#a99213f218f9ab0c51f9c679228a1e436)H5FD_DIRECT
#define H5FD_DIRECT ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_DIRECT_id_g)  
---  
ID for the direct VFD 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#ab8bd8b277af48acc156d0f122e977a63)H5FD_DIRECT_VALUE
#define H5FD_DIRECT_VALUE [H5_VFD_DIRECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#af360414fdad9d3109a40887f6dd64046)  
---  
Identifier for the direct VFD  

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html#ab417064ff567fd14a8b9bb24164134dd)MBOUNDARY_DEF
#define MBOUNDARY_DEF 4096  
---  
Default value for memory boundary 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDdirect.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddirect_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


