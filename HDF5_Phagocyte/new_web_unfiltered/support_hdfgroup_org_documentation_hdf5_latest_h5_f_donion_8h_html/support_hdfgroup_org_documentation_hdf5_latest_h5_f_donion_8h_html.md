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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title1 "H2")
  * [ Enumerations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title2 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title3 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title4 "H2")
  * [◆ H5FD_ONION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title5 "H2")
  * [◆ H5FD_ONION_FAPL_INFO_COMMENT_MAX_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title6 "H2")
  * [◆ H5FD_ONION_FAPL_INFO_CREATE_FLAG_ENABLE_PAGE_ALIGNMENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title7 "H2")
  * [◆ H5FD_ONION_FAPL_INFO_REVISION_ID_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title8 "H2")
  * [◆ H5FD_ONION_FAPL_INFO_VERSION_CURR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title9 "H2")
  * [◆ H5FD_ONION_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title10 "H2")
  * [Enumeration Type Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title11 "H2")
  * [◆ H5FD_onion_target_file_constant_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#title12 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#nested-classes) | [Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#define-members) | [Enumerations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#enum-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#func-members)
H5FDonion.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Data Structures  
---  
struct  | [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html)  
##  Macros  
---  
#define  |  [H5FD_ONION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a1d6673897b4ebd1bad9846b5695ba346) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_ONION_id_g)  
#define  |  [H5FD_ONION_FAPL_INFO_COMMENT_MAX_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a79df74491cceb74d43ca9e10d0be25ce) 255  
#define  |  [H5FD_ONION_FAPL_INFO_CREATE_FLAG_ENABLE_PAGE_ALIGNMENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a3d376561fd9da23647f9731f749fd68f) (0x0001u)  
#define  |  [H5FD_ONION_FAPL_INFO_REVISION_ID_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a3e252443bb7c90b2d05c0bb5f97891f0) UINT64_MAX  
#define  |  [H5FD_ONION_FAPL_INFO_VERSION_CURR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a4562e4fe67ed2c5448e4145c44734420) 1  
#define  |  [H5FD_ONION_VALUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a45dad20deb25126398919c7d6e16ce9e) [H5_VFD_ONION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#ad0fbc20e7979087c577f8e50a4ebe121)  
##  Enumerations  
---  
enum  |  [H5FD_onion_target_file_constant_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#abbf57ea23523a7fceff6f1212bd68a6f) { [H5FD_ONION_STORE_TARGET_ONION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#abbf57ea23523a7fceff6f1212bd68a6fa680be8b6ba69ed683e247e0bd5e2ed4a) }  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5FDonion_get_revision_count](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html#ga54309eee8a5a936e3e56a2e20eba23f0) (const char *filename, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, uint64_t *revision_count)  
| get the number of revisions   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaedf8b83d7571bfffb7336a8e11a98107) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) *fa_out)  
| get the onion info from the file access property list   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, const [H5FD_onion_fapl_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__onion__fapl__info__t.html) *fa)  
| set the onion info for the file access property list   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a1d6673897b4ebd1bad9846b5695ba346)H5FD_ONION
#define H5FD_ONION ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_ONION_id_g)  
---  
ID for the onion VFD 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a79df74491cceb74d43ca9e10d0be25ce)H5FD_ONION_FAPL_INFO_COMMENT_MAX_LEN
#define H5FD_ONION_FAPL_INFO_COMMENT_MAX_LEN 255  
---  
Max length of a comment. The buffer is defined to be this size + 1 to handle the NUL.  

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a3d376561fd9da23647f9731f749fd68f)H5FD_ONION_FAPL_INFO_CREATE_FLAG_ENABLE_PAGE_ALIGNMENT
#define H5FD_ONION_FAPL_INFO_CREATE_FLAG_ENABLE_PAGE_ALIGNMENT (0x0001u)  
---  
Onion history metadata will align to page_size. Partial pages of unused space will occur in the file, but may improve read performance from the backing store on some systems. If disabled (0), padding will not be inserted to align to page boundaries.  

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a3e252443bb7c90b2d05c0bb5f97891f0)H5FD_ONION_FAPL_INFO_REVISION_ID_LATEST
#define H5FD_ONION_FAPL_INFO_REVISION_ID_LATEST UINT64_MAX  
---  
Indicates that you want the latest revision. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a4562e4fe67ed2c5448e4145c44734420)H5FD_ONION_FAPL_INFO_VERSION_CURR
#define H5FD_ONION_FAPL_INFO_VERSION_CURR 1  
---  
Current version of the onion VFD fapl info struct 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#a45dad20deb25126398919c7d6e16ce9e)H5FD_ONION_VALUE
#define H5FD_ONION_VALUE [H5_VFD_ONION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#ad0fbc20e7979087c577f8e50a4ebe121)  
---  
Identifier for the onion VFD  

Since
    1.14.0 
## Enumeration Type Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#abbf57ea23523a7fceff6f1212bd68a6f)H5FD_onion_target_file_constant_t
enum [H5FD_onion_target_file_constant_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html#abbf57ea23523a7fceff6f1212bd68a6f)  
---  
Indicates how the new onion data will be stored. 
Enumerator  
---  
H5FD_ONION_STORE_TARGET_ONION  |  Onion history is stored in a single, separate "onion file". Shares filename and path as [hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html) file (if any), with only a different filename extension.   
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDonion.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_donion_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


