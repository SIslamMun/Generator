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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title2 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title3 "H2")
  * [◆ H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title4 "H2")
  * [◆ H5FD_IOC_CURR_FAPL_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title5 "H2")
  * [◆ H5FD_IOC_DEFAULT_THREAD_POOL_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title6 "H2")
  * [◆ H5FD_IOC_FAPL_MAGIC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title7 "H2")
  * [◆ H5FD_IOC_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title8 "H2")
  * [◆ H5FD_IOC_THREAD_POOL_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#nested-classes) | [Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#func-members)
H5FDioc.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Data Structures  
---  
struct  | [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html)  
| Configuration structure for [H5Pset_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0 "Modifies the specified File Access Property List to use the H5FD_IOC driver.") / [H5Pget_fapl_ioc()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8482cdac1272818e4e0f86a665f8ef98 "Queries a File Access Property List for H5FD_IOC file driver properties.") [More...](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html#details)  
  
##  Macros  
---  
#define  |  [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_IOC_id_g)  
#define  |  [H5FD_IOC_CURR_FAPL_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#aa868930a086c6686ada02d9282e7e440) 1  
#define  |  [H5FD_IOC_DEFAULT_THREAD_POOL_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a53e8780ddbbfa83666ec9c95ef66c013) 4  
#define  |  [H5FD_IOC_FAPL_MAGIC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#af37c70faac9bf49fabb12e6b317267a9) 0xFED21331  
#define  |  [H5FD_IOC_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a93d75b8d6719a9b7075247fbe12c3d80) "ioc"  
#define  |  [H5FD_IOC_THREAD_POOL_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a4a6f4e95278aebf8a830585a2607648a) "H5FD_IOC_THREAD_POOL_SIZE"  
##  Functions  
---  
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8482cdac1272818e4e0f86a665f8ef98) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html) *config_out)  
| Queries a File Access Property List for [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) file driver properties.   
  
H5_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_ioc](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga714340ec9cbb65cb0a819e1d775825f0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html) *vfd_config)  
| Modifies the specified File Access Property List to use the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986)H5FD_IOC
#define H5FD_IOC ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_IOC_id_g)  
---  
Macro that returns the identifier for the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver. Returns a file driver identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba). 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#aa868930a086c6686ada02d9282e7e440)H5FD_IOC_CURR_FAPL_VERSION
#define H5FD_IOC_CURR_FAPL_VERSION 1  
---  
The version number of the [H5FD_ioc_config_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ioc__config__t.html "Configuration structure for H5Pset_fapl_ioc\(\) / H5Pget_fapl_ioc\(\)") configuration structure for the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a53e8780ddbbfa83666ec9c95ef66c013)H5FD_IOC_DEFAULT_THREAD_POOL_SIZE
#define H5FD_IOC_DEFAULT_THREAD_POOL_SIZE 4  
---  
The default number of I/O concentrator worker threads 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#af37c70faac9bf49fabb12e6b317267a9)H5FD_IOC_FAPL_MAGIC
#define H5FD_IOC_FAPL_MAGIC 0xFED21331  
---  
Unique number used to distinguish the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver from other HDF5 file drivers 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a93d75b8d6719a9b7075247fbe12c3d80)H5FD_IOC_NAME
#define H5FD_IOC_NAME "ioc"  
---  
The canonical name for the [H5FD_IOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a9d6649a67050fb1101cca3596421b986) driver 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html#a4a6f4e95278aebf8a830585a2607648a)H5FD_IOC_THREAD_POOL_SIZE
#define H5FD_IOC_THREAD_POOL_SIZE "H5FD_IOC_THREAD_POOL_SIZE"  
---  
Macro for name of the environment variable that controls/overrides the number of I/O concentrator worker threads
The value set for this environment variable is interpreted as an int value and must be > 0. 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDsubfiling](https://support.hdfgroup.org/documentation/hdf5/latest/dir_cecf7846bd65f3ca7010c1fa8537af78.html)
  * [H5FDioc.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dioc_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


