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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title0 "H2")
  * [ Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title1 "H2")
  * [Field Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title2 "H2")
  * [◆ atime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title3 "H2")
  * [◆ btime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title4 "H2")
  * [◆ ctime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title5 "H2")
  * [◆ fileno](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title6 "H2")
  * [◆ mtime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title7 "H2")
  * [◆ num_attrs](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title8 "H2")
  * [◆ rc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title9 "H2")
  * [◆ token](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title10 "H2")
  * [◆ type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#pub-attribs)
H5O_info2_t Struct Reference
`#include <src/H5Opublic.h>`
## Detailed Description
Data model information struct for objects (For [H5Oget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5), [H5Oget_info_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga16d8ac07f9244cfccb42b5f309ca6b3c), [H5Oget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafe764884e1530f86079288dd5895a7bd) version 3) 
##  Data Fields  
---  
time_t  | [atime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a8688b65a745fcb4ef567a480b22a65b5)  
time_t  | [btime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#acdd8ce155a94e9a2748566d20ec1f92f)  
time_t  | [ctime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a2b105009c2d9020cbaf51d4457c20e3d)  
unsigned long  | [fileno](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#aa056ca8d2414b80b34a00c06cd69d29e)  
time_t  | [mtime](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a4a2d3a53446ef3217564ced73636f9bf)  
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) | [num_attrs](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#abc37a3659a46ce6096446cfd0d9f67ff)  
unsigned  | [rc](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a299ecaad7f9548089654d47a1b06291f)  
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) | [token](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#aec1bc6735ca478c420f5ff729bcb3888)  
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec) | [type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a6e12ef8203e400ee995a41a24e981539)  
## Field Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a8688b65a745fcb4ef567a480b22a65b5)atime
time_t atime  
---  
Access time 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#acdd8ce155a94e9a2748566d20ec1f92f)btime
time_t btime  
---  
Birth time 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a2b105009c2d9020cbaf51d4457c20e3d)ctime
time_t ctime  
---  
Change time 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#aa056ca8d2414b80b34a00c06cd69d29e)fileno
unsigned long fileno  
---  
File number that object is located in. Constant across multiple opens of the same file 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a4a2d3a53446ef3217564ced73636f9bf)mtime
time_t mtime  
---  
Modification time 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#abc37a3659a46ce6096446cfd0d9f67ff)num_attrs
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) num_attrs  
---  
Number of attributes attached to object 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a299ecaad7f9548089654d47a1b06291f)rc
unsigned rc  
---  
Reference count of object 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#aec1bc6735ca478c420f5ff729bcb3888)token
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) token  
---  
Token representing the object 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html#a6e12ef8203e400ee995a41a24e981539)type
[H5O_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html#a929ff459574495f461657f6be804cdec) type  
---  
Basic object type (group, dataset, etc.) 
* * *
The documentation for this struct was generated from the following file:
  * src/[H5Opublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html)


  * [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


