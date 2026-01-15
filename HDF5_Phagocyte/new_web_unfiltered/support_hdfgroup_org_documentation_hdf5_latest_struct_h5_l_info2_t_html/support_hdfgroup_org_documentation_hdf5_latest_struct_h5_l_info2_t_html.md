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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title0 "H2")
  * [ Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title1 "H2")
  * [Field Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title2 "H2")
  * [◆ corder](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title3 "H2")
  * [◆ corder_valid](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title4 "H2")
  * [◆ cset](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title5 "H2")
  * [◆ token](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title6 "H2")
  * [◆ type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title7 "H2")
  * [◆ [union]](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title8 "H2")
  * [◆ val_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#pub-attribs)
H5L_info2_t Struct Reference
`#include <src/H5Lpublic.h>`
## Detailed Description
Information struct for links. 
##  Data Fields  
---  
int64_t  | [corder](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#afd86affdef2bd3f0fd3e33f62c687281)  
bool  | [corder_valid](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#a06fc831372a56bad53e322f1a105a32b)  
[H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) | [cset](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#a4f8eb75035d688b44a1e5b5b7a7051c0)  
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) | [type](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#a7c59b7f846e24f72fc4c6d21e16ed0ee)  
union {  |   
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) [token](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#aec1bc6735ca478c420f5ff729bcb3888) |   
size_t [val_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#a415813c756bd7c76f826fa542d782807) |   
}  |  [u](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#ada308fdaba4519d19a73473b5bc72413) |   
## Field Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#afd86affdef2bd3f0fd3e33f62c687281)corder
int64_t corder  
---  
Creation order 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#a06fc831372a56bad53e322f1a105a32b)corder_valid
bool corder_valid  
---  
Indicate if creation order is valid 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#a4f8eb75035d688b44a1e5b5b7a7051c0)cset
[H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) cset  
---  
Character set of link name 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#aec1bc6735ca478c420f5ff729bcb3888)token
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) token  
---  
Token of location that hard link points to 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#a7c59b7f846e24f72fc4c6d21e16ed0ee)type
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) type  
---  
Type of link 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#ada308fdaba4519d19a73473b5bc72413)[union]
union { ... } u  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#a415813c756bd7c76f826fa542d782807)val_size
size_t val_size  
---  
Size of a soft link or user-defined link value 
* * *
The documentation for this struct was generated from the following file:
  * src/[H5Lpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html)


  * [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


