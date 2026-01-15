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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#title0 "H2")
  * [ Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#title1 "H2")
  * [Field Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#title2 "H2")
  * [◆ attr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#title3 "H2")
  * [◆ hdr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#title4 "H2")
  * [◆ [struct]](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#title5 "H2")
  * [◆ obj](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#title6 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#pub-attribs)
H5O_native_info_t Struct Reference
`#include <src/H5Opublic.h>`
## Detailed Description
Native file format information struct for objects. (For [H5Oget_native_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga677d99ab106e2032b991b75b75de0e46 "Retrieve native file format information about an object."), [H5Oget_native_info_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga296ded21aeac3921fee07272353b8476 "Retrieve native file format information about an object given its name."), [H5Oget_native_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafa6570d8b0ef6e2aff75093e1f99f67e "Retrieve native file format information about an object according to the order of an index.")) 
##  Data Fields  
---  
[H5O_hdr_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html) | [hdr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#a6cfeed1003ece5686449118f36dce19f)  
struct {  |   
[H5_ih_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5__ih__info__t.html) [attr](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#aba95c19d6477004a0fa4591e467412b6) |   
[H5_ih_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5__ih__info__t.html) [obj](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#ad99b3bb65fd4d3f3ec75ab5eea43eb15) |   
}  |  [meta_size](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#a73a8b12326b16b529bf531d6023ad8b8) |   
## Field Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#aba95c19d6477004a0fa4591e467412b6)attr
[H5_ih_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5__ih__info__t.html) attr  
---  
v2 B-tree & heap for attributes 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#a6cfeed1003ece5686449118f36dce19f)hdr
[H5O_hdr_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html) hdr  
---  
Object header information 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#a73a8b12326b16b529bf531d6023ad8b8)[struct]
struct { ... } meta_size  
---  
Extra metadata storage for obj & attributes 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html#ad99b3bb65fd4d3f3ec75ab5eea43eb15)obj
[H5_ih_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5__ih__info__t.html) obj  
---  
v1/v2 B-tree & local/fractal heap for groups, B-tree for chunked datasets 
* * *
The documentation for this struct was generated from the following file:
  * src/[H5Opublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html)


  * [H5O_native_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__native__info__t.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


