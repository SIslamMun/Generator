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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title0 "H2")
  * [ Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title1 "H2")
  * [Field Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title2 "H2")
  * [◆ flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title3 "H2")
  * [◆ free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title4 "H2")
  * [◆ mesg [1/2]](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title5 "H2")
  * [◆ [struct] [2/2]](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title6 "H2")
  * [◆ meta](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title7 "H2")
  * [◆ nchunks](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title8 "H2")
  * [◆ nmesgs](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title9 "H2")
  * [◆ present](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title10 "H2")
  * [◆ shared](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title11 "H2")
  * [◆ [struct]](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title12 "H2")
  * [◆ total](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title13 "H2")
  * [◆ version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#title14 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#pub-attribs)
H5O_hdr_info_t Struct Reference
`#include <src/H5Opublic.h>`
## Detailed Description
Information struct for object header metadata (for [H5Oget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5), [H5Oget_info_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga16d8ac07f9244cfccb42b5f309ca6b3c), [H5Oget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafe764884e1530f86079288dd5895a7bd)) 
##  Data Fields  
---  
unsigned  | [flags](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a8b08a4d2ec878257d64c55f64a62242c)  
struct {  |   
uint64_t [present](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a85692f5f8e411b5cb7164a7ab671ddd4) |   
uint64_t [shared](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a4f9c2211d5095250a8f01b262e831243) |   
}  |  [mesg](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a4d4f838e047e4cc7200fd14760a7f405) |   
unsigned  | [nchunks](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a0ede30b8596ce981ce6e8ad09d2d3382)  
unsigned  | [nmesgs](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a005ffdb55b518afa0634576e5c3d0833)  
struct {  |   
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) [free](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#acdbfc9793e4909dbda74715d9d651bec) |   
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) [mesg](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#ad5c04140b241dcf36ccb8c4e2ff66662) |   
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) [meta](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a5ee8b6f9a7fbc8bf3ea0243589341968) |   
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) [total](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a257cfedcb31932215bea0c04f3a52fef) |   
}  |  [space](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a0d12d11a37a661fe7a38dfeba5c7df36) |   
unsigned  | [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a27fe8e82cf2ce359010cc1d08af2e105)  
## Field Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a8b08a4d2ec878257d64c55f64a62242c)flags
unsigned flags  
---  
Object header status flags 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#acdbfc9793e4909dbda74715d9d651bec)free
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) free  
---  
Free space within object header 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#ad5c04140b241dcf36ccb8c4e2ff66662)mesg [1/2]
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) mesg  
---  
Space within header for actual message information 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a4d4f838e047e4cc7200fd14760a7f405)[struct] [2/2]
struct { ... } mesg  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a5ee8b6f9a7fbc8bf3ea0243589341968)meta
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) meta  
---  
Space within header for object header metadata information 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a0ede30b8596ce981ce6e8ad09d2d3382)nchunks
unsigned nchunks  
---  
Number of object header chunks 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a005ffdb55b518afa0634576e5c3d0833)nmesgs
unsigned nmesgs  
---  
Number of object header messages 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a85692f5f8e411b5cb7164a7ab671ddd4)present
uint64_t present  
---  
Flags to indicate presence of message type in header 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a4f9c2211d5095250a8f01b262e831243)shared
uint64_t shared  
---  
Flags to indicate message type is shared in header 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a0d12d11a37a661fe7a38dfeba5c7df36)[struct]
struct { ... } space  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a257cfedcb31932215bea0c04f3a52fef)total
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) total  
---  
Total space for storing object header in file 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html#a27fe8e82cf2ce359010cc1d08af2e105)version
unsigned version  
---  
Version number of header format in file 
* * *
The documentation for this struct was generated from the following file:
  * src/[H5Opublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html)


  * [H5O_hdr_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__hdr__info__t.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


