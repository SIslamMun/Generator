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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#title2 "H2")
  * [◆ H5Tget_tag()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#title3 "H2")
  * [◆ H5Tset_tag()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#func-members)
Opaque Datatypes
[Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html)
## Detailed Description
##  Functions  
---  
char *  |  [H5Tget_tag](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#gaf1d0f634ac1a3b4220b8fe0197b93832) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  
| Gets the tag associated with an opaque datatype.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tset_tag](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#ga3543ad909983a2a20e651d16502de43d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, const char *tag)  
| Tags an opaque datatype.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#gaf1d0f634ac1a3b4220b8fe0197b93832)H5Tget_tag()
char * H5Tget_tag  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _type_ | ) |   
---|---|---|---|---|---  
Gets the tag associated with an opaque datatype.  

Parameters
     [in] | type | Datatype identifier of an opaque datatype  
---|---|--- 

Returns
    Returns a pointer to an allocated string if successful; otherwise returns NULL.  
[H5Tget_tag()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#gaf1d0f634ac1a3b4220b8fe0197b93832 "Gets the tag associated with an opaque datatype.") returns the tag associated with the opaque datatype `type`. 

Attention
    The tag is returned via a pointer to an allocated string, which the caller must free. 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#ga3543ad909983a2a20e651d16502de43d)H5Tset_tag()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tset_tag  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_ ,   
---|---|---|---  
|  | const char * |  _tag_ )  
Tags an opaque datatype.  

Parameters
     [in] | type | Datatype identifier of an opaque datatype   
---|---|---  
[in] | tag | Descriptive ASCII string with which the opaque datatype is to be tagged 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tset_tag()](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#ga3543ad909983a2a20e651d16502de43d "Tags an opaque datatype.") tags an opaque datatype `type` with a descriptive ASCII identifier, `tag`.
`tag` is intended to provide a concise description; the maximum size is hard-coded in the HDF5 library as 256 bytes ([H5T_OPAQUE_TAG_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#acfb0eb3c4d3d0c22b41997e69a69cc7f)). 

Version
    1.6.5 The [H5T_OPAQUE_TAG_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#acfb0eb3c4d3d0c22b41997e69a69cc7f) macro constant, specifying the maximum size of an opaque datatype tag, was added in [H5Tpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html). 

Since
    1.2.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


