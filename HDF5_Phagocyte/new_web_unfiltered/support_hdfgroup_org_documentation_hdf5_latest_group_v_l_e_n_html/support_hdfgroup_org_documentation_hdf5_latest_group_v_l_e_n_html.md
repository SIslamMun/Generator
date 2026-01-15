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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#title2 "H2")
  * [◆ H5Treclaim()](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#title3 "H2")
  * [◆ H5Tvlen_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#func-members)
Variable-length Sequence Datatypes
[Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html)
## Detailed Description
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, void *buf)  
| Reclaims the variable length (VL) datatype memory buffers.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Tvlen_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) base_id)  
| Creates a new variable-length array datatype.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe)H5Treclaim()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Treclaim  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
|  | void * |  _buf_ )  
Reclaims the variable length (VL) datatype memory buffers.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[in] | space_id | Dataspace identifier   
[in] | plist_id | Dataset transfer property list identifier used to create the buffer   
[in] | buf | Pointer to the buffer to be reclaimed 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Treclaim()](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe "Reclaims the variable length \(VL\) datatype memory buffers.") reclaims memory buffers created to store VL datatypes. It only frees the variable length data in the selection defined in the dataspace specified by `space_id`. The dataset transfer property list `plist_id` is required to find the correct allocation and/or free methods for the variable-length data in the buffer. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f)H5Tvlen_create()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Tvlen_create  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _base_id_ | ) |   
---|---|---|---|---|---  
Creates a new variable-length array datatype.  

Parameters
     [in] | base_id | Datatype identifier, the element type of the datatype to create  
---|---|--- 

Returns
    Returns a variable-length datatype identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Tvlen_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f "Creates a new variable-length array datatype.") creates a new one-dimensional array datatype of variable-length (VL) with the base datatype `base_id`.
This one-dimensional array often represents a data sequence of the base datatype, such as characters for character sequences or vertex coordinates for polygon lists. The base type specified for the VL datatype can be any HDF5 datatype, including another VL datatype, a compound datatype, or an atomic datatype.
When necessary, use [H5Tget_super()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga331e8f7b388a50af77294018db788de3 "Returns the base datatype from which a datatype is derived.") to determine the base type of the VL datatype.
The datatype identifier returned from this function should be released with [H5Tclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0 "Releases a datatype.") or resource leaks will result. Under certain circumstances, [H5Dvlen_reclaim()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga222a2fd93868e2524b2e42c3c6146119 "Reclaims variable-length \(VL\) datatype memory buffers.") must also be used. 

Attention
     [H5Tvlen_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f "Creates a new variable-length array datatype.") cannot be used to create a variable-length string datatype. [H5Tvlen_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f "Creates a new variable-length array datatype.") called with a string or character base type creates a variable-length sequence of strings (a variable-length, 1-dimensional array), with each element of the array being of the string or character base type.  
To create a variable-length string datatype, see _Creating variable-length string datatypes_. 

Since
    1.2.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


