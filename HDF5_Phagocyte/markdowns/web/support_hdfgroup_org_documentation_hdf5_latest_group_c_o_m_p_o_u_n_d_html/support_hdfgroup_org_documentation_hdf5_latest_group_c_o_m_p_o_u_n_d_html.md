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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#title2 "H2")
  * [◆ H5Tget_member_class()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#title3 "H2")
  * [◆ H5Tget_member_offset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#title4 "H2")
  * [◆ H5Tget_member_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#title5 "H2")
  * [◆ H5Tinsert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#title6 "H2")
  * [◆ H5Tpack()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#func-members)
Compound Datatypes
[Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html) » [Compound and Enumeration Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html)
## Detailed Description
##  Functions  
---  
[H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2) |  [H5Tget_member_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gac8476d164fb972fbf7b8c4584b8e916b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, unsigned membno)  
| Returns datatype class of compound datatype member.   
  
size_t  |  [H5Tget_member_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, unsigned membno)  
| Retrieves the offset of a field of a compound datatype.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, unsigned membno)  
| Returns the datatype of the specified member.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) parent_id, const char *name, size_t offset, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) member_id)  
| Adds a new member to a compound datatype.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tpack](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga1a28ac30f83a4920aba49bb1b0a6a0f3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id)  
| Recursively removes padding from within a compound datatype.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gac8476d164fb972fbf7b8c4584b8e916b)H5Tget_member_class()
[H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2) H5Tget_member_class  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | unsigned |  _membno_ )  
Returns datatype class of compound datatype member.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[in] | membno | Zero-based index of the field or element 

Returns
    Returns the datatype class, a non-negative value, if successful; otherwise returns a negative value.  
Given a compound datatype, `dtype_id`, [H5Tget_member_class()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gac8476d164fb972fbf7b8c4584b8e916b "Returns datatype class of compound datatype member.") returns the datatype class of the member specified by `member_no`.
Valid class identifiers, as defined in [H5Tpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html), are: 
typedef enum [H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2) {
[H5T_NO_CLASS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a89ce244a6cbb3d0205291f41f08647a8) = -1, 
[H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037) = 0, 
[H5T_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2e92f1a42a19de186a139ab8ff0745a9) = 1, 
[H5T_TIME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a8d3af61b1a73c5682f7f9b131754f6e3) = 2, 
[H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) = 3, 
[H5T_BITFIELD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aae32f37ec15a835aa08d9277ad7ffaa2) = 4, 
[H5T_OPAQUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aaf11325a64ed5369e88d8d0d600b5cce) = 5, 
[H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6) = 6, 
[H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd) = 7, 
[H5T_ENUM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5ee305303f12787367ac271d8f28f2e6) = 8, 
[H5T_VLEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2ad8ff83b6b7ca22575263561221193028) = 9, 
[H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854) = 10, 
[H5T_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2af2675ccd96bbe2c60daf885185774120) = 11, 
[H5T_NCLASSES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a900b6431883be49bde07daf9263ed117)
} [H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2); 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af)H5Tget_member_offset()
size_t H5Tget_member_offset  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | unsigned |  _membno_ )  
Retrieves the offset of a field of a compound datatype.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[in] | membno | Zero-based index of the field or element 

Returns
    Returns the byte offset of the field if successful; otherwise returns 0 (zero).  
[H5Tget_member_offset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af "Retrieves the offset of a field of a compound datatype.") retrieves the byte offset of the beginning of a field within a compound datatype with respect to the beginning of the compound datatype datum.
Note that zero is a valid offset and that this function will fail only if a call to [H5Tget_member_class()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gac8476d164fb972fbf7b8c4584b8e916b "Returns datatype class of compound datatype member.") fails with the same arguments. 

Version
    1.6.4 `member_no` parameter type changed to unsigned. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb)H5Tget_member_type()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Tget_member_type  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | unsigned |  _membno_ )  
Returns the datatype of the specified member.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[in] | membno | Zero-based index of the field or element 

Returns
    Returns the identifier of a copy of the datatype of the field if successful; otherwise returns a negative value.  
[H5Tget_member_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb "Returns the datatype of the specified member.") returns the datatype of the specified member. The caller should invoke [H5Tclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0 "Releases a datatype.") to release resources associated with the type. 

Version
    1.6.4 `membno` parameter type changed to unsigned. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)H5Tinsert()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tinsert  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _parent_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | size_t |  _offset_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _member_id_ )  
Adds a new member to a compound datatype.  

Parameters
     [in] | parent_id | Datatype identifier   
---|---|---  
[in] | name | Name of the field to insert   
[in] | offset | Offset in memory structure of the field to insert   
[in] | member_id | Datatype identifier of the field to insert 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tinsert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081 "Adds a new member to a compound datatype.") adds another member to the compound datatype, specified `type_id`.
The new member has a `name` which must be unique within the compound datatype. The `offset` argument defines the start of the member in an instance of the compound datatype, and `member_id` is the datatype identifier of the new member. 

Note
    Members of a compound datatype do not have to be atomic datatypes; a compound datatype can have a member which is a compound datatype. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga1a28ac30f83a4920aba49bb1b0a6a0f3)H5Tpack()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tpack  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _type_id_ | ) |   
---|---|---|---|---|---  
Recursively removes padding from within a compound datatype.  

Parameters
     [in] | type_id | Datatype identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tpack()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga1a28ac30f83a4920aba49bb1b0a6a0f3 "Recursively removes padding from within a compound datatype.") recursively removes padding from within a compound datatype to make it more efficient (space-wise) to store that data. 

Since
    1.0.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


