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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#title2 "H2")
  * [◆ H5Tenum_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#title3 "H2")
  * [◆ H5Tenum_insert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#title4 "H2")
  * [◆ H5Tenum_nameof()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#title5 "H2")
  * [◆ H5Tenum_valueof()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#title6 "H2")
  * [◆ H5Tget_member_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#func-members)
Enumeration Datatypes
[Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html) » [Compound and Enumeration Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html)
## Detailed Description
##  Functions  
---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Tenum_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gafd8d8cfead535791b3f753d21e79991f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) base_id)  
| Creates a new enumeration datatype.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tenum_insert](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, const char *name, const void *value)  
| Inserts a new enumeration datatype member.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tenum_nameof](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gab45f148151595716006ad8b603d55623) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, const void *value, char *name, size_t size)  
| Returns the symbol name corresponding to a specified member of an enumeration datatype.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tenum_valueof](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga5a50f4172640de713e16f0ecd12aeb30) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, const char *name, void *value)  
| Returns the value corresponding to a specified member of an enumeration datatype.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Tget_member_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga4fc2d29dcde5af45b905bbc7355d2b76) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, unsigned membno, void *value)  
| Returns the value of an enumeration datatype member.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gafd8d8cfead535791b3f753d21e79991f)H5Tenum_create()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Tenum_create  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _base_id_ | ) |   
---|---|---|---|---|---  
Creates a new enumeration datatype.  

Parameters
     [in] | base_id | Datatype identifier for the base datatype. Must be an integer datatype  
---|---|--- 

Returns
    Returns a enumeration datatype identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Tenum_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gafd8d8cfead535791b3f753d21e79991f "Creates a new enumeration datatype.") creates a new enumeration datatype based on the specified base datatype, dtype_id, which must be an integer datatype.
If a particular architecture datatype is required, a little endian or big endian datatype for example, use a native datatype as the base datatype and use [H5Tconvert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a "Converts data from one specified datatype to another.") on values as they are read from or written to a dataset. 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f)H5Tenum_insert()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tenum_insert  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | const void * |  _value_ )  
Inserts a new enumeration datatype member.  

Parameters
     [in] | type | Datatype identifier   
---|---|---  
[in] | name | Name of the new member   
[in] | value | Pointer to the value of the new member 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tenum_insert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f "Inserts a new enumeration datatype member.") inserts a new enumeration datatype member into an enumeration datatype.
`type_id` is the datatype identifier for the enumeration datatype, `name` is the name of the new member, and `value` points to the value of the new member.
`name` and `value` must both be unique within `dtype_id`.
`value` points to data which must be of the integer base datatype used when the enumeration datatype was created. If a particular architecture datatype is required, a little endian or big endian datatype for example, use a native datatype as the base datatype and use [H5Tconvert()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a "Converts data from one specified datatype to another.") on values as they are read from or written to a dataset. 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gab45f148151595716006ad8b603d55623)H5Tenum_nameof()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tenum_nameof  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_ ,   
---|---|---|---  
|  | const void * |  _value_ ,   
|  | char * |  _name_ ,   
|  | size_t |  _size_ )  
Returns the symbol name corresponding to a specified member of an enumeration datatype.  

Parameters
     [in] | type | Datatype identifier   
---|---|---  
[in] | value | Value of the enumeration datatype   
[out] | name | Buffer for output of the symbol name   
[in] | size | Anticipated size of the symbol name, in bytes 

Returns
    Returns a non-negative value if successful. Otherwise returns a negative value  
[H5Tenum_nameof()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gab45f148151595716006ad8b603d55623 "Returns the symbol name corresponding to a specified member of an enumeration datatype.") finds the symbol name that corresponds to the specified `value` of the enumeration datatype `type`.
At most `size` characters of the symbol `name` are copied into the `name` buffer. If the entire symbol name and null terminator do not fit in the name buffer, then as many characters as possible are copied (not null terminated) and the function fails. 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga5a50f4172640de713e16f0ecd12aeb30)H5Tenum_valueof()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tenum_valueof  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | void * |  _value_ )  
Returns the value corresponding to a specified member of an enumeration datatype.  

Parameters
     [in] | type | Datatype identifier   
---|---|---  
[in] | name | Symbol name of the enumeration datatype   
[out] | value | Buffer for the value of the enumeration datatype 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tenum_valueof()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga5a50f4172640de713e16f0ecd12aeb30 "Returns the value corresponding to a specified member of an enumeration datatype.") finds the value that corresponds to the specified name of the enumeration datatype `dtype_id`.
Values returned in `value` will be of the enumerated type's base type, that is, the datatype used by [H5Tenum_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gafd8d8cfead535791b3f753d21e79991f "Creates a new enumeration datatype.") when the enumerated type was created.
The `value` buffer must be at least large enough to hold a value of that base type. If the size is unknown, you can determine it with [H5Tget_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e "Returns the size of a datatype."). 

Since
    1.2.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga4fc2d29dcde5af45b905bbc7355d2b76)H5Tget_member_value()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Tget_member_value  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | unsigned |  _membno_ ,   
|  | void * |  _value_ )  
Returns the value of an enumeration datatype member.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[in] | membno | Number of the enumeration datatype member   
[out] | value | Buffer for the value of the enumeration datatype member 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tget_member_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga4fc2d29dcde5af45b905bbc7355d2b76 "Returns the value of an enumeration datatype member.") returns the value of the enumeration datatype member `member_no`.
The member value is returned in a user-supplied buffer pointed to by `value`. Values returned in `value` will be of the enumerated type's base type, that is, the datatype used by [H5Tenum_create()](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gafd8d8cfead535791b3f753d21e79991f "Creates a new enumeration datatype.") when the enumerated type was created.
The value buffer must be at least large enough to hold a value of that base type. If the size is unknown, you can determine it with [H5Tget_size()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e "Returns the size of a datatype."). 

Since
    1.0.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


