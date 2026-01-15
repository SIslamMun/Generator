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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#title0 "H2")
  * [ Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#title2 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#title3 "H2")
  * [◆ H5Tget_member_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#title4 "H2")
  * [◆ H5Tget_member_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#title5 "H2")
  * [◆ H5Tget_nmembers()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#title6 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#groups) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#func-members)
Compound and Enumeration Datatypes
[Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html)
## Detailed Description
##  Topics  
---  
| [Compound Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html)  
| [Enumeration Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html)  
##  Functions  
---  
int  |  [H5Tget_member_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#gabe31b13b2b8bf29d1a4c3b04cf917c6c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, const char *name)  
| Retrieves the index of a compound or enumeration datatype member.   
  
char *  |  [H5Tget_member_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, unsigned membno)  
| Retrieves the name of a compound or enumeration datatype member.   
  
int  |  [H5Tget_nmembers](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id)  
| Retrieves the number of elements in a compound or enumeration datatype.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#gabe31b13b2b8bf29d1a4c3b04cf917c6c)H5Tget_member_index()
int H5Tget_member_index  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | const char * |  _name_ )  
Retrieves the index of a compound or enumeration datatype member.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[in] | name | Name of the field or member 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Tget_member_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#gabe31b13b2b8bf29d1a4c3b04cf917c6c "Retrieves the index of a compound or enumeration datatype member.") retrieves the index of a field of a compound datatype or an element of an enumeration datatype.
The name of the target field or element is specified by `name`.
Fields are stored in no particular order with index values of 0 through N-1, where N is the value returned by [H5Tget_nmembers()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626 "Retrieves the number of elements in a compound or enumeration datatype.") . 

Since
    1.4.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66)H5Tget_member_name()
char * H5Tget_member_name  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
---|---|---|---  
|  | unsigned |  _membno_ )  
Retrieves the name of a compound or enumeration datatype member.  

Parameters
     [in] | type_id | Datatype identifier   
---|---|---  
[in] | membno | Zero-based index of the field or element 

Returns
    Returns a valid pointer to a string allocated with malloc() if successful; otherwise returns NULL.  
[H5Tget_member_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66 "Retrieves the name of a compound or enumeration datatype member.") retrieves the name of a field of a compound datatype or an element of an enumeration datatype.
The index of the target field or element is specified in `member_no`. Compound datatype fields and enumeration datatype elements are stored in no particular order with index values of 0 through N-1, where N is the value returned by [H5Tget_nmembers()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626 "Retrieves the number of elements in a compound or enumeration datatype.").
The HDF5 library allocates a buffer to receive the name of the field. The caller must subsequently free the buffer with [H5free_memory()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga71872bf6445cba956da86d4762b662cf "Frees memory allocated by the HDF5 library."). 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626)H5Tget_nmembers()
int H5Tget_nmembers  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _type_id_ | ) |   
---|---|---|---|---|---  
Retrieves the number of elements in a compound or enumeration datatype.  

Parameters
     [in] | type_id | Datatype identifier  
---|---|--- 

Returns
    Returns the number of elements if successful; otherwise returns a negative value.  
[H5Tget_nmembers()](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626 "Retrieves the number of elements in a compound or enumeration datatype.") retrieves the number of fields in a compound datatype or the number of members of an enumeration datatype. 

Since
    1.0.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


