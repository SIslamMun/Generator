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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#title2 "H2")
  * [◆ H5VLnative_addr_to_token()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#title3 "H2")
  * [◆ H5VLnative_token_to_addr()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#func-members)
Native VOL
[VOL connector (H5VL)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l.html)
## Detailed Description
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5VLnative_addr_to_token](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga09ca3912386a8c8c66edbcbbe2c10c1f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) *token)  
| Convert a [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) address to a native VOL connector token.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5VLnative_token_to_addr](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga7136f48f79f4b88d87002d5c218ceb40) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) token, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) *addr)  
| Convert a native VOL connector token to a [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) address.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga09ca3912386a8c8c66edbcbbe2c10c1f)H5VLnative_addr_to_token()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5VLnative_addr_to_token  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) |  _addr_ ,   
|  |  [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) * |  _token_ )  
Convert a [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) address to a native VOL connector token.  

Parameters
     [in] | loc_id | Location identifier of object. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | addr | Object address   
[out] | token | Object token 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
This API call maps pre-VOL [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) native file format addresses to the more generic [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) tokens used by the VOL. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_l_n_a_t.html#ga7136f48f79f4b88d87002d5c218ceb40)H5VLnative_token_to_addr()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5VLnative_token_to_addr  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) |  _token_ ,   
|  |  [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) * |  _addr_ )  
Convert a native VOL connector token to a [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) address.  

Parameters
     [in] | loc_id | Location identifier of object. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | token | Object token   
[out] | addr | Object address 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
This API call maps generic [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) tokens used by the VOL to pre-VOL [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) native file format addresses. 

Since
    1.12.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


