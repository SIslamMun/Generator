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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#title2 "H2")
  * [◆ H5Pget_char_encoding()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#title3 "H2")
  * [◆ H5Pset_char_encoding()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#title4 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#func-members)
Attribute Creation Properties
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) » [String Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_t_r_c_p_l.html)
## Detailed Description
The creation property, the choice of a character encoding, applies to attributes. 
Attribute creation property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.")/[H5Pget_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d "Retrieves the character encoding used to create a link or attribute name.") | Sets/gets the character encoding used to encode link and attribute names.  

See also
     [String Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_t_r_c_p_l.html)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) *encoding)  
| Retrieves the character encoding used to create a link or attribute name.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, [H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) encoding)  
| Sets the character encoding used to encode link and attribute names.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d)H5Pget_char_encoding()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_char_encoding  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  |  [H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) * |  _encoding_ )  
Retrieves the character encoding used to create a link or attribute name.  

Parameters
     [in] | plist_id | Link creation or attribute creation property list identifier   
---|---|---  
[out] | encoding | String encoding character set 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_char_encoding()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d "Retrieves the character encoding used to create a link or attribute name.") retrieves the character encoding used to encode link or attribute names that are created with the property list `plist_id`.
Valid values for `encoding` are defined in [H5Tpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html) and include the following:
[H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663) | US ASCII  
---|---  
[H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658) | UTF-8 Unicode encoding 

Note
     [H5Pget_char_encoding()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d "Retrieves the character encoding used to create a link or attribute name.") retrieves the character set used for an HDF5 link or attribute name while [H5Tget_cset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga5bc2f3e8f708f5bcdd0d8667950310c1 "Retrieves the character set type of a string datatype.") retrieves the character set used in a character or string datatype. 

Since
    1.8.0   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc)H5Pset_char_encoding()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_char_encoding  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | [H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) |  _encoding_ )  
Sets the character encoding used to encode link and attribute names.  

Parameters
     [in] | plist_id | Link creation or attribute creation property list identifier   
---|---|---  
[in] | encoding | String encoding character set 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_char_encoding()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.") sets the character encoding used for the names of links (which provide the names by which objects are referenced) or attributes created with the property list `plist_id`.
Valid values for encoding include the following: 
[H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663) | US ASCII  
---|---  
[H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658) | UTF-8 Unicode encoding  
For example, if the character set for the property list `plist_id` is set to [H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658), link names pointing to objects created with the link creation property list `plist_id` will be encoded using the UTF-8 character set. Similarly, names of attributes created with the attribute creation property list `plist_id` will be encoded as UTF-8.
ASCII and UTF-8 Unicode are the only currently supported character encodings. Extended ASCII encodings (for example, ISO 8859) are not supported. This encoding policy is not enforced by the HDF5 library. Using encodings other than ASCII and UTF-8 can lead to compatibility and usability problems. 

Note
     [H5Pset_char_encoding()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.") sets the character set used for an HDF5 link or attribute name while [H5Tset_cset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga4909c0c3d97c3d212fee032cc8dc031a "Sets character set to be used in a string or character datatype.") sets the character set used in a character or string datatype. 

Since
    1.8.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


