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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title2 "H2")
  * [◆ H5Pclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title3 "H2")
  * [◆ H5Pcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title4 "H2")
  * [◆ H5Pcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title5 "H2")
  * [◆ H5Pdecode()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title6 "H2")
  * [◆ H5Pencode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title7 "H2")
  * [◆ H5Pget_class()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#func-members)
Property List Class Root
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html)
## Detailed Description
Use the functions in this module to manage HDF5 property lists. 
Property list class root functions (H5P) Function  | Purpose   
---|---  
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb "Terminates access to a property list.") | Terminates access to a property list.   
[H5Pcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b "Copies an existing property list to create a new property list.") | Copies an existing property list to create a new property list.   
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class.") | Creates a new property list as an instance of a property list class.   
[H5Pencode](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af1a9ff52a69251d57ffa686102f162a8)/[H5Pdecode](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gafd75009eb96922e72beffa78718d4bdd "Decodes property list received in a binary object buffer and returns a new property list identifier.") | Encodes/ecodes property list into/from a binary object buffer.   
[H5Pget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5 "Returns the property list class identifier for a property list.") | Returns the property list class identifier for a property list   
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id)  
| Terminates access to a property list.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Pcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id)  
| Copies an existing property list to create a new property list.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cls_id)  
| Creates a new property list as an instance of a property list class.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Pdecode](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gafd75009eb96922e72beffa78718d4bdd) (const void *buf)  
| Decodes property list received in a binary object buffer and returns a new property list identifier.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pencode2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, void *buf, size_t *nalloc, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)  
| Encodes the property values in a property list into a binary buffer.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Pget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id)  
| Returns the property list class identifier for a property list.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)H5Pclose()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pclose  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _plist_id_ | ) |   
---|---|---|---|---|---  
Terminates access to a property list.  

Parameters
     [in] | plist_id | Property list identifier  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb "Terminates access to a property list.") terminates access to a property list. All property lists should be closed when the application is finished accessing them. This frees resources used by the property list. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b)H5Pcopy()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Pcopy  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _plist_id_ | ) |   
---|---|---|---|---|---  
Copies an existing property list to create a new property list.  

Parameters
     [in] | plist_id | Property list identifier  
---|---|--- 

Returns
    Returns a property list identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Pcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b "Copies an existing property list to create a new property list.") copies an existing property list to create a new property list. The new property list has the same properties and values as the original property list. 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)H5Pcreate()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Pcreate  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _cls_id_ | ) |   
---|---|---|---|---|---  
Creates a new property list as an instance of a property list class.  

Parameters
     [in] | cls_id | Property list class identifier  
---|---|--- 

Returns
    Returns a property list identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Pcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class.") creates a new property list as an instance of some property list class. The new property list is initialized with default values for the specified class. The classes are as follows:
Class Identifier  | Class Name  | Comments   
---|---|---  
[H5P_ATTRIBUTE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa0102211c679e031e2e9831b66c48a12) | attribute create  | Properties for attribute creation   
[H5P_DATASET_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afd849c0834c8ce6580b7c2537dbd9b5d) | dataset access  | Properties for dataset access   
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102) | dataset create  | Properties for dataset creation   
[H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d) | data transfer  | Properties for raw data transfer   
[H5P_DATATYPE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa67fdce00f24807a835955ac474febce) | datatype access  | Properties for datatype access   
[H5P_DATATYPE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6d9318d499a66b4a934fe1839b29566e) | datatype create  | Properties for datatype creation   
[H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e) | file access  | Properties for file access   
[H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977) | file create  | Properties for file creation   
[H5P_FILE_MOUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a3f57eb3c4081b40ff8b036f438e68e5b) | file mount  | Properties for file mounting   
[H5P_GROUP_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aca0fe0d98945364fe1320bf3af056b9d) | group access  | Properties for group access   
[H5P_GROUP_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a8330a95b257d45d6347a2daa96f261e9) | group create  | Properties for group creation   
[H5P_LINK_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a130a641715c9a0f3597792ce630fbe6f) | link access  | Properties governing link traversal when accessing objects   
[H5P_LINK_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad2c1d10104f0262c826350ccbf7c49f2) | link create  | Properties governing link creation   
[H5P_OBJECT_COPY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a7cedfd989522e8d7697a414d1d707e43) | object copy  | Properties governing the object copying process   
[H5P_OBJECT_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a42df2a1c964d2b985abc6e422abf6463) | object create  | Properties for object creation   
[H5P_STRING_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad5c40cc58ce5ddb42dff95eb684bd6cf) | string create  | Properties for character encoding when encoding strings or object names   
[H5P_VOL_INITIALIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afebc2bfbcba7288957a33837b6a070a5) | vol initialize  | Properties for VOL initialization   
This property list must eventually be closed with [H5Pclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb "Terminates access to a property list."); otherwise, errors are likely to occur. 

Version
    1.12.0 The [H5P_VOL_INITIALIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afebc2bfbcba7288957a33837b6a070a5) property list class was added       1.8.15 For each class, the class name returned by [H5Pget_class_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gac94a17bcb6d988a7ccb1cf2c6f4a3a82 "Retrieves the name of a class.") was added. The list of possible Fortran values was updated.       1.8.0 The following property list classes were added at this release: [H5P_DATASET_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afd849c0834c8ce6580b7c2537dbd9b5d), [H5P_GROUP_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a8330a95b257d45d6347a2daa96f261e9), [H5P_GROUP_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aca0fe0d98945364fe1320bf3af056b9d), [H5P_DATATYPE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6d9318d499a66b4a934fe1839b29566e), [H5P_DATATYPE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa67fdce00f24807a835955ac474febce), [H5P_ATTRIBUTE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa0102211c679e031e2e9831b66c48a12) 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gafd75009eb96922e72beffa78718d4bdd)H5Pdecode()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Pdecode  | ( | const void * | _buf_ | ) |   
---|---|---|---|---|---  
Decodes property list received in a binary object buffer and returns a new property list identifier.  

Parameters
     [in] | buf | Buffer holding the encoded property list  
---|---|--- 

Returns
    Returns an object identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
Given a binary property list description in a buffer, [H5Pdecode()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gafd75009eb96922e72beffa78718d4bdd "Decodes property list received in a binary object buffer and returns a new property list identifier.") reconstructs the HDF5 property list and returns an identifier for the new property list. The binary description of the property list is encoded by [H5Pencode()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af1a9ff52a69251d57ffa686102f162a8).
The user is responsible for passing in the correct buffer.
The property list identifier returned by this function should be released with [H5Pclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb "Terminates access to a property list.") when the identifier is no longer needed so that resource leaks will not develop. 

Note
    Some properties cannot be encoded and therefore will not be available in the decoded property list. These properties are discussed in [H5Pencode()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af1a9ff52a69251d57ffa686102f162a8). 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa)H5Pencode2()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pencode2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | void * |  _buf_ ,   
|  | size_t * |  _nalloc_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ )  
Encodes the property values in a property list into a binary buffer.  

Parameters
     [in] | plist_id | Property list identifier   
---|---|---  
[out] | buf | Buffer into which the property list will be encoded. If the provided buffer is NULL, the size of the buffer required is returned through `nalloc`; the function does nothing more.   
[out] | nalloc | The size of the required buffer   
[in] | fapl_id | File access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pencode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa "Encodes the property values in a property list into a binary buffer.") encodes the property list `plist_id` into the binary buffer `buf`, according to the file format setting specified by the file access property list `fapl_id`.
If the required buffer size is unknown, `buf` can be passed in as NULL and the function will set the required buffer size in `nalloc`. The buffer can then be created and the property list encoded with a subsequent [H5Pencode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa "Encodes the property values in a property list into a binary buffer.") call.
If the buffer passed in is not big enough to hold the encoded properties, the [H5Pencode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa "Encodes the property values in a property list into a binary buffer.") call can be expected to fail with a segmentation fault.
The file access property list `fapl_id` is used to control the encoding via the _libver_bounds_ property (see [H5Pset_libver_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.")). If the _libver_bounds_ property is missing, [H5Pencode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa "Encodes the property values in a property list into a binary buffer.") proceeds as if the _libver_bounds_ property were set to ([H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8), [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)). (Functionally, [H5Pencode1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gaf40518cb161ee9508da4b9c0d34553bf "Encodes the property values in a property list into a binary buffer.") is identical to [H5Pencode2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga37b1b6666e62a86389015e7dfc384faa "Encodes the property values in a property list into a binary buffer.") with _libver_bounds_ set to ([H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8), [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)).) Properties that do not have encode callbacks will be skipped. There is currently no mechanism to register an encode callback for a user-defined property, so user-defined properties cannot currently be encoded.
Some properties cannot be encoded, particularly properties that are reliant on local context.
**Motivation:** This function was introduced in HDF5-1.12 as part of the _H5Sencode_ format change to enable 64-bit selection encodings and a dataspace selection that is tied to a file. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5)H5Pget_class()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Pget_class  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _plist_id_ | ) |   
---|---|---|---|---|---  
Returns the property list class identifier for a property list.  

Parameters
     [in] | plist_id | Property list identifier  
---|---|--- 

Returns
    Returns a property list class identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Pget_class()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5 "Returns the property list class identifier for a property list.") returns the property list class identifier for the property list identified by the `plist_id` parameter.
Note that [H5Pget_class()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5 "Returns the property list class identifier for a property list.") returns a value of [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, an internal HDF5 identifier, rather than directly returning a property list class. That identifier can then be used with either [H5Pequal()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20 "Compares two property lists or classes for equality.") or [H5Pget_class_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gac94a17bcb6d988a7ccb1cf2c6f4a3a82 "Retrieves the name of a class.") to determine which predefined HDF5 property list class [H5Pget_class()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5 "Returns the property list class identifier for a property list.") has returned.
A full list of valid predefined property list classes appears in the description of [H5Pcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class.").
Determining the HDF5 property list class name with [H5Pequal()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20 "Compares two property lists or classes for equality.") requires a series of [H5Pequal()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20 "Compares two property lists or classes for equality.") calls in an if-else sequence. An iterative sequence of [H5Pequal()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20 "Compares two property lists or classes for equality.") calls can compare the identifier returned by [H5Pget_class()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5 "Returns the property list class identifier for a property list.") to members of the list of valid property list class names. A pseudo-code snippet might read as follows:
plist_class_id = [H5Pget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5) (dsetA_plist);
if [H5Pequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20) (plist_class_id, [H5P_OBJECT_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a42df2a1c964d2b985abc6e422abf6463)) = true;
[ [H5P_OBJECT_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a42df2a1c964d2b985abc6e422abf6463) is the property list class ]
[ returned by [H5Pget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5). ]
else if [H5Pequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20) (plist_class_id, [H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)) = true;
[ [H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102) is the property list class. ]
else if [H5Pequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20) (plist_class_id, [H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d)) = true;
[ [H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d) is the property list class. ]
.
. [ Continuing the iteration until a match is found. ]
.
[H5P_OBJECT_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a42df2a1c964d2b985abc6e422abf6463)
#define H5P_OBJECT_CREATE
**Definition** H5Ppublic.h:48
[H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d)
#define H5P_DATASET_XFER
**Definition** H5Ppublic.h:68
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)
#define H5P_DATASET_CREATE
**Definition** H5Ppublic.h:60
[H5Pequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20)
htri_t H5Pequal(hid_t id1, hid_t id2)
Compares two property lists or classes for equality.
[H5Pget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5)
hid_t H5Pget_class(hid_t plist_id)
Returns the property list class identifier for a property list.
[H5Pget_class_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gac94a17bcb6d988a7ccb1cf2c6f4a3a82 "Retrieves the name of a class.") returns the property list class name directly as a string:
plist_class_id = [H5Pget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5) (dsetA_plist);
plist_class_name = [H5Pget_class_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gac94a17bcb6d988a7ccb1cf2c6f4a3a82) (plist_class_id)
[H5Pget_class_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gac94a17bcb6d988a7ccb1cf2c6f4a3a82)
char * H5Pget_class_name(hid_t pclass_id)
Retrieves the name of a class.
Note that frequent use of [H5Pget_class_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#gac94a17bcb6d988a7ccb1cf2c6f4a3a82 "Retrieves the name of a class.") can become a performance problem in a high-performance environment. The [H5Pequal()](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r_a.html#ga9425ef9f3bc3ee661eca6be654aeae20 "Compares two property lists or classes for equality.") approach is generally much faster. 

Version
    1.6.0 Return type changed in this release.  

Since
    1.0.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


