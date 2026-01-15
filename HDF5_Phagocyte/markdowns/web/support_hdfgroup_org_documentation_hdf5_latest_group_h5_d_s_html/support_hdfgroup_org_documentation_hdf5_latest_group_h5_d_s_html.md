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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title2 "H2")
  * [◆ H5DSattach_scale()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title3 "H2")
  * [◆ H5DSdetach_scale()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title4 "H2")
  * [◆ H5DSget_label()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title5 "H2")
  * [◆ H5DSget_num_scales()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title6 "H2")
  * [◆ H5DSget_scale_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title7 "H2")
  * [◆ H5DSis_attached()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title8 "H2")
  * [◆ H5DSis_scale()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title9 "H2")
  * [◆ H5DSiterate_scales()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title10 "H2")
  * [◆ H5DSset_label()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title11 "H2")
  * [◆ H5DSset_scale()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title12 "H2")
  * [◆ H5DSwith_new_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#title13 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#func-members)
HDF5 Dimension Scales APIs (H5DS)
## Detailed Description
_Creating and manipulating HDF5 datasets that are associated with the dimension of another HDF5 dataset (H5DS)_ 

Note
     **Programming hints:**      To use any of these functions or subroutines, you must first include the relevant include file (C) or module (Fortran) in your application.       The following line includes the HDF5 Dimension Scale package, H5DS, in C applications: 
#include "hdf5_hl.h"      This line includes the H5DS module in Fortran applications: 
use [h5ds](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5ds.html)
[h5ds](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5ds.html)
This module contains Fortran interfaces for H5DS.
**Definition** H5DSff.F90:35
  * [H5DSwith_new_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gafd2a9719c14946b1478e3e79736b3094)   
Determines if new references are used with dimension scales.
  * [H5DSattach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b)   
Attach dimension scale dsid to dimension idx of dataset did.
  * [H5DSdetach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad4af4f5d5c880c5c93ca950cdf5a01fd)   
Detach dimension scale dsid from the dimension idx of Dataset did.
  * [H5DSget_label](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga265aa3dadb5d067c112620ffe01f96d7)   
Read the label for dimension idx of did into buffer label.
  * [H5DSget_num_scales](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga562b91296f739a1833a1e1cfc64c7726)   
Determines how many Dimension Scales are attached to dimension idx of did.
  * [H5DSget_scale_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga704bbb7eb2c9696f0f2c84e0267e4ffa)   
Retrieves name of scale did into buffer name.
  * [H5DSis_attached](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad7bbd73a68b031f6ac4c305333b0c8b7)   
Report if dimension scale dsid is currently attached to dimension idx of dataset did.
  * [H5DSis_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaf6c043d90b502ecf96783e8f50cadcd5)   
Determines whether dset is a Dimension Scale.
  * [H5DSiterate_scales](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga7cb78620d05e7dcb81a9f53a8cc496f5)   
Iterates the operation visitor through the scales attached to dimension dim.
  * [H5DSset_label](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga10e953a37cf817f491fe47acb1370d88)   
Set label for the dimension idx of did to the value label.
  * [H5DSset_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaca0f37c36c4e1288d161d26daa3a41d3)   
Convert dataset dsid to a dimension scale, with optional name, dimname. 


##  Functions  
---  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DSattach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dsid, unsigned int idx)  
| Attach dimension scale `dsid` to dimension `idx` of dataset did.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DSdetach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad4af4f5d5c880c5c93ca950cdf5a01fd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dsid, unsigned int idx)  
| Detach dimension scale `dsid` from the dimension `idx` of dataset `did`.   
  
H5HL_DLL ssize_t  |  [H5DSget_label](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga265aa3dadb5d067c112620ffe01f96d7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did, unsigned int idx, char *label, size_t size)  
| Read the label for dimension `idx` of `did` into buffer `label`.   
  
H5HL_DLL int  |  [H5DSget_num_scales](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga562b91296f739a1833a1e1cfc64c7726) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did, unsigned int idx)  
| Determines how many Dimension Scales are attached to dimension `idx` of `did`.   
  
H5HL_DLL ssize_t  |  [H5DSget_scale_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga704bbb7eb2c9696f0f2c84e0267e4ffa) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did, char *name, size_t size)  
| Retrieves name of scale `did` into buffer `name`.   
  
H5HL_DLL [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5DSis_attached](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad7bbd73a68b031f6ac4c305333b0c8b7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dsid, unsigned int idx)  
| Report if dimension scale `dsid` is currently attached to dimension `idx` of dataset `did`.   
  
H5HL_DLL [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5DSis_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaf6c043d90b502ecf96783e8f50cadcd5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did)  
| Determines whether `did` is a Dimension Scale.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DSiterate_scales](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga7cb78620d05e7dcb81a9f53a8cc496f5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did, unsigned int dim, int *idx, [H5DS_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a7979d0cd5a423ae1cbccd0b65600bdf6) visitor, void *visitor_data)  
| Iterates the operation visitor through the scales attached to dimension `dim`.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DSset_label](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga10e953a37cf817f491fe47acb1370d88) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did, unsigned int idx, const char *label)  
| Set label for the dimension `idx` of `did` to the value `label`.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DSset_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaca0f37c36c4e1288d161d26daa3a41d3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dsid, const char *dimname)  
| Convert dataset `dsid` to a dimension scale, with optional name, `dimname`.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DSwith_new_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gafd2a9719c14946b1478e3e79736b3094) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, bool *with_new_ref)  
| Determines if new references are used with dimension scales.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b)H5DSattach_scale()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DSattach_scale  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _did_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dsid_ ,   
|  | unsigned int |  _idx_ )  
Attach dimension scale `dsid` to dimension `idx` of dataset did. 
* * * 

Parameters
     [in] | did | The dataset   
---|---|---  
[in] | dsid | The scale to be attached   
[in] | idx | The dimension of `did` that `dsid` is associated with 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Define Dimension Scale `dsid` to be associated with dimension `idx` of dataset `did`.
Entries are created in the [DIMENSION_LIST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a3c3e0049491003d0fb4f085bb8f504db) and [REFERENCE_LIST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a16370c2d086f86916320c34e416cf445) attributes, as defined in [Storage Profile](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#subsec_dim_scales_spec_store) section of the [HDF5 High Level Dimension Scales](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html).
Fails if:
  * Bad arguments
  * If `dsid` is not a Dimension Scale
  * If `did` is a Dimension Scale (A Dimension Scale cannot have scales.)



Note
    The Dimension Scale `dsid` can be attached to the same dimension more than once, which has no effect. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad4af4f5d5c880c5c93ca950cdf5a01fd)H5DSdetach_scale()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DSdetach_scale  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _did_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dsid_ ,   
|  | unsigned int |  _idx_ )  
Detach dimension scale `dsid` from the dimension `idx` of dataset `did`. 
* * * 

Parameters
     [in] | did | The dataset   
---|---|---  
[in] | dsid | The scale to be detached   
[in] | idx | The dimension of `did` to detach 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
If possible, deletes association of Dimension Scale `dsid` with dimension `idx` of dataset `did`. This deletes the entries in the [DIMENSION_LIST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a3c3e0049491003d0fb4f085bb8f504db) and [REFERENCE_LIST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a16370c2d086f86916320c34e416cf445) attributes, as defined in [Storage Profile](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#subsec_dim_scales_spec_store) section of the [HDF5 High Level Dimension Scales](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html).
Fails if:
  * Bad arguments
  * The dataset `did` or `dsid` do not exist
  * The `dsid` is not a Dimension Scale
  * `dsid` is not attached to `did`



Note
    A scale may be associated with more than dimension of the same dataset. If so, the detach operation only deletes one of the associations, for `did`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga265aa3dadb5d067c112620ffe01f96d7)H5DSget_label()
H5HL_DLL ssize_t H5DSget_label  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _did_ ,   
---|---|---|---  
|  | unsigned int |  _idx_ ,   
|  | char * |  _label_ ,   
|  | size_t |  _size_ )  
Read the label for dimension `idx` of `did` into buffer `label`. 
* * * 

Parameters
     [in] | did | The dataset   
---|---|---  
[in] | idx | The dimension   
[out] | label | The label   
[in] | size | The length of the label buffer 

Returns
    Upon success, size of label or zero if no label found. Negative if fail.  
Returns the value of the [DIMENSION_LABELS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a2938efb8ad35991157e48a19900eae61) for dimension `idx` of dataset `did`, if set. Up to `size` characters of the name are copied into the buffer `label`. If the label is longer than `size`, it will be truncated to fit. The parameter `size` is set to the size of the returned `label`.
If `did` has no label, the return value of `label` is NULL.
Fails if:
  * Bad arguments 


##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga562b91296f739a1833a1e1cfc64c7726)H5DSget_num_scales()
H5HL_DLL int H5DSget_num_scales  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _did_ ,   
---|---|---|---  
|  | unsigned int |  _idx_ )  
Determines how many Dimension Scales are attached to dimension `idx` of `did`. 
* * * 

Parameters
     [in] | did | The dataset to query   
---|---|---  
[in] | idx | The dimension of `did` to query 

Returns
    Returns the number of Dimension Scales associated with `did`, if successful, otherwise returns a negative value.  
[H5DSget_num_scales()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga562b91296f739a1833a1e1cfc64c7726 "Determines how many Dimension Scales are attached to dimension idx of did.") determines how many Dimension Scales are attached to dimension `idx` of dataset `did`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga704bbb7eb2c9696f0f2c84e0267e4ffa)H5DSget_scale_name()
H5HL_DLL ssize_t H5DSget_scale_name  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _did_ ,   
---|---|---|---  
|  | char * |  _name_ ,   
|  | size_t |  _size_ )  
Retrieves name of scale `did` into buffer `name`. 
* * * 

Parameters
     [in] | did | Dimension scale identifier   
---|---|---  
[out] | name | Buffer to contain the returned name   
[in] | size | Size in bytes, of the `name` buffer 

Returns
    Upon success, the length of the scale name or zero if no name found. Negative if fail.  
[H5DSget_scale_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga704bbb7eb2c9696f0f2c84e0267e4ffa "Retrieves name of scale did into buffer name.") retrieves the name attribute for scale `did`.
Up to `size` characters of the scale name are returned in `name`; additional characters, if any, are not returned to the user application.
If the length of the name, which determines the required value of `size`, is unknown, a preliminary [H5DSget_scale_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga704bbb7eb2c9696f0f2c84e0267e4ffa "Retrieves name of scale did into buffer name.") call can be made by setting `name` to NULL. The return value of this call will be the size of the scale name; that value plus one (1) can then be assigned to `size` for a second [H5DSget_scale_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga704bbb7eb2c9696f0f2c84e0267e4ffa "Retrieves name of scale did into buffer name.") call, which will retrieve the actual name. (The value passed in with the parameter `size` must be one greater than size in bytes of the actual name in order to accommodate the null terminator; if `size` is set to the exact size of the name, the last byte passed back will contain the null terminator and the last character will be missing from the name passed back to the calling application.) 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad7bbd73a68b031f6ac4c305333b0c8b7)H5DSis_attached()
H5HL_DLL [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5DSis_attached  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _did_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dsid_ ,   
|  | unsigned int |  _idx_ )  
Report if dimension scale `dsid` is currently attached to dimension `idx` of dataset `did`. 
* * * 

Parameters
     [in] | did | The dataset   
---|---|---  
[in] | dsid | The scale to be attached   
[in] | idx | The dimension of `did` that `dsid` is associated with 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
Report if dimension scale `dsid` is currently attached to dimension `idx` of dataset `did`.
Fails if:
  * Bad arguments
  * If `dsid` is not a Dimension Scale
  * The `dsid` is not a Dimension Scale
  * If `did` is a Dimension Scale (A Dimension Scale cannot have scales.) 


##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaf6c043d90b502ecf96783e8f50cadcd5)H5DSis_scale()
H5HL_DLL [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5DSis_scale  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _did_ | ) |   
---|---|---|---|---|---  
Determines whether `did` is a Dimension Scale. 
* * * 

Parameters
     [in] | did | The dataset to query  
---|---|--- 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5DSis_scale()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaf6c043d90b502ecf96783e8f50cadcd5 "Determines whether did is a Dimension Scale.") determines if `did` is a Dimension Scale, i.e., has class="DIMENSION_SCALE"). 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga7cb78620d05e7dcb81a9f53a8cc496f5)H5DSiterate_scales()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DSiterate_scales  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _did_ ,   
---|---|---|---  
|  | unsigned int |  _dim_ ,   
|  | int * |  _idx_ ,   
|  | [H5DS_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a7979d0cd5a423ae1cbccd0b65600bdf6) |  _visitor_ ,   
|  | void * |  _visitor_data_ )  
Iterates the operation visitor through the scales attached to dimension `dim`. 
* * * 

Parameters
     [in] | did | The dataset   
---|---|---  
[in] | dim | The dimension of dataset `did`  
[in,out] | idx |  [Input](https://support.hdfgroup.org/documentation/hdf5/latest/struct_input.html) the index to start iterating, output the next index to visit. If NULL, start at the first position.   
[in] | visitor | The visitor function   
[in] | visitor_data | Arbitrary data to pass to the visitor function 

Returns
    Returns the return value of the last operator if it was non-zero, or zero if all scales were processed.  
[H5DSiterate_scales()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga7cb78620d05e7dcb81a9f53a8cc496f5 "Iterates the operation visitor through the scales attached to dimension dim.") iterates over the scales attached to dimension `dim` of dataset `did`. For each scale in the list, the `visitor_data` and some additional information, specified below, are passed to the `visitor` function. The iteration begins with the `idx` object in the group and the next element to be processed by the operator is returned in `idx`. If `idx` is NULL, then the iterator starts at the first group member; since no stopping point is returned in this case, the iterator cannot be restarted if one of the calls to its operator returns non-zero.
The prototype for [H5DS_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a7979d0cd5a423ae1cbccd0b65600bdf6) is: 
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*[H5DS_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a7979d0cd5a423ae1cbccd0b65600bdf6))([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset, unsigned dim, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) scale, void *visitor_data);
The operation receives the Dimension Scale dataset identifier, `scale`, and the pointer to the operator data passed in to [H5DSiterate_scales()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga7cb78620d05e7dcb81a9f53a8cc496f5 "Iterates the operation visitor through the scales attached to dimension dim."), `visitor_data`.
The return values from an operator are:
  * Zero causes the iterator to continue, returning zero when all group members have been processed.
  * Positive causes the iterator to immediately return that positive value, indicating short-circuit success. The iterator can be restarted at the next group member.
  * Negative causes the iterator to immediately return that value, indicating failure. The iterator can be restarted at the next group member.


[H5DSiterate_scales()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga7cb78620d05e7dcb81a9f53a8cc496f5 "Iterates the operation visitor through the scales attached to dimension dim.") assumes that the scales of the dimension identified by `dim` remain unchanged through the iteration. If the membership changes during the iteration, the function's behavior is undefined. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga10e953a37cf817f491fe47acb1370d88)H5DSset_label()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DSset_label  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _did_ ,   
---|---|---|---  
|  | unsigned int |  _idx_ ,   
|  | const char * |  _label_ )  
Set label for the dimension `idx` of `did` to the value `label`. 
* * * 

Parameters
     [in] | did | The dataset   
---|---|---  
[in] | idx | The dimension   
[in] | label | The label 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Sets the [DIMENSION_LABELS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a2938efb8ad35991157e48a19900eae61) for dimension `idx` of dataset `did`. If the dimension had a label, the new value replaces the old.
Fails if:
  * Bad arguments 


##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaca0f37c36c4e1288d161d26daa3a41d3)H5DSset_scale()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DSset_scale  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dsid_ ,   
---|---|---|---  
|  | const char * |  _dimname_ )  
Convert dataset `dsid` to a dimension scale, with optional name, `dimname`. 
* * * 

Parameters
     [in] | dsid | The dataset to be made a Dimemsion Scale   
---|---|---  
[in] | dimname | The dimension name (optional), NULL if the dimension has no name. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The dataset `dsid` is converted to a Dimension Scale dataset, as defined above. Creates the CLASS attribute, set to the value "DIMENSION_SCALE" and an empty [REFERENCE_LIST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_spublic_8h.html#a16370c2d086f86916320c34e416cf445) attribute, as described in [Storage Profile](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#subsec_dim_scales_spec_store) section of the [HDF5 High Level Dimension Scales](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html).
If `dimname` is specified, then an attribute called NAME is created, with the value `dimname`.
Fails if:
  * Bad arguments
  * If `dsid` is already a scale
  * If `dsid` is a dataset which already has dimension scales


If the dataset was created with the Table, Image, or Palette interface [9], it is not recommended to convert to a Dimension Scale. (These Datasets will have a CLASS Table, Image, or Palette.) 

**[Todo](https://support.hdfgroup.org/documentation/hdf5/latest/todo.html#_todo000003)**
    what is [9] after Palette interface? 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gafd2a9719c14946b1478e3e79736b3094)H5DSwith_new_ref()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DSwith_new_ref  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | bool * |  _with_new_ref_ )  
Determines if new references are used with dimension scales. 
* * * 

Parameters
     [in] | obj_id | Object identifier   
---|---|---  
[out] | with_new_ref | New references are used or not 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5DSwith_new_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gafd2a9719c14946b1478e3e79736b3094 "Determines if new references are used with dimension scales.") takes any object identifier and checks if new references are used for dimension scales. Currently, new references are used when non-native VOL connector is used or when H5_DIMENSION_SCALES_WITH_NEW_REF is set up via configure option. 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


