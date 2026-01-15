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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title0 "H2")
  * [ Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title2 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title3 "H2")
  * [◆ H5Idec_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title4 "H2")
  * [◆ H5Iget_file_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title5 "H2")
  * [◆ H5Iget_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title6 "H2")
  * [◆ H5Iget_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title7 "H2")
  * [◆ H5Iget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title8 "H2")
  * [◆ H5Iinc_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title9 "H2")
  * [◆ H5Iis_valid()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title10 "H2")
  * [◆ H5Iregister_future()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#groups) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#func-members)
Identifiers (H5I)
## Detailed Description
Use the functions in this module to manage identifiers defined by the HDF5 library. See [User-defined ID Types](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html) for user-defined identifiers and identifier types.
HDF5 identifiers are usually created as a side-effect of creating HDF5 entities such as groups, datasets, attributes, or property lists.
Identifiers defined by the HDF5 library can be used to retrieve information such as path names and reference counts, and their validity can be checked.
Identifiers can be updated by manipulating their reference counts.
Unused identifiers should be reclaimed by closing the associated item, e.g., HDF5 object, or decrementing the reference count to 0. 

Note
    Identifiers (of type [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)) are run-time auxiliaries and not persisted in the file. Create | Read   
---|---  
15 { 16 __label__ fail_dcpl; 17 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl; 18 19 // create an ID of a pre-defined ID type 20 if ((dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 21 ret_val = EXIT_FAILURE; 22 goto fail_dcpl; 23 } 24 25 // we can reliably predict the ID type 26 assert([H5Iget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga4941435d4d64de3d7095d2316f415f2d)(dcpl) == [H5I_GENPROP_LST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832bab6161706783d4bca26a889f1ac0cf91a)); 27 printf("%d\n", [H5Iget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga4941435d4d64de3d7095d2316f415f2d)(dcpl)); 28 29 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(dcpl); 30fail_dcpl:; 31 } |  35 { 36 __label__ fail_dcpl; 37 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl; 38 39 // create an ID of a pre-defined ID type 40 if ((dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 41 ret_val = EXIT_FAILURE; 42 goto fail_dcpl; 43 } 44 45 // this better be valid 46 assert([H5Iis_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a)(dcpl) > 0); 47 48 // this ID is NOT associated with any HDF5 file; 49 // see the error message 50 assert([H5Iget_file_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga6ce32e88051e4cdf7d02439d86e5e042)(dcpl) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)); 51 52 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(dcpl); 53fail_dcpl:; 54 }  
Update | Delete   
58 { 59 __label__ fail_rc, fail_dcpl; 60 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl; 61 int rc; 62 63 // create an ID of a pre-defined ID type 64 if ((dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 65 ret_val = EXIT_FAILURE; 66 goto fail_dcpl; 67 } 68 69 // retrieve the IDs reference count 70 if ((rc = [H5Iget_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gac9638ade14cc75b7b125b3723f319c81)(dcpl)) < 0) { 71 ret_val = EXIT_FAILURE; 72 goto fail_rc; 73 } 74 printf("Reference count: %d\n", rc); 75 76 // bump its reference count (by 1) 77 if ((rc = [H5Iinc_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga3fd0c157704573965cafd6e1aa7f368f)(dcpl)) < 0) { 78 ret_val = EXIT_FAILURE; 79 goto fail_rc; 80 } 81 printf("Reference count: %d\n", rc); 82 83fail_rc: 84 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(dcpl); 85fail_dcpl:; 86 } |  90 { 91 __label__ fail_rc, fail_dcpl; 92 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl; 93 int rc; 94 95 // create an ID of a pre-defined ID type 96 if ((dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 97 ret_val = EXIT_FAILURE; 98 goto fail_dcpl; 99 } 100 101 // decrease its reference count (from 1) to 0 102 if ((rc = [H5Idec_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gaea2aa78caea892edf2a6a6ac70486ed9)(dcpl)) < 0) { 103 ret_val = EXIT_FAILURE; 104 goto fail_rc; 105 } 106 printf("Reference count: %d\n", rc); 107 108 // at this point, the ID is no longer valid 109 assert([H5Iis_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a)(dcpl) == 0); 110 111 // dropping the ref. count to 0 has the side-effect of closing 112 // the associated item, and calling H5Pclose would create an error 113 goto fail_dcpl; 114 115fail_rc: 116 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(dcpl); 117fail_dcpl:; 118 }  
##  Topics  
---  
| [User-defined ID Types](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html)  
##  Functions  
---  
int  |  [H5Idec_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gaea2aa78caea892edf2a6a6ac70486ed9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id)  
| Decrements the reference count for an object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Iget_file_id](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga6ce32e88051e4cdf7d02439d86e5e042) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id)  
| Retrieves an identifier for the file containing the specified object.   
  
ssize_t  |  [H5Iget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id, char *name, size_t size)  
| Retrieves a name of an object based on the object identifier.   
  
int  |  [H5Iget_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gac9638ade14cc75b7b125b3723f319c81) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id)  
| Retrieves the reference count for an object.   
  
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  [H5Iget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga4941435d4d64de3d7095d2316f415f2d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id)  
| Retrieves the type of an object.   
  
int  |  [H5Iinc_ref](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga3fd0c157704573965cafd6e1aa7f368f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id)  
| Increments the reference count for an object.   
  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Iis_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) id)  
| Determines whether an identifier is valid.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Iregister_future](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gad61703cd16f392fb194da648099e69a9) ([H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) type, const void *object, [H5I_future_realize_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_idevelop_8h.html#ab0f349ec701c3d8c6ded067f160da44c) realize_cb, [H5I_future_discard_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_idevelop_8h.html#a127fbac3b8283a7804d8470676a65518) discard_cb)  
| Registers a "future" object under a type and returns an ID for it.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gaea2aa78caea892edf2a6a6ac70486ed9)H5Idec_ref()
int H5Idec_ref  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _id_ | ) |   
---|---|---|---|---|---  
Decrements the reference count for an object.  

Parameters
     [in] | id | Object identifier  
---|---|--- 

Returns
    Returns a non-negative reference count of the object ID after decrementing it, if successful; otherwise a negative value is returned.  
[H5Idec_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gaea2aa78caea892edf2a6a6ac70486ed9 "Decrements the reference count for an object.") decrements the reference count of the object identified by `id`.
The reference count for an object ID is attached to the information about an object in memory and has no relation to the number of links to an object on disk.
The reference count for a newly created object will be 1. Reference counts for objects may be explicitly modified with this function or with [H5Iinc_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga3fd0c157704573965cafd6e1aa7f368f "Increments the reference count for an object."). When an object identifier's reference count reaches zero, the object will be closed. Calling an object identifier's `close` function decrements the reference count for the identifier, which normally closes the object, but if the reference count for the identifier has been incremented with [H5Iinc_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga3fd0c157704573965cafd6e1aa7f368f "Increments the reference count for an object."), the object will only be closed when the reference count reaches zero with further calls to this function or the object identifier's `close` function.
If the object ID was created by a collective parallel call (such as [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf), [H5Gopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245), etc.), the reference count should be modified by all the processes which have copies of the ID. Generally, this means that group, dataset, attribute, file and named datatype IDs should be modified by all the processes and that all other types of IDs are safe to modify by individual processes.
This function is of particular value when an application maintains multiple copies of an object ID. The object ID can be incremented when a copy is made. Each copy of the ID can then be safely closed or decremented and the HDF5 object will be closed when the reference count for that object drops to zero. 

Since
    1.6.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga6ce32e88051e4cdf7d02439d86e5e042)H5Iget_file_id()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Iget_file_id  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _id_ | ) |   
---|---|---|---|---|---  
Retrieves an identifier for the file containing the specified object.  

Parameters
     [in] | id | Object identifier  
---|---|--- 

Returns
    Returns a file identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Iget_file_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga6ce32e88051e4cdf7d02439d86e5e042 "Retrieves an identifier for the file containing the specified object.") returns the identifier of the file associated with the object referenced by `id`. 

Note
    Note that the HDF5 library permits an application to close a file while objects within the file remain open. If the file containing the object `id` is still open, [H5Iget_file_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga6ce32e88051e4cdf7d02439d86e5e042 "Retrieves an identifier for the file containing the specified object.") will retrieve the existing file identifier. If there is no existing file identifier for the file, i.e., the file has been closed, [H5Iget_file_id()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga6ce32e88051e4cdf7d02439d86e5e042 "Retrieves an identifier for the file containing the specified object.") will reopen the file and return a new file identifier. In either case, the file identifier must eventually be released using [H5Fclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file."). 

Since
    1.6.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1)H5Iget_name()
ssize_t H5Iget_name  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _id_ ,   
---|---|---|---  
|  | char * |  _name_ ,   
|  | size_t |  _size_ )  
Retrieves a name of an object based on the object identifier.  

Parameters
     [in] | id | Object identifier   
---|---|---  
[out] | name | A buffer for the name associated with the identifier   
[in] | size | The size, in bytes, of the `name` buffer. Must be the size of the object name in bytes plus 1 for a NULL terminator 

Returns
    ssize_t  
[H5Iget_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1 "Retrieves a name of an object based on the object identifier.") retrieves a name for the object identified by `id`.
Up to size characters of the name are returned in `name`; additional characters, if any, are not returned to the user application.
Up to `size` characters of the object name are returned in `name`; additional characters, if any, are not returned to the user application.  
  
If the length of the object name, which determines the required value of `size`, is unknown, a preliminary call to [H5Iget_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1 "Retrieves a name of an object based on the object identifier.") with the last two parameters set to NULL and zero respectively can be made. The return value of this call will be the size in bytes of the object name. That value, plus 1 for a NULL terminator, must then be assigned to `size` for a second [H5Iget_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1 "Retrieves a name of an object based on the object identifier.") call, which will retrieve the actual object name.
If the object identified by `id` is an attribute, as determined via [H5Iget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga4941435d4d64de3d7095d2316f415f2d "Retrieves the type of an object."), [H5Iget_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1 "Retrieves a name of an object based on the object identifier.") retrieves the name of the object to which that attribute is attached. To retrieve the name of the attribute itself, use [H5Aget_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga05e195aabab8c623b1c52009aeb99674 "Gets an attribute name.").
If there is no name associated with the object identifier or if the name is NULL, [H5Iget_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1 "Retrieves a name of an object based on the object identifier.") returns 0 (zero). 

Note
    Note that an object in an HDF5 file may have multiple paths if there are multiple links pointing to it. This function may return any one of these paths. When possible, [H5Iget_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga9c84a8dc29566b82b6d1ff7a6e6828f1 "Retrieves a name of an object based on the object identifier.") returns the path with which the object was opened. 

Since
    1.6.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gac9638ade14cc75b7b125b3723f319c81)H5Iget_ref()
int H5Iget_ref  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _id_ | ) |   
---|---|---|---|---|---  
Retrieves the reference count for an object.  

Parameters
     [in] | id | Object identifier  
---|---|--- 

Returns
    Returns a non-negative current reference count of the object identifier if successful; otherwise a negative value is returned.  
[H5Iget_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gac9638ade14cc75b7b125b3723f319c81 "Retrieves the reference count for an object.") retrieves the reference count of the object identified by `id`.
The reference count for an object identifier is attached to the information about an object in memory and has no relation to the number of links to an object on disk.
The function [H5Iis_valid()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a "Determines whether an identifier is valid.") is used to determine whether a specific object identifier is valid. 

Since
    1.6.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga4941435d4d64de3d7095d2316f415f2d)H5Iget_type()
[H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) H5Iget_type  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _id_ | ) |   
---|---|---|---|---|---  
Retrieves the type of an object.  

Parameters
     [in] | id | Object identifier  
---|---|--- 

Returns
    Returns the object type if successful; otherwise [H5I_BADID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba3e61c9654de6398dc9676ad37cbe6133).  
[H5Iget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga4941435d4d64de3d7095d2316f415f2d "Retrieves the type of an object.") retrieves the type of the object identified by `id`. If no valid type can be determined or the identifier submitted is invalid, the function returns [H5I_BADID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832ba3e61c9654de6398dc9676ad37cbe6133).
This function is of particular use in determining the type of object closing function ([H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset."), [H5Gclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221 "Closes the specified group."), etc.) to call after a call to [H5Rdereference()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a). 

Note
    Note that this function returns only the type of object that `id` would identify if it were valid; it does not determine whether `id` is valid identifier. Validity can be determined with a call to [H5Iis_valid()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a "Determines whether an identifier is valid."). 

Since
    1.0.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga3fd0c157704573965cafd6e1aa7f368f)H5Iinc_ref()
int H5Iinc_ref  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _id_ | ) |   
---|---|---|---|---|---  
Increments the reference count for an object.  

Parameters
     [in] | id | Object identifier  
---|---|--- 

Returns
    Returns a non-negative reference count of the object ID after incrementing it if successful; otherwise a negative value is returned.  
[H5Iinc_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga3fd0c157704573965cafd6e1aa7f368f "Increments the reference count for an object.") increments the reference count of the object identified by `id`.
The reference count for an object ID is attached to the information about an object in memory and has no relation to the number of links to an object on disk.
The reference count for a newly created object will be 1. Reference counts for objects may be explicitly modified with this function or with [H5Idec_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gaea2aa78caea892edf2a6a6ac70486ed9 "Decrements the reference count for an object."). When an object ID's reference count reaches zero, the object will be closed. Calling an object ID's `close` function decrements the reference count for the ID which normally closes the object, but if the reference count for the ID has been incremented with this function, the object will only be closed when the reference count reaches zero with further calls to [H5Idec_ref()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gaea2aa78caea892edf2a6a6ac70486ed9 "Decrements the reference count for an object.") or the object ID's `close` function.
If the object ID was created by a collective parallel call (such as [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf), [H5Gopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245), etc.), the reference count should be modified by all the processes which have copies of the ID. Generally this means that group, dataset, attribute, file and named datatype IDs should be modified by all the processes and that all other types of IDs are safe to modify by individual processes.
This function is of particular value when an application is maintaining multiple copies of an object ID. The object ID can be incremented when a copy is made. Each copy of the ID can then be safely closed or decremented and the HDF5 object will be closed when the reference count for that that object drops to zero. 

Since
    1.6.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a)H5Iis_valid()
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5Iis_valid  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _id_ | ) |   
---|---|---|---|---|---  
Determines whether an identifier is valid.  

Parameters
     [in] | id | Object identifier  
---|---|--- 

Returns
    Returns zero (false), a positive (true) or a negative (failure) value.  
[H5Iis_valid()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a "Determines whether an identifier is valid.") determines whether the identifier `id` is valid.
Valid identifiers are those that have been obtained by an application and can still be used to access the original target. Examples of invalid identifiers include: 
  * Out-of-range values: negative, for example 
  * Previously-valid identifiers that have been released: for example, a dataset identifier for which the dataset has been closed


[H5Iis_valid()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga20eb10c559d9ed5ba6f77b31d6a3ba9a "Determines whether an identifier is valid.") can be used with any type of identifier: object identifier, property list identifier, attribute identifier, error message identifier, etc. When necessary, a call to [H5Iget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#ga4941435d4d64de3d7095d2316f415f2d "Retrieves the type of an object.") can determine the type of object that the `id` identifies. 

Since
    1.8.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gad61703cd16f392fb194da648099e69a9)H5Iregister_future()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Iregister_future  | ( | [H5I_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a13afe14178faf81b89fa2167e7ab832b) |  _type_ ,   
---|---|---|---  
|  | const void * |  _object_ ,   
|  | [H5I_future_realize_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_idevelop_8h.html#ab0f349ec701c3d8c6ded067f160da44c) |  _realize_cb_ ,   
|  | [H5I_future_discard_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_idevelop_8h.html#a127fbac3b8283a7804d8470676a65518) |  _discard_cb_ )  
Registers a "future" object under a type and returns an ID for it.  

Parameters
     [in] | type | The identifier of the type of the new ID   
---|---|---  
[in] | object | Pointer to "future" object for which a new ID is created   
[in] | realize_cb | Function pointer to realize a future object   
[in] | discard_cb | Function pointer to destroy a future object 

Returns
    Returns a object identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Iregister_future()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gad61703cd16f392fb194da648099e69a9 "Registers a "future" object under a type and returns an ID for it.") creates and returns a new ID for a "future" object. Future objects are a special kind of object and represent a placeholder for an object that has not yet been created or opened. The `realize_cb` will be invoked by the HDF5 library to 'realize' the future object as an actual object. A call to [H5Iobject_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga82b68eac53ed4c55efd98ab4d0eb9035 "Returns the object referenced by an ID.") will invoke the `realize_cb` callback and if it successfully returns, will return the actual object, not the future object.
The `type` parameter is the identifier for the ID type to which this new future ID will belong. This identifier may have been created by a call to [H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab) or may be one of the HDF5 pre-defined ID classes (e.g. H5I_FILE, H5I_GROUP, H5I_DATASPACE, etc).
The `object` parameter is a pointer to the memory which the new ID will be a reference to. This pointer will be stored by the library, but will not be returned to a call to [H5Iobject_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga82b68eac53ed4c55efd98ab4d0eb9035 "Returns the object referenced by an ID.") until the `realize_cb` callback has returned the actual pointer for the object.
A NULL value for `object` is allowed.
The `realize_cb` parameter is a function pointer that will be invoked by the HDF5 library to convert a future object into an actual object. The `realize_cb` function may be invoked by [H5Iobject_verify()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_u_d.html#ga82b68eac53ed4c55efd98ab4d0eb9035 "Returns the object referenced by an ID.") to return the actual object for a user-defined ID class (i.e. an ID class registered with [H5Iregister_type()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af51df104e1e3ff0b99b7a8ca368f14ab)) or internally by the HDF5 library in order to use or get information from an HDF5 pre-defined ID type. For example, the `realize_cb` for a future dataspace object will be called during the process of returning information from [H5Sget_simple_extent_dims()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3 "Retrieves dataspace dimension size and maximum size.").
Note that although the `realize_cb` routine returns an ID (as a parameter) for the actual object, the HDF5 library will swap the actual object in that ID for the future object in the future ID. This ensures that the ID value for the object doesn't change for the user when the object is realized.
Note that the `realize_cb` callback could receive a NULL value for a future object pointer, if one was used when [H5Iregister_future()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gad61703cd16f392fb194da648099e69a9 "Registers a "future" object under a type and returns an ID for it.") was initially called. This is permitted as a means of allowing the `realize_cb` to act as a generator of new objects, without requiring creation of unnecessary future objects.
It is an error to pass NULL for `realize_cb`.
The `discard_cb` parameter is a function pointer that will be invoked by the HDF5 library to destroy a future object. This callback will always be invoked for _every_ future object, whether the `realize_cb` is invoked on it or not. It's possible that the `discard_cb` is invoked on a future object without the `realize_cb` being invoked, e.g. when a future ID is closed without requiring the future object to be realized into an actual one.
Note that the `discard_cb` callback could receive a NULL value for a future object pointer, if one was used when [H5Iregister_future()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gad61703cd16f392fb194da648099e69a9 "Registers a "future" object under a type and returns an ID for it.") was initially called.
It is an error to pass NULL for `discard_cb`. 

Note
    The [H5Iregister_future()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html#gad61703cd16f392fb194da648099e69a9 "Registers a "future" object under a type and returns an ID for it.") function is primarily targeted at VOL connector authors and is _not_ designed for general-purpose application use. 

Since
    1.14.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


