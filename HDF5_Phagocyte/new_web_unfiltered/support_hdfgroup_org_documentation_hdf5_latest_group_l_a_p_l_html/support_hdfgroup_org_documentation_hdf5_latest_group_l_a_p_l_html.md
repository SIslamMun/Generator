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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title0 "H2")
  * [ Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title2 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title3 "H2")
  * [◆ H5Pget_elink_acc_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title4 "H2")
  * [◆ H5Pget_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title5 "H2")
  * [◆ H5Pget_elink_fapl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title6 "H2")
  * [◆ H5Pget_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title7 "H2")
  * [◆ H5Pget_nlinks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title8 "H2")
  * [◆ H5Pset_elink_acc_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title9 "H2")
  * [◆ H5Pset_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title10 "H2")
  * [◆ H5Pset_elink_fapl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title11 "H2")
  * [◆ H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title12 "H2")
  * [◆ H5Pset_nlinks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#title13 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#groups) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#func-members)
Link Access Properties
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html)
## Detailed Description
Link access property list functions (H5P) Function  | Purpose   
---|---  
[H5Pset_elink_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2 "Sets the external link traversal callback function in a link access property list.")/[H5Pget_elink_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gacbf576bd8f7e63f3a91134b12d6b2d12 "Retrieves the external link traversal callback function from the specified link access property list.") | Sets/gets the external link traversal callback function.   
[H5Pset_elink_acc_flags](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga020f7eb2eae01043286af50db0a76d82 "Sets the external link traversal file access flag in a link access property list.")/[H5Pget_elink_acc_flags](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaf1357eb0940f171efecae06a9ed6155b "Retrieves the external link traversal file access flag from the specified link access property list.") | Sets/gets the external link traversal file access flag.   
[H5Pset_elink_fapl](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga3895e8e60ce8f0b6f32ab7a22c715d1a "Sets a file access property list for use in accessing a file pointed to by an external link.")/[H5Pget_elink_fapl](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga2c2fe0a0396b9a0a02b28402e4ee108a "Retrieves the file access property list identifier associated with the link access property list.") | Sets/gets a file access property list for use in accessing a file pointed to by an external link   
[H5Pset_elink_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths.")/[H5Pget_elink_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga7960f746797bcf35f70746cd644f8b5a "Retrieves prefix applied to external link paths.") | Sets/gets prefix to be applied to external link paths.   
[H5Pset_nlinks](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaa46c63c196a0cf5cd94dede039c030f4 "Sets maximum number of soft or user-defined link traversals.")/[H5Pget_nlinks](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga6bfa33fa9a77011cbdc06d0fbc907177 "Retrieves the maximum number of link traversals.") | Sets/gets maximum number of soft or user-defined link traversals.   
##  Topics  
---  
| [Dataset Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html)  
| [Datatype Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_a_p_l.html)  
| [Group Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_elink_acc_flags](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaf1357eb0940f171efecae06a9ed6155b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, unsigned *flags)  
| Retrieves the external link traversal file access flag from the specified link access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_elink_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gacbf576bd8f7e63f3a91134b12d6b2d12) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [H5L_elink_traverse_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76) *func, void **op_data)  
| Retrieves the external link traversal callback function from the specified link access property list.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Pget_elink_fapl](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga2c2fe0a0396b9a0a02b28402e4ee108a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Retrieves the file access property list identifier associated with the link access property list.   
  
ssize_t  |  [H5Pget_elink_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga7960f746797bcf35f70746cd644f8b5a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, char *[prefix](https://support.hdfgroup.org/documentation/hdf5/latest/h5dump_8h.html#ad2849cf781a4db22cc1b31eaaee50a4f), size_t size)  
| Retrieves prefix applied to external link paths.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_nlinks](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga6bfa33fa9a77011cbdc06d0fbc907177) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, size_t *nlinks)  
| Retrieves the maximum number of link traversals.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_elink_acc_flags](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga020f7eb2eae01043286af50db0a76d82) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, unsigned flags)  
| Sets the external link traversal file access flag in a link access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_elink_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [H5L_elink_traverse_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76) func, void *op_data)  
| Sets the external link traversal callback function in a link access property list.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_elink_fapl](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga3895e8e60ce8f0b6f32ab7a22c715d1a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id)  
| Sets a file access property list for use in accessing a file pointed to by an external link.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_elink_prefix](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, const char *[prefix](https://support.hdfgroup.org/documentation/hdf5/latest/h5dump_8h.html#ad2849cf781a4db22cc1b31eaaee50a4f))  
| Sets prefix to be applied to external link paths.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_nlinks](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaa46c63c196a0cf5cd94dede039c030f4) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id, size_t nlinks)  
| Sets maximum number of soft or user-defined link traversals.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaf1357eb0940f171efecae06a9ed6155b)H5Pget_elink_acc_flags()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_elink_acc_flags  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
---|---|---|---  
|  | unsigned * |  _flags_ )  
Retrieves the external link traversal file access flag from the specified link access property list.  

Parameters
     [in] | lapl_id | Link access property list identifier   
---|---|---  
[out] | flags | File access flag for link traversal 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_elink_acc_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaf1357eb0940f171efecae06a9ed6155b "Retrieves the external link traversal file access flag from the specified link access property list.") retrieves the file access flag used to open an external link target file from the specified link access property list.
Valid values for `flags` include: 
  * [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4) - Files opened through external links will be opened with write access 
  * [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) - Files opened through external links will be opened with read-only access 
  * [H5F_ACC_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#ab0ce75eb6c23c77bf2736d53e2ea5dce) - Files opened through external links will be opened with the same access flag as the parent file


The value returned, if it is not [H5F_ACC_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#ab0ce75eb6c23c77bf2736d53e2ea5dce), will override the default access flag, which is the access flag used to open the parent file.
**Example Usage:** The following code retrieves the external link access flag settings on the link access property list `lapl_id` into a local variable: 
```
        unsigned acc_flags;
        status = H5Pget_elink_acc_flags(lapl_id, &acc_flags);
      
```


Since
    1.8.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gacbf576bd8f7e63f3a91134b12d6b2d12)H5Pget_elink_cb()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_elink_cb  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
---|---|---|---  
|  |  [H5L_elink_traverse_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76) * |  _func_ ,   
|  | void ** |  _op_data_ )  
Retrieves the external link traversal callback function from the specified link access property list.  

Parameters
     [in] | lapl_id | Link access property list identifier   
---|---|---  
[out] | func | User-defined external link traversal callback function   
[out] | op_data | User-defined input data for the callback function 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gacbf576bd8f7e63f3a91134b12d6b2d12 "Retrieves the external link traversal callback function from the specified link access property list.") retrieves the user-defined external link traversal callback function defined in the specified link access property list.
The callback function may adjust the file access property list and file access flag to use when opening a file through an external link. The callback will be executed by the HDF5 library immediately before opening the target file.
**Failure Modes:** [H5Pget_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gacbf576bd8f7e63f3a91134b12d6b2d12 "Retrieves the external link traversal callback function from the specified link access property list.") will fail if the link access property list identifier, `lapl_id`, is invalid.
An invalid function pointer or data pointer, `func` or `op_data` respectively, may cause a segmentation fault or an invalid memory access.
**Example Usage:** The following code retrieves the external link callback settings on the link access property list `lapl_id` into local variables: 
```
      H5L_elink_traverse_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76 "Callback for external link traversal.") elink_callback_func;
      void *elink_callback_udata;
      status = H5Pget_elink_cb (lapl_id, &elink_callback_func,
                                &elink_callback_udata);
      
```


Since
    1.8.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga2c2fe0a0396b9a0a02b28402e4ee108a)H5Pget_elink_fapl()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Pget_elink_fapl  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _lapl_id_ | ) |   
---|---|---|---|---|---  
Retrieves the file access property list identifier associated with the link access property list.  

Parameters
     [in] | lapl_id | Link access property list identifier  
---|---|--- 

Returns
    Returns a file access property list identifier if successful; otherwise returns [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba).  
[H5Pget_elink_fapl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga2c2fe0a0396b9a0a02b28402e4ee108a "Retrieves the file access property list identifier associated with the link access property list.") retrieves the file access property list identifier that is set for the link access property list identifier, `lapl_id`. The library uses this file access property list identifier to open the target file for the external link access. When no such identifier is set, this routine returns [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f). 

See also
     [H5Pset_elink_fapl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga3895e8e60ce8f0b6f32ab7a22c715d1a "Sets a file access property list for use in accessing a file pointed to by an external link.") and [H5Lcreate_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file."). 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga7960f746797bcf35f70746cd644f8b5a)H5Pget_elink_prefix()
ssize_t H5Pget_elink_prefix  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | char * |  _prefix_ ,   
|  | size_t |  _size_ )  
Retrieves prefix applied to external link paths.  

Parameters
     [in] | plist_id | Link access property list identifier   
---|---|---  
[out] | prefix | Prefix applied to external link paths   
[in] | size | Size of prefix, including null terminator 

Returns
    If successful, returns a non-negative value specifying the size in bytes of the prefix without the NULL terminator; otherwise returns a negative value.  
[H5Pget_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga7960f746797bcf35f70746cd644f8b5a "Retrieves prefix applied to external link paths.") retrieves the prefix applied to the path of any external links traversed.
When an external link is traversed, the prefix is retrieved from the link access property list `plist_id`, returned in the user-allocated buffer pointed to by `prefix`, and prepended to the filename stored in the external link.
The size in bytes of the prefix, including the NULL terminator, is specified in `size`. If size is unknown, a preliminary [H5Pget_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga7960f746797bcf35f70746cd644f8b5a "Retrieves prefix applied to external link paths.") call with the pointer `prefix` set to NULL will return the size of the prefix without the NULL terminator. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga6bfa33fa9a77011cbdc06d0fbc907177)H5Pget_nlinks()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pget_nlinks  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | size_t * |  _nlinks_ )  
Retrieves the maximum number of link traversals.  

Parameters
     [in] | plist_id | Link access property list identifier   
---|---|---  
[out] | nlinks | Maximum number of links to traverse 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pget_nlinks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga6bfa33fa9a77011cbdc06d0fbc907177 "Retrieves the maximum number of link traversals.") retrieves the maximum number of soft or user-defined link traversals allowed, `nlinks`, before the library assumes it has found a cycle and aborts the traversal. This value is retrieved from the link access property list `plist_id`.
The limit on the number of soft or user-defined link traversals is designed to terminate link traversal if one or more links form a cycle. User control is provided because some files may have legitimate paths formed of large numbers of soft or user-defined links. This property can be used to allow traversal of as many links as desired. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga020f7eb2eae01043286af50db0a76d82)H5Pset_elink_acc_flags()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_elink_acc_flags  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
---|---|---|---  
|  | unsigned |  _flags_ )  
Sets the external link traversal file access flag in a link access property list.  

Parameters
     [in] | lapl_id | Link access property list identifier   
---|---|---  
[in] | flags | The access flag for external link traversal 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_elink_acc_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga020f7eb2eae01043286af50db0a76d82 "Sets the external link traversal file access flag in a link access property list.") specifies the file access flag to use to open the target file of an external link. This allows read-only access of files reached through an external link in a file opened with write access, or vice-versa.
Valid values for `flags` include: 
  * [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4) - Causes files opened through external links to be opened with write access 
  * [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394) - Causes files opened through external links to be opened with read-only access 
  * [H5F_ACC_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#ab0ce75eb6c23c77bf2736d53e2ea5dce) - Removes any external link file access flag setting from `lapl_id`, causing the file access flag setting to be taken from the parent file


The library will normally use the file access flag used to open the parent file as the file access flag for the target file. This function provides a way to override that behavior. The external link traversal callback function set by [H5Pset_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2 "Sets the external link traversal callback function in a link access property list.") can override the setting from [H5Pset_elink_acc_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga020f7eb2eae01043286af50db0a76d82 "Sets the external link traversal file access flag in a link access property list.").
**Motivation:** [H5Pset_elink_acc_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga020f7eb2eae01043286af50db0a76d82 "Sets the external link traversal file access flag in a link access property list.") is used to adjust the file access flag used to open files reached through external links. This may be useful to, for example, prevent modifying files accessed through an external link. Otherwise, the target file is opened with whatever flag was used to open the parent.
**Example Usage:** The following code sets the link access property list `lapl_id` to open external link target files with read-only access: 
```
        status = H5Pset_elink_acc_flags(lapl_id, H5F_ACC_RDONLY);
       
```


Since
    1.8.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2)H5Pset_elink_cb()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_elink_cb  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
---|---|---|---  
|  | [H5L_elink_traverse_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76) |  _func_ ,   
|  | void * |  _op_data_ )  
Sets the external link traversal callback function in a link access property list.  

Parameters
     [in] | lapl_id | Link access property list identifier   
---|---|---  
[in] | func | User-defined external link traversal callback function   
[in] | op_data | User-defined input data for the callback function 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2 "Sets the external link traversal callback function in a link access property list.") sets a user-defined external link traversal callback function in the link access property list `lapl_id`. The callback function `func` must conform to the prototype specified in [H5L_elink_traverse_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76 "Callback for external link traversal.").
The callback function may adjust the file access property list and file access flags to use when opening a file through an external link. The callback will be executed by the HDF5 library immediately before opening the target file.
The callback will be made after the file access property list set by [H5Pset_elink_fapl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga3895e8e60ce8f0b6f32ab7a22c715d1a "Sets a file access property list for use in accessing a file pointed to by an external link.") and the file access flag set by [H5Pset_elink_acc_flags()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga020f7eb2eae01043286af50db0a76d82 "Sets the external link traversal file access flag in a link access property list.") are applied, so changes made by this callback function will take precedence. 

Attention
    A file close degree property setting ([H5Pset_fclose_degree()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c "Sets the file close degree.")) in this callback function or an associated property list will be ignored. A file opened by means of traversing an external link is always opened with the weak file close degree property setting, [H5F_CLOSE_WEAK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475faea1127311a219b44e4af3cb12609035f).
**Motivation:** [H5Pset_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2 "Sets the external link traversal callback function in a link access property list.") is used to specify a callback function that is executed by the HDF5 library when traversing an external link. This provides a mechanism to set specific access permissions, modify the file access property list, modify the parent or target file, or take any other user-defined action. This callback function is used in situations where the HDF5 library's default behavior is not suitable.
**Failure Modes:** [H5Pset_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2 "Sets the external link traversal callback function in a link access property list.") will fail if the link access property list identifier, `lapl_id`, is invalid or if the function pointer, `func`, is NULL.
An invalid function pointer, `func`, will cause a segmentation fault or other failure when an attempt is subsequently made to traverse an external link.
**Example Usage:** This example defines a callback function that prints the name of the target file every time an external link is followed, and sets this callback function on `lapl_id`. 
```
         herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) elink_callback(const char *parent_file_name, const char
                *parent_group_name, const char *child_file_name, const char
                *child_object_name, unsigned *acc_flags, hid_t fapl_id, void *op_data) {
             puts(child_file_name);
             return 0;
         }
         int main(void) {
             hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id = H5Pcreate(H5P_LINK_ACCESS);
             H5Pset_elink_cb(lapl_id, elink_callback, NULL);
               ...
         }
         
```


Since
    1.8.3 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga3895e8e60ce8f0b6f32ab7a22c715d1a)H5Pset_elink_fapl()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_elink_fapl  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ )  
Sets a file access property list for use in accessing a file pointed to by an external link.  

Parameters
     [in] | lapl_id | Link access property list identifier   
---|---|---  
[in] | fapl_id | File access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_elink_fapl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga3895e8e60ce8f0b6f32ab7a22c715d1a "Sets a file access property list for use in accessing a file pointed to by an external link.") sets the file access property list, `fapl_id`, to be used when accessing the target file of an external link associated with `lapl_id`. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294)H5Pset_elink_prefix()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_elink_prefix  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | const char * |  _prefix_ )  
Sets prefix to be applied to external link paths.  

Parameters
     [in] | plist_id | Link access property list identifier   
---|---|---  
[in] | prefix | Prefix to be applied to external link paths 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths.") sets the prefix to be applied to the path of any external links traversed. The prefix is prepended to the filename stored in the external link.
The prefix is specified in the user-allocated buffer `prefix` and set in the link access property list `plist_id`. The buffer should not be freed until the property list has been closed. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaa46c63c196a0cf5cd94dede039c030f4)H5Pset_nlinks()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Pset_nlinks  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ ,   
---|---|---|---  
|  | size_t |  _nlinks_ )  
Sets maximum number of soft or user-defined link traversals.  

Parameters
     [in] | plist_id | Link access property list identifier   
---|---|---  
[in] | nlinks | Maximum number of links to traverse 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Pset_nlinks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaa46c63c196a0cf5cd94dede039c030f4 "Sets maximum number of soft or user-defined link traversals.") sets the maximum number of soft or user-defined link traversals allowed, `nlinks`, before the library assumes it has found a cycle and aborts the traversal. This value is set in the link access property list `plist_id`.
The limit on the number of soft or user-defined link traversals is designed to terminate link traversal if one or more links form a cycle. User control is provided because some files may have legitimate paths formed of large numbers of soft or user-defined links. This property can be used to allow traversal of as many links as desired. 

Since
    1.8.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


