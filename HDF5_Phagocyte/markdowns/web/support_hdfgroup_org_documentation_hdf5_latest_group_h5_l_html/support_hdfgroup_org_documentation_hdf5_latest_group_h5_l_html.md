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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title0 "H2")
  * [ Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title1 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title2 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title3 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title4 "H2")
  * [◆ H5Lget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title5 "H2")
  * [◆ H5Lget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title6 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title7 "H2")
  * [◆ H5Lcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title8 "H2")
  * [◆ H5Lcreate_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title9 "H2")
  * [◆ H5Lcreate_hard()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title10 "H2")
  * [◆ H5Lcreate_soft()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title11 "H2")
  * [◆ H5Lcreate_ud()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title12 "H2")
  * [◆ H5Ldelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title13 "H2")
  * [◆ H5Ldelete_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title14 "H2")
  * [◆ H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title15 "H2")
  * [◆ H5Lget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title16 "H2")
  * [◆ H5Lget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title17 "H2")
  * [◆ H5Lget_info_by_idx1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title18 "H2")
  * [◆ H5Lget_info_by_idx2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title19 "H2")
  * [◆ H5Lget_name_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title20 "H2")
  * [◆ H5Lget_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title21 "H2")
  * [◆ H5Lget_val_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title22 "H2")
  * [◆ H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title23 "H2")
  * [◆ H5Lunpack_elink_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#title24 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Topics](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#groups) | [Macros](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#func-members)
Links (H5L)
## Detailed Description
Use the functions in this module to manage HDF5 links and link types.  

See also
     [Link Traversal](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html) for [H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051), [H5Literate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga655b002428e0176c2fa23a0315fbbcc2) and [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69), [H5Lvisit_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga138405315e233673741893e4e250f055)      [Advanced Link Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html) for [H5Lregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class."), [H5Lunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga01ddc889d27306a96a7cd27b6084a5ec "Unregisters a class of user-defined links.") and [H5Lis_registered](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga02c1cf38efea97d18e5e2f65df3f08a2 "Determines whether a class of user-defined links is registered.")
##  Topics  
---  
| [Advanced Link Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html)  
| [Link Traversal](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html)  
##  Macros  
---  
#define  |  [H5Lget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) [H5Lget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400)  
#define  |  [H5Lget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67) [H5Lget_info_by_idx2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_loc, const char *src_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_loc, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates an identical copy of a link with the same creation time and target. The new link can have a different name and be in a different location than the original.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_external](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683) (const char *file_name, const char *obj_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) link_loc_id, const char *link_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates an external link, a soft link to an object in a different file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_hard](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cur_loc, const char *cur_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_loc, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates a hard link to an object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78) (const char *link_target, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) link_loc_id, const char *link_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates a soft link.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_ud](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gadaf9732947c45cd4d2442e7f58873fc2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) link_loc_id, const char *link_name, [H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) link_type, const void *udata, size_t udata_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates a link of a user-defined type.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Removes a link from a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ldelete_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaaf5f820856afdd34f9070a797a246805) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Removes the _n_ -th link in a group.   
  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Lexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Determines whether a link with the specified name exists in a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lget_info1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gacc2ad7f2b402c4bf9bb122d7f43b98dc) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html) *linfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Returns information about a link.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) *linfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Returns information about a link.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lget_info_by_idx1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga7ed207f47e0e0f768f0d540c73e37e2a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html) *linfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Retrieves metadata for a link in a group, according to the order within a field or index.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lget_info_by_idx2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) *linfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Retrieves metadata for a link in a group, according to the order within a field or index.   
  
ssize_t  |  [H5Lget_name_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, char *name, size_t size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Retrieves name of the _n_ -th link in a group, according to the order within a specified field or index.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lget_val](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, void *buf, size_t size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Returns the value of a link.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lget_val_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaf7be56de947e09a8d084e9d13a90bf3c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, void *buf, size_t size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Retrieves value of the _n_ -th link in a group, according to the order within an index.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_loc, const char *src_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_loc, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Moves a link within an HDF5 file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lunpack_elink_val](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220) (const void *ext_linkval, size_t link_size, unsigned *flags, const char **filename, const char **obj_path)  
| Decodes external link information.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698)H5Lget_info
#define H5Lget_info [H5Lget_info2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400)  
---  
[H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) is a macro that is mapped to either [H5Lget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gacc2ad7f2b402c4bf9bb122d7f43b98dc "Returns information about a link.") or [H5Lget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400 "Returns information about a link.").  


See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67)H5Lget_info_by_idx
#define H5Lget_info_by_idx [H5Lget_info_by_idx2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9)  
---  
[H5Lget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67) is a macro that is mapped to either [H5Lget_info_by_idx1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga7ed207f47e0e0f768f0d540c73e37e2a "Retrieves metadata for a link in a group, according to the order within a field or index.") or [H5Lget_info_by_idx2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9 "Retrieves metadata for a link in a group, according to the order within a field or index.").  


See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html)
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6)H5Lcopy()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lcopy  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_loc_ ,   
---|---|---|---  
|  | const char * |  _src_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_loc_ ,   
|  | const char * |  _dst_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Creates an identical copy of a link with the same creation time and target. The new link can have a different name and be in a different location than the original.  

Parameters
     [in] | src_loc | Location identifier. The identifier may be that of a file, group, dataset, or named datatype.   
---|---|---  
[in] | src_name | Name of the link to be copied   
[in] | dst_loc | Location identifier. The identifier may be that of a file, group, dataset, or named datatype.   
[in] | dst_name | Name to be assigned to the new copy   
[in] | lcpl_id | Link creation property list identifier   
[in] | lapl_id | Link access property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6 "Creates an identical copy of a link with the same creation time and target. The new link can have a d...") copies the link specified by `src_name` from the location specified by `src_loc_id` to the location specified by `dst_loc_id`. The new copy of the link is created with the name `dst_name`.
If `dst_loc_id` is a file identifier, `dst_name` will be interpreted relative to that file's root group.
The new link is created with the creation and access property lists specified by `lcpl_id` and `lapl_id`. The interpretation of `lcpl_id` is limited in the manner described in the next paragraph.
[H5Lcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6 "Creates an identical copy of a link with the same creation time and target. The new link can have a d...") retains the creation time and the target of the original link. However, since the link may be renamed, the character encoding is specified in `lcpl_id` rather than in that of the original link. Other link creation properties are ignored.
If the link is a soft link, also known as a symbolic link, its target is interpreted relative to the location of the copy.
Several properties are available to govern the behavior of [H5Lcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6 "Creates an identical copy of a link with the same creation time and target. The new link can have a d..."). These properties are set in the link creation and access property lists, `lcpl_id` and `lapl_id`, respectively. The property controlling creation of missing intermediate groups is set in the link creation property list with [H5Pset_create_intermediate_group()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c "Specifies in property list whether to create missing intermediate groups."); this function ignores any other properties in the link creation property list. Properties controlling character encoding, link traversals, and external link prefixes are set in the link access property list with [H5Pset_char_encoding()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names."), [H5Pset_nlinks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaa46c63c196a0cf5cd94dede039c030f4 "Sets maximum number of soft or user-defined link traversals."), and [H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths."). 

Note
     [H5Lcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6 "Creates an identical copy of a link with the same creation time and target. The new link can have a d...") does not affect the object that the link points to. 

Attention
     [H5Lcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6 "Creates an identical copy of a link with the same creation time and target. The new link can have a d...") cannot copy hard links across files as a hard link is not valid without a target object; to copy objects from one file to another, see [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file."). 

See also
    [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.") 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683)H5Lcreate_external()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lcreate_external  | ( | const char * |  _file_name_ ,   
---|---|---|---  
|  | const char * |  _obj_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _link_loc_id_ ,   
|  | const char * |  _link_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Creates an external link, a soft link to an object in a different file.  

Parameters
     [in] | file_name | Name of the target file containing the target object.   
---|---|---  
[in] | obj_name | Path within the target file to the target object   
[in] | link_loc_id | Location identifier. The identifier may be that of a file, group, dataset, or named datatype.   
[in] | link_name | Name of the new link, relative to `link_loc_id`  
[in] | lcpl_id | Link creation property list identifier   
[in] | lapl_id | Link access property list identifier  

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lcreate_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.") creates a new external link. An external link is a soft link to an object in a different HDF5 file from the location of the link, i.e., to an external object.
`file_name` identifies the target file containing the target object; `obj_name` specifies the path of the target object within that file. `obj_name` must be an absolute pathname in `file_name`, i.e., it must start at the target file's root group, but it is not interpreted until an application attempts to traverse it.
`link_loc_id` and `link_name` specify the location and name, respectively, of the new link. `link_name` is interpreted relative to `link_loc_id`. If `link_name` begins with `/`, then it is interpreted as an absolute path in the file. The name of the created link will be the last element of the provided path. Prior elements in the path are used to locate the parent group of the new link.
`lcpl_id` is the link creation property list used in creating the new link.
`lapl_id` is the link access property list used in traversing the new link. Note that an external file opened by the traversal of an external link is always opened with the weak file close degree property setting, [H5F_CLOSE_WEAK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475faea1127311a219b44e4af3cb12609035f) (see [H5Pset_fclose_degree()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c "Sets the file close degree.")); any file close degree property setting in `lapl_id` is ignored.
An external link behaves similarly to a soft link, and like a soft link in an HDF5 file, it may dangle: the target file and object need not exist at the time that the external link is created.
When the external link `link_name` is accessed, the library will search for the target file `file_name` as described below:
  * If `file_name` is a relative pathname, the following steps are performed:
    * The library will get the [prefix(es)](https://support.hdfgroup.org/documentation/hdf5/latest/h5dump_8h.html#ad2849cf781a4db22cc1b31eaaee50a4f) set in the environment variable `HDF5_EXT_PREFIX` and will try to prepend each prefix to `file_name` to form a new `file_name`.
    * If the new `file_name` does not exist or if `HDF5_EXT_PREFIX` is not set, the library will get the prefix set via [H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths.") and prepend it to `file_name` to form a new `file_name`.
    * If the new `file_name` does not exist or no prefix is being set by [H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths."), then the path of the file associated with `link_loc_id` is obtained. This path can be the absolute path or the current working directory plus the relative path of that file when it is created/opened. The library will prepend this path to `file_name` to form a new `file_name`.
    * If the new `file_name` does not exist, then the library will look for `file_name` and will return failure/success accordingly.
  * If `file_name` is an absolute pathname, the library will first try to find `file_name`. If `file_name` does not exist, `file_name` is stripped of directory paths to form a new `file_name`. The search for the new `file_name` then follows the same steps as described above for a relative pathname. See examples below illustrating how target_file_name is stripped to form a new `file_name`.


Note that `file_name` is considered to be an absolute pathname when the following condition is true:
  * For Unix, the first character of `file_name` is a slash (`/`). For example, consider a `file_name` of `/tmp/A`.h5. If that target file does not exist, the new `file_name` after stripping will be `A.h5`.
  * For Windows, there are 6 cases:
    1. `file_name` is an absolute drive with an absolute pathname. For example, consider a `file_name` of `/tmp/A`.h5. If that target file does not exist, the new `file_name` after stripping will be `A.h5`.
    2. `file_name` is an absolute pathname without specifying drive name. For example, consider a `file_name` of `/tmp/A`.h5. If that target file does not exist, the new `file_name` after stripping will be `A.h5`.
    3. `file_name` is an absolute drive with a relative pathname. For example, consider a `file_name` of `/tmp/A`.h5. If that target file does not exist, the new `file_name` after stripping will be `tmp\A.h5`.
    4. `file_name` is in UNC (Uniform Naming Convention) format with a server name, share name, and pathname. For example, consider a `file_name` of `/tmp/A`.h5. If that target file does not exist, the new `file_name` after stripping will be `A.h5`.
    5. `file_name` is in Long UNC (Uniform Naming Convention) format with a server name, share name, and pathname. For example, consider a `file_name` of `/tmp/A`.h5. If that target file does not exist, the new `file_name` after stripping will be `A.h5`.
    6. `file_name` is in Long UNC (Uniform Naming Convention) format with an absolute drive and an absolute pathname. For example, consider a `file_name` of `/tmp/A`.h5. If that target file does not exist, the new `file_name` after stripping will be `A.h5`.


The library opens the target file `file_name` with the file access property list that is set via [H5Pset_elink_fapl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga3895e8e60ce8f0b6f32ab7a22c715d1a "Sets a file access property list for use in accessing a file pointed to by an external link.") when the external link link_name is accessed. If no such property list is set, the library uses the file access property list associated with the file of `link_loc_id` to open the target file.
If an application requires additional control over file access flags or the file access property list, see [H5Pset_elink_cb()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2 "Sets the external link traversal callback function in a link access property list."); this function enables the use of an external link callback function as described in [H5L_elink_traverse_t()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76 "Callback for external link traversal."). 

Attention
    A file close degree property setting ([H5Pset_fclose_degree()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c "Sets the file close degree.")) in the external link file access property list or in the external link callback function will be ignored. A file opened by means of traversing an external link is always opened with the weak file close degree property setting, [H5F_CLOSE_WEAK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475faea1127311a219b44e4af3cb12609035f) . 

See also
     [H5Lcreate_hard()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2 "Creates a hard link to an object."), [H5Lcreate_soft()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link."), [H5Lcreate_ud()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gadaf9732947c45cd4d2442e7f58873fc2 "Creates a link of a user-defined type.") 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2)H5Lcreate_hard()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lcreate_hard  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _cur_loc_ ,   
---|---|---|---  
|  | const char * |  _cur_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_loc_ ,   
|  | const char * |  _dst_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Creates a hard link to an object.  

Parameters
     [in] | cur_loc | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | cur_name | Name of the target object, which must already exist   
[in] | dst_loc | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
[in] | dst_name | The name of the new link   
[in] | lcpl_id | Link creation property list identifier   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lcreate_hard()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2 "Creates a hard link to an object.") creates a new hard link to a pre-existing object in an HDF5 file.
`cur_loc` and `cur_name` specify the location and name, respectively, of the target object, i.e., the object that the new hard link points to. `dst_loc` and `dst_name` specify the location and name, respectively, of the new hard link.
`cur_name` and `dst_name` are interpreted relative to `cur_loc` and `dst_loc`, respectively. If a given name begins with `/`, then it will be interpreted as absolute path in the file. The names of the created links will be the last element of each provided path. Prior elements in each path are used to locate the parent groups of each new link. If `cur_loc` and `dst_loc` are the same location, the HDF5 macro [H5L_SAME_LOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a484bbac16dab421b9e32c4d9c2128658) can be used for either parameter (but not both).
`lcpl_id` and `lapl_id` are the link creation and access property lists associated with the new link. 

Note
    Hard and soft links are for use only if the target object is in the current file. If the desired target object is in a different file from the new link, an external link may be created with [H5Lcreate_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.").      The HDF5 library keeps a count of all hard links pointing to an object; if the hard link count reaches zero (0), the object will be deleted from the file. Creating new hard links to an object will prevent it from being deleted if other links are removed. The library maintains no similar count for soft links and they can dangle.      The new link may be one of many that point to that object. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78)H5Lcreate_soft()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lcreate_soft  | ( | const char * |  _link_target_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _link_loc_id_ ,   
|  | const char * |  _link_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Creates a soft link.  

Parameters
     [in] | link_target | An HDF5 path name   
---|---|---  
[in] | link_loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
[in] | link_name | The name of the new link   
[in] | lcpl_id | Link creation property list identifier   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lcreate_soft()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link.") creates a new soft link to an object in an HDF5 file.
`link_target` specifies the HDF5 path name the soft link contains. `link_target` can be an arbitrary HDF5 path name and is interpreted only at lookup time. This path may be absolute in the file or relative to `link_loc_id`.
`link_loc_id` and `link_name` specify the location and name, respectively, of the new soft link. `link_name` is interpreted as a path relative to `link_loc_id`, or an absolute path if it begins with `/`. The name of the created link will be the last element of the provided path. Prior elements in the path are used to locate the parent group of the new link.
If `link_loc_id` is a group identifier, the object pointed to by `link_name` will be accessed as a member of that group. If `link_loc_id` is a file identifier, the object will be accessed as a member of the file's root group.
`lcpl_id` and `lapl_id` are the link creation and access property lists associated with the new link.
For instance, if target_path is `link_loc_id` specifies `new_link`, then a subsequent request for  

Note
     [H5Lcreate_soft()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link.") is for use only if the target object is in the current file. If the desired target object is in a different file from the new link, use [H5Lcreate_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.") to create an external link.      Soft links and external links are also known as symbolic links as they use a name to point to an object; hard links employ an object's address in the file.      Unlike hard links, a soft link in an HDF5 file is allowed to dangle, meaning that the target object need not exist at the time that the link is created.      The HDF5 library does not keep a count of soft links as it does of hard links.      The new link may be one of many that point to that object. 

See also
     [H5Lcreate_hard()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2 "Creates a hard link to an object."), [H5Lcreate_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.") 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gadaf9732947c45cd4d2442e7f58873fc2)H5Lcreate_ud()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lcreate_ud  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _link_loc_id_ ,   
---|---|---|---  
|  | const char * |  _link_name_ ,   
|  | [H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) |  _link_type_ ,   
|  | const void * |  _udata_ ,   
|  | size_t |  _udata_size_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Creates a link of a user-defined type.  

Parameters
     [in] | link_loc_id | Location identifier   
---|---|---  
[in] | link_name | Link name   
[in] | link_type | User-defined link class   
[in] | udata | User-supplied link information   
[in] | udata_size | Size of udata buffer   
[in] | lcpl_id | Link creation property list identifier   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lcreate_ud()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gadaf9732947c45cd4d2442e7f58873fc2 "Creates a link of a user-defined type.") creates a link of user-defined type `link_type` named `link_name` at the location specified in `link_loc_id` with user-specified data `udata`.
`link_name` is interpreted relative to `link_loc_id`. If `link_name` begins with `/`, then it will be interpreted as an absolute path in the file. The name of the created link will be the last element of the provided path. Prior elements in the path are used to locate the parent group of the new link.
Valid values for the link class of the new link, `link_type`, include [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06) and any user-defined link classes that have been registered with the library. See [H5Lregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga5073a814de9878bad53e1d3c900ea77f "Registers a user-defined link class or changes behavior of an existing class.") for further information.
The format of the information pointed to by `udata` is defined by the user. `udata_size` specifies the size of the `udata` buffer. `udata` may be NULL if `udata_size` is zero (0).
The property lists specified by `lcpl_id` and `lapl_id` specify properties used to create and access the link. 

Note
    The external link type, [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06), included in the HDF5 library distribution, is implemented as a user-defined link type. This was done, in part, to provide a model for the implementation of other user-defined links. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf)H5Ldelete()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Ldelete  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Removes a link from a group.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Name of the link to delete   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Ldelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") removes the link specified by `name` from the location `loc_id`.
If the link being removed is a hard link, [H5Ldelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") also decrements the link count for the object to which name points. Unless there is a duplicate hard link in that group, this action removes the object to which name points from the group that previously contained it.
Object headers keep track of how many hard links refer to an object; when the hard link count, also referred to as the reference count, reaches zero, the object can be removed from the file. The file space associated will then be released, i.e., identified in memory as freespace. Objects which are open are not removed until all identifiers to the object are closed. 

Attention
    Exercise caution in the use of [H5Ldelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group."); if the link being removed is on the only path leading to an HDF5 object, that object may become permanently inaccessible in the file. 

See also
     [H5Lcreate_hard()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2 "Creates a hard link to an object."), [H5Lcreate_soft()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link."), [H5Lcreate_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.") 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaaf5f820856afdd34f9070a797a246805)H5Ldelete_by_idx()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Ldelete_by_idx  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Removes the _n_ -th link in a group.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | group_name | Name of subject group   
[in] | idx_type | Index or field which determines the order   
[in] | order | Order within field or index   
[in] | n | Link for which to retrieve information   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Ldelete_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaaf5f820856afdd34f9070a797a246805 "Removes the n-th link in a group.") removes the _n_ -th link in a group according to the specified order, `order`, in the specified index, `index`.
If `loc_id` specifies the group in which the link resides, `group_name` can be a dot ( 

See also
    [H5Ldelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801)H5Lexists()
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5Lexists  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Determines whether a link with the specified name exists in a group.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Link name   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") allows an application to determine whether the link `name` exists in the location specified by `loc_id`. The link may be of any type; only the presence of a link with that name is checked.
Note that [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") verifies only that the target link exists. If name includes either a relative path or an absolute path to the target link, intermediate steps along the path must be verified before the existence of the target link can be safely checked. If the path is not verified, and an intermediate element of the path does not exist, [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") will fail. The example in the next paragraph illustrates one step-by-step method for verifying the existence of a link with a relative or absolute path.
**Example:** Use the following steps to verify the existence of the link `datasetD` in the `group` group1/group2/softlink_to_group3/, where `group1` is a member of the group specified by `loc_id:`
  1. First, use [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") to verify that the `group1` exists.
  2. If `group1` exists, use [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") again, this time with name set to `group1/group2`, to verify that `group2` exists.
  3. If `group2` exists, use [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") with name set to `group1/group2/softlink_to_group3` to verify that `softlink_to_group3` exists.
  4. If `softlink_to_group3` exists, you can now safely use [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") with `name` set to `group1/group2/softlink_to_group3/datasetD` to verify that the target link, `datasetD`, exists.


If the link to be verified is specified with an absolute path, the same approach should be used, but starting with the first link in the file's root group. For instance, if `datasetD` were in `/group1/group2/softlink_to_group3`, the first call to [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") would have name set to `/group1`.
Note that this is an outline and does not include all the necessary details. Depending on circumstances, for example, you may need to verify that an intermediate link points to a group and that a soft link points to an existing target. 

Note
    The behavior of [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") was changed in the 1.10 release in the case where the root group, `"/"`, is the name of the link. This change is described below: 
  1. Let `file` denote a valid HDF5 file identifier, and let `lapl` denote a valid link access property list identifier. A call to [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") with arguments `file`, `"/"`, and `lapl` returns a positive value; in other words, `H5Lexists(file, "/", lapl)` returns a positive value. In the HDF5 1.8 release, this function returns 0. 
  2. Let `root` denote a valid HDF5 group identifier that refers to the root group of an HDF5 file, and let `lapl` denote a valid link access property list identifier. A call to [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") with arguments c root, `"/"`, and `lapl` returns a positive value; in other words, `H5Lexists(root, "/", lapl)` returns a positive value. In the HDF5 1.8 release, this function returns 0. 

Note that the function accepts link names and path names. This is potentially misleading to callers, and we plan to separate the functionality for link names and path names in a future release. 

Attention
     [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.") checks the existence of only the final element in a relative or absolute path; it does not check any other path elements. The function will therefore fail when both of the following conditions exist:
  * `name` is not local to the group specified by `loc_id` or, if `loc_id` is something other than a group identifier, `name` is not local to the root group.
  * Any element of the relative path or absolute path in name, except the target link, does not exist.



Version
    1.10.0 Function behavior changed in this release. (See the note.)  

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gacc2ad7f2b402c4bf9bb122d7f43b98dc)H5Lget_info1()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lget_info1  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  |  [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html) * |  _linfo_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Returns information about a link.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Link name   
[out] | linfo | Buffer in which link information is returned   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000059)**
    As of HDF5-1.12 this function has been deprecated in favor of the function [H5Lget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400 "Returns information about a link.") or the macro [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698).  
[H5Lget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gacc2ad7f2b402c4bf9bb122d7f43b98dc "Returns information about a link.") returns information about the specified link through the `linfo` argument.
The location identifier, `loc_id`, specifies the location of the link. A link name, `name`, interpreted relative to `loc_id`, specifies the link being queried.
`lapl_id` is the link access property list associated with the link `name`. In the general case, when default link access properties are acceptable, this can be passed in as [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f). An example of a situation that requires a non-default link access property list is when the link is an external link; an external link may require that a link prefix be set in a link access property list (see [H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths.")).
[H5Lget_info1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gacc2ad7f2b402c4bf9bb122d7f43b98dc "Returns information about a link.") returns information about name in the data structure [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html), which is described below and defined in [H5Lpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html). This structure is returned in the buffer `linfo`. 
typedef struct {
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) type; 
bool corder_valid; 
int64_t corder; 
[H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) cset; 
union {
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) address; 
size_t val_size; 
} u;
} [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html);
In the above struct, type specifies the link class. Valid values include the following: 
[H5L_TYPE_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48ab90f13082490fcd293a228b2785489e3) | Hard link  
---|---  
[H5L_TYPE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a38eb885df3f43f179b973f576fe996ed) | Soft link  
[H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06) | External link  
[H5L_TYPE_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a4a582d434de3ee2c583384c4d3a3273d) | Error  
There will be additional valid values if user-defined links have been registered.
`corder` specifies the link's creation order position while `corder_valid` indicates whether the value in `corder` is valid.
If `corder_valid` is `true`, the value in `corder` is known to be valid; if `corder_valid` is `false`, the value in `corder` is presumed to be invalid;
`corder` starts at zero (0) and is incremented by one (1) as new links are created. But higher-numbered entries are not adjusted when a lower-numbered link is deleted; the deleted link's creation order position is simply left vacant. In such situations, the value of `corder` for the last link created will be larger than the number of links remaining in the group.
`cset` specifies the character set in which the link name is encoded. Valid values include the following: 
[H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663) | US ASCII  
---|---  
[H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658) | UTF-8 Unicode encoding  
This value is set with [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.").
`address` and `val_size` are returned for hard and symbolic links, respectively. Symbolic links include soft and external links and some user-defined links.
If the link is a hard link, `address` specifies the file address that the link points to.
If the link is a symbolic link, `val_size` will be the length of the link value, e.g., the length of the HDF5 path name with a null terminator. 

Version
    1.12.0 Function was deprecated.       1.8.2 Fortran subroutine added in this release.       1.8.4 Fortran subroutine syntax changed in this release. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400)H5Lget_info2()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lget_info2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  |  [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) * |  _linfo_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Returns information about a link.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Link name   
[out] | linfo | Buffer in which link information is returned   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400 "Returns information about a link.") returns information about the specified link through the `linfo` argument.
The location identifier, `loc_id`, specifies the location of the link. A link name, `name`, interpreted relative to `loc_id`, specifies the link being queried.
`lapl_id` is the link access property list associated with the link name. In the general case, when default link access properties are acceptable, this can be passed in as [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f). An example of a situation that requires a non-default link access property list is when the link is an external link; an external link may require that a link prefix be set in a link access property list (see [H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths.")).
[H5Lget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400 "Returns information about a link.") returns information about name in the data structure [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html "Information struct for links."), which is described below and defined in [H5Lpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html). This structure is returned in the buffer `linfo`. 
typedef struct {
[H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) type; 
bool corder_valid; 
int64_t corder; 
[H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) cset; 
union {
[H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) token; 
size_t val_size; 
} u;
} [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html);
In the above struct, `type` specifies the link class. Valid values include the following: 
[H5L_TYPE_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48ab90f13082490fcd293a228b2785489e3) | Hard link  
---|---  
[H5L_TYPE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a38eb885df3f43f179b973f576fe996ed) | Soft link  
[H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06) | External link  
[H5L_TYPE_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a4a582d434de3ee2c583384c4d3a3273d) | Error  
There will be additional valid values if user-defined links have been registered.
`corder` specifies the link's creation order position, while `corder_valid` indicates whether the value in corder is valid.
If `corder_valid` is `true`, the value in `corder` is known to be valid; if `corder_valid` is `false`, the value in `corder` is presumed to be invalid; `corder` starts at zero (0) and is incremented by one (1) as new links are created. But higher-numbered entries are not adjusted when a lower-numbered link is deleted; the deleted link's creation order position is simply left vacant. In such situations, the value of `corder` for the last link created will be larger than the number of links remaining in the group.
`cset` specifies the character set in which the link name is encoded. Valid values include the following: 
[H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663) | US ASCII  
---|---  
[H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658) | UTF-8 Unicode encoding  
This value is set with [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.").
`token` is the location that a hard link points to, and `val_size` is the size of a soft link or user-defined link value. [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) is used in the VOL layer. It is defined in [H5public.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html) as: 
typedef struct [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) {
uint8_t [__data](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html#adaac031833a234d10c1ff3130f6aa4cc)[[H5O_MAX_TOKEN_SIZE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#ac91e46b83ee173747f9792b33755ff0e)];
} [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html);
If the link is a symbolic link, `val_size` will be the length of the link value, e.g., the length of the HDF5 path name with a null terminator. 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga7ed207f47e0e0f768f0d540c73e37e2a)H5Lget_info_by_idx1()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lget_info_by_idx1  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  |  [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html) * |  _linfo_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Retrieves metadata for a link in a group, according to the order within a field or index.  

Parameters
     [in] | loc_id | Location identifier   
---|---|---  
[in] | group_name | Group name   
[in] | idx_type | Index type   
[in] | order | Iteration order   
[in] | n | Link position for which to retrieve information   
[out] | linfo | Buffer in which link information is returned   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000060)**
    As of HDF5-1.12 this function has been deprecated in favor of the function [H5Lget_info_by_idx2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9 "Retrieves metadata for a link in a group, according to the order within a field or index.") and the macro [H5Lget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67).  
[H5Lget_info_by_idx1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga7ed207f47e0e0f768f0d540c73e37e2a "Retrieves metadata for a link in a group, according to the order within a field or index.") returns the metadata for a link in a group according to a specified field or index and a specified order.
The link for which information is to be returned is specified by `idx_type`, `order`, and `n` as follows:
  * `idx_type` specifies the field by which the links in `group_name` are ordered. The links may be indexed on this field, in which case operations seeking specific links are likely to complete more quickly.
  * `order` specifies the order in which the links are to be referenced for the purposes of this function.
  * `n` specifies the position of the subject link. Note that this count is zero-based; 0 (zero) indicates that the function will return the value of the first link; if `n` is 5, the function will return the value of the sixth link; etc.


For example, assume that `idx_type`, `order`, and `n` are [H5_INDEX_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3a644e6701706be4d37660864336c7bd3e), [H5_ITER_DEC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a2f123d7a4d68054e8aa2d0f1d0a3fcd2), and 5, respectively. [H5_INDEX_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3a644e6701706be4d37660864336c7bd3e) indicates that the links are accessed in lexicographic order by their names. [H5_ITER_DEC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a2f123d7a4d68054e8aa2d0f1d0a3fcd2) specifies that the list be traversed in reverse order, or in decremented order. And 5 specifies that this call to the function will return the metadata for the 6th link (`n` + 1) from the end.
See [H5Literate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1e7c0a8cf17699563c02e128f27042f1 "Iterates over links in a group, with user callback routine, according to the order within an index.") for a list of valid values and further discussion regarding `idx_type` and `order`.
If `loc_id` specifies the group in which the link resides, `group_name` can be a dot ( 

Version
    1.12.0 Function was renamed to H5Lget_index_by_idx1() and deprecated.       1.8.4 Fortran subroutine syntax changed in this release.       1.8.2 Fortran subroutine added in this release. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9)H5Lget_info_by_idx2()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lget_info_by_idx2  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  |  [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) * |  _linfo_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Retrieves metadata for a link in a group, according to the order within a field or index.  

Parameters
     [in] | loc_id | Location identifier   
---|---|---  
[in] | group_name | Group name   
[in] | idx_type | Index type   
[in] | order | Iteration order   
[in] | n | Link position for which to retrieve information   
[out] | linfo | Buffer in which link information is returned   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lget_info_by_idx2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaecfb3ef8520e9224b24a151ff8459ba9 "Retrieves metadata for a link in a group, according to the order within a field or index.") returns the metadata for a link in a group according to a specified field or index and a specified order. The link for which information is to be returned is specified by `idx_type`, `order`, and `n` as follows:
  * `idx_type` specifies the field by which the links in `group_name` are ordered. The links may be indexed on this field, in which case operations seeking specific links are likely to complete more quickly.
  * `order` specifies the order in which the links are to be referenced for the purposes of this function.
  * `n` specifies the position of the subject link. Note that this count is zero-based; 0 (zero) indicates that the function will return the value of the first link; if `n` is 5, the function will return the value of the sixth link; etc.


For example, assume that `idx_type`, `order`, and `n` are [H5_INDEX_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3a644e6701706be4d37660864336c7bd3e), [H5_ITER_DEC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a2f123d7a4d68054e8aa2d0f1d0a3fcd2), and 5, respectively. [H5_INDEX_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3a644e6701706be4d37660864336c7bd3e) indicates that the links are accessed in lexicographic order by their names. [H5_ITER_DEC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a2f123d7a4d68054e8aa2d0f1d0a3fcd2) specifies that the list be traversed in reverse order, or in decremented order. And 5 specifies that this call to the function will return the metadata for the 6th link (`n` + 1) from the end.
See [H5Literate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73 "Iterates over links in a group, with user callback routine, according to the order within an index.") for a list of valid values and further discussion regarding `idx_type` and `order`.
If `loc_id` specifies the group in which the link resides, `group_name` can be a dot ( 

Since
    1.12.0 

See also
     [H5Lget_info2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga65e63c6e880fd0183c40486d6748e400 "Returns information about a link.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90)H5Lget_name_by_idx()
ssize_t H5Lget_name_by_idx  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  | char * |  _name_ ,   
|  | size_t |  _size_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Retrieves name of the _n_ -th link in a group, according to the order within a specified field or index.  

Parameters
     [in] | loc_id | Location identifier   
---|---|---  
[in] | group_name | Group name   
[in] | idx_type | Index type   
[in] | order | Iteration order   
[in] | n | Link position for which to retrieve information   
[out] | name | Buffer in which link name is returned   
[in] | size | Size in bytes of `name`  
[in] | lapl_id | Link access property list identifier 

Returns
    Returns the size of the link name if successful; otherwise returns a negative value.  
[H5Lget_name_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90 "Retrieves name of the n-th link in a group, according to the order within a specified field or index.") retrieves the name of the _n_ -th link in a group, according to the specified order, `order`, within a specified field or index, `idx_type`.
`idx_type` specifies the index that is used. Valid values include the following: 
[H5_INDEX_NAME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3a644e6701706be4d37660864336c7bd3e) | Lexicographic order on name  
---|---  
[H5_INDEX_CRT_ORDER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3a315475479b34056043b4b6583bafb280) | Index on creation order  
`order` specifies the order in which objects are inspected along the index specified in `idx_type`. Valid values include the following: 
[H5_ITER_INC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a80c3e083c0a77063b1a66553decfcb08) | Increasing order  
---|---  
[H5_ITER_DEC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a2f123d7a4d68054e8aa2d0f1d0a3fcd2) | Decreasing order  
[H5_ITER_NATIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a15eebaf7d85d5e37e918858634e29d01) | Fastest available order  
If `loc_id` specifies the group in which the link resides, `group_name` can be a dot (
Up to `size` characters of the link name are returned in `name`; additional characters, if any, are not returned to the user application.  
  
If the length of the link name, which determines the required value of `size`, is unknown, a preliminary call to [H5Lget_name_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90 "Retrieves name of the n-th link in a group, according to the order within a specified field or index.") with the last two parameters set to NULL and zero respectively can be made. The return value of this call will be the size in bytes of the link name. That value, plus 1 for a NULL terminator, must then be assigned to `size` for a second [H5Lget_name_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90 "Retrieves name of the n-th link in a group, according to the order within a specified field or index.") call, which will retrieve the actual link name. 

Note
    Please note that in order for the specified index to correspond to the creation order index, `order` must be set to [H5_ITER_INC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a80c3e083c0a77063b1a66553decfcb08) or [H5_ITER_DEC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a2f123d7a4d68054e8aa2d0f1d0a3fcd2) when calling [H5Lget_name_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90 "Retrieves name of the n-th link in a group, according to the order within a specified field or index.").       The index `n` passed to [H5Lget_name_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90 "Retrieves name of the n-th link in a group, according to the order within a specified field or index.") is the index of the link within the link table, sorted according to `order` and `idx_type`. If order is [H5_ITER_NATIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a15eebaf7d85d5e37e918858634e29d01), then the link table is not sorted, and it does not matter what `idx_type` is. Specifying [H5_ITER_NATIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9a15eebaf7d85d5e37e918858634e29d01) does not guarantee any particular order, only that it remains consistent. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4)H5Lget_val()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lget_val  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | void * |  _buf_ ,   
|  | size_t |  _size_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Returns the value of a link.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | name | Link name   
[out] | buf | The buffer to hold the link value   
[in] | size | Maximum number of bytes of link value to be returned   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lget_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.") returns the value of link `name`. For symbolic links, this is the path to which the link points, including the null terminator. For external and user-defined links, it is the link buffer.
`size` is the size of `buf` and should be the size of the link value being returned. This size value can be determined through a call to [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698); it is returned in the `val_size` field of the [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) `struct`.
If `size` is smaller than the size of the returned value, then the string stored in `buf` will be truncated to `size` bytes. For soft links, this means that the value will not be null terminated.
In the case of external links, the target file and object names are extracted from `buf` by calling [H5Lunpack_elink_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220 "Decodes external link information.").
The link class of link `name` can be determined with a call to [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698).
`lapl_id` specifies the link access property list associated with the link `name`. In the general case, when default link access properties are acceptable, this can be passed in as [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f). An example of a situation that requires a non-default link access property list is when the link is an external link; an external link may require that a link prefix be set in a link access property list (see [H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths.")).
This function should be used only after [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) has been called to verify that `name` is a symbolic link. This can be determined from the `link_type` field of the [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) `struct`. 

Note
    This function will fail if called on a hard link. 

See also
    [H5Lget_val_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaf7be56de947e09a8d084e9d13a90bf3c "Retrieves value of the n-th link in a group, according to the order within an index.") 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaf7be56de947e09a8d084e9d13a90bf3c)H5Lget_val_by_idx()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lget_val_by_idx  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  | void * |  _buf_ ,   
|  | size_t |  _size_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Retrieves value of the _n_ -th link in a group, according to the order within an index.  

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | group_name | Group name   
[in] | idx_type | Type of index   
[in] | order | Order within field or index   
[in] | n | Link position for which to retrieve information   
[out] | buf | The buffer to hold the link value   
[in] | size | Maximum number of bytes of link value to be returned   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lget_val_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaf7be56de947e09a8d084e9d13a90bf3c "Retrieves value of the n-th link in a group, according to the order within an index.") retrieves the value of the _n_ -th link in a group, according to the specified order, `order`, within an index, `index`.
For soft links, the value is an HDF5 path name.
For external links, this is a compound value containing file and path name information; to use this external link information, it must first be decoded with [H5Lunpack_elink_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220 "Decodes external link information.")
For user-defined links, this value will be described in the definition of the user-defined link type.
`loc_id` specifies the location identifier of the group specified by `group_name`.
`group_name` specifies the group in which the link exists. If `loc_id` already specifies the group in which the link exists, `group_name` must be a dot (
The size in bytes of link_val is specified in `size`. The size value can be determined through a call to [H5Lget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67); it is returned in the `val_size` field of the [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) `struct`. If size is smaller than the size of the returned value, then the string stored in link_val will be truncated to size bytes. For soft links, this means that the value will not be null terminated.
If the type of the link is unknown or uncertain, [H5Lget_val_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaf7be56de947e09a8d084e9d13a90bf3c "Retrieves value of the n-th link in a group, according to the order within an index.") should be called only after the type has been determined via a call to [H5Lget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67). 

Note
    This function will fail if called on a hard link. 

See also
    [H5Lget_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.") 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb)H5Lmove()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lmove  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_loc_ ,   
---|---|---|---  
|  | const char * |  _src_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_loc_ ,   
|  | const char * |  _dst_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ )  
Moves a link within an HDF5 file.  

Parameters
     [in] | src_loc | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
---|---|---  
[in] | src_name | Original link name   
[in] | dst_loc | Location identifier. The identifier may be that of a file, group, dataset, named datatype, or attribute.   
[in] | dst_name | New link name   
[in] | lcpl_id | Link creation property list identifier   
[in] | lapl_id | Link access property list identifier 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file.") moves a link within an HDF5 file. The original link, `src_name`, is removed from `src_loc` and the new link, `dst_name`, is inserted at dst_loc. This change is accomplished as an atomic operation.
`src_loc` and `src_name` identify the original link. `src_loc` is the original location identifier; `src_name` is the path to the link and is interpreted relative to `src_loc`.
`dst_loc` and `dst_name` identify the new link. `dst_loc` is either a file or group identifier; `dst_name` is the path to the link and is interpreted relative to `dst_loc`.
`lcpl_id` and `lapl_id` are the link creation and link access property lists, respectively, associated with the new link, `dst_name`.
Through these property lists, several properties are available to govern the behavior of [H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file."). The property controlling creation of missing intermediate groups is set in the link creation property list with [H5Pset_create_intermediate_group()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c "Specifies in property list whether to create missing intermediate groups."); [H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file.") ignores any other properties in the link creation property list. Properties controlling character encoding, link traversals, and external link prefixes are set in the link access property list with [H5Pset_char_encoding()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names."), [H5Pset_nlinks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaa46c63c196a0cf5cd94dede039c030f4 "Sets maximum number of soft or user-defined link traversals."), and [H5Pset_elink_prefix()](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gafa5eced13ba3a00cdd65669626dc7294 "Sets prefix to be applied to external link paths."), respectively. 

Note
    Note that [H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file.") does not modify the value of the link; the new link points to the same object as the original link pointed to. Furthermore, if the object pointed to by the original link was already open with a valid object identifier, that identifier will remain valid after the call to [H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file."). 

Attention
    Exercise care in moving links as it is possible to render data in a file inaccessible with [H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file."). If the link being moved is on the only path leading to an HDF5 object, that object may become permanently inaccessible in the file. 

Since
    1.8.0
* * *
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220)H5Lunpack_elink_val()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lunpack_elink_val  | ( | const void * |  _ext_linkval_ ,   
---|---|---|---  
|  | size_t |  _link_size_ ,   
|  | unsigned * |  _flags_ ,   
|  | const char ** |  _filename_ ,   
|  | const char ** |  _obj_path_ )  
Decodes external link information.  

Parameters
     [in] | ext_linkval | Buffer containing external link information   
---|---|---  
[in] | link_size | Size, in bytes, of the `ext_linkval` buffer   
[out] | flags | External link flags, packed as a bitmap (_Reserved as a bitmap for flags; no flags are currently defined, so the only valid value * is 0._)   
[out] | filename | Returned filename  
[out] | obj_path | Returned object path, relative to `filename` 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5Lunpack_elink_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220 "Decodes external link information.") decodes the external link information returned by [H5Lget_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.") in the `ext_linkval` buffer.
`ext_linkval` should be the buffer set by [H5Lget_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.") and will consist of two NULL-terminated strings, the filename and object path, one after the other.
Given this buffer, [H5Lunpack_elink_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220 "Decodes external link information.") creates pointers to the filename and object path within the buffer and returns them in `filename` and `obj_path`, unless they are passed in as NULL.
[H5Lunpack_elink_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220 "Decodes external link information.") requires that `ext_linkval` contain a concatenated pair of null-terminated strings, so use of this function on a string that is not an external link `udata` buffer may result in a segmentation fault. This failure can be avoided by adhering to the following procedure: 
  1. Call [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) to get the link type and the size of the link value.
  2. Verify that the link is an external link, i.e., that its link type is [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06). 
  3. Call [H5Lget_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.") to get the link value. 
  4. Call [H5Lunpack_elink_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220 "Decodes external link information.") to unpack that value. 



Since
    1.8.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


