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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title0 "H2")
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title1 "H2")
  * [ Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title2 "H2")
  * [ Enumerations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title3 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title4 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title5 "H2")
  * [◆ H5L_MAX_LINK_NAME_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title6 "H2")
  * [◆ H5L_SAME_LOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title7 "H2")
  * [◆ H5L_TYPE_BUILTIN_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title8 "H2")
  * [◆ H5L_TYPE_UD_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title9 "H2")
  * [◆ H5L_TYPE_UD_MIN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title10 "H2")
  * [Typedef Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title11 "H2")
  * [◆ H5L_elink_traverse_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title12 "H2")
  * [◆ H5L_iterate1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title13 "H2")
  * [◆ H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title14 "H2")
  * [Enumeration Type Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title15 "H2")
  * [◆ H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#title16 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#nested-classes) | [Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#define-members) | [Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#typedef-members) | [Enumerations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#enum-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#func-members)
H5Lpublic.h File Reference
`#include "H5public.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html)"`  
`#include "H5Ipublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html)"`  
`#include "H5Opublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html)"`  
`#include "H5Tpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html)"`  

##  Data Structures  
---  
struct  | [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html)  
struct  | [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html)  
| Information struct for links. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html#details)  
  
##  Macros  
---  
#define  |  [H5L_MAX_LINK_NAME_LEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a81e20e71d0708535b22f679128bcdda2) UINT32_MAX  
#define  |  [H5L_SAME_LOC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a484bbac16dab421b9e32c4d9c2128658) 0 /* ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)) */  
#define  |  [H5L_TYPE_BUILTIN_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#adc65173efbfc9b6029391d6de808b2e6) [H5L_TYPE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a38eb885df3f43f179b973f576fe996ed)  
#define  |  [H5L_TYPE_UD_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a6f9109ae61b6bd1b6668237cce13c75f) [H5L_TYPE_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a810ce4b5dddcd41521557af0273dd5cd)  
#define  |  [H5L_TYPE_UD_MIN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#afd6fe1db1c142b14a7f879b8d277aa3f) [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06)  
##  Typedefs  
---  
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(*  |  [H5L_elink_traverse_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76)) (const char *parent_file_name, const char *parent_group_name, const char *child_file_name, const char *child_object_name, unsigned *acc_flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, void *op_data)  
| Callback for external link traversal.   
  
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(*  |  [H5L_iterate1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#aa92ad6ac7f9720520690785ad53d8b08)) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group, const char *name, const [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html) *info, void *op_data)  
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(*  |  [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group, const char *name, const [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) *info, void *op_data)  
| Prototype for [H5Literate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73 "Iterates over links in a group, with user callback routine, according to the order within an index."), [H5Literate_by_name2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga745a65eb516ce40a3be43490aaeb5c5e "Iterates through links in a group.") operator.   
  
##  Enumerations  
---  
enum  |  [H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) {   
[H5L_TYPE_ERROR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a4a582d434de3ee2c583384c4d3a3273d) = (-1) , [H5L_TYPE_HARD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48ab90f13082490fcd293a228b2785489e3) = 0 , [H5L_TYPE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a38eb885df3f43f179b973f576fe996ed) = 1 , [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06) = 64 ,   
[H5L_TYPE_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a810ce4b5dddcd41521557af0273dd5cd) = 255   
}  
| Link class types. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48)  
  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gafd4624f1c040d5f1df36cb1e6986aac6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_loc, const char *src_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_loc, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates an identical copy of a link with the same creation time and target. The new link can have a different name and be in a different location than the original.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_external](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683) (const char *file_name, const char *obj_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) link_loc_id, const char *link_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates an external link, a soft link to an object in a different file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_hard](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cur_loc, const char *cur_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_loc, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates a hard link to an object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_hard_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaecc69b84cafb71d27dbcc244c35930c7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cur_loc_id, const char *cur_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_loc_id, const char *new_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78) (const char *link_target, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) link_loc_id, const char *link_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates a soft link.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_soft_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7ba84c7db6ef67e270fa7bc3413d4def) (const char *link_target, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) link_loc_id, const char *link_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_ud](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gadaf9732947c45cd4d2442e7f58873fc2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) link_loc_id, const char *link_name, [H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) link_type, const void *udata, size_t udata_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates a link of a user-defined type.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Removes a link from a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ldelete_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga6074beb8fd1d7761db082fc611519b54) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ldelete_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaaf5f820856afdd34f9070a797a246805) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Removes the _n_ -th link in a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ldelete_by_idx_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7dc3198d8cfaeb6a780863af8d0af253) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Lexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Determines whether a link with the specified name exists in a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lexists_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga4fddab130acbcef4dd8c7ccf75a31b2c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, bool *exists, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
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
  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Lis_registered](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_a.html#ga02c1cf38efea97d18e5e2f65df3f08a2) ([H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48) id)  
| Determines whether a class of user-defined links is registered.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Literate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1e7c0a8cf17699563c02e128f27042f1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) grp_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [H5L_iterate1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#aa92ad6ac7f9720520690785ad53d8b08) op, void *op_data)  
| Iterates over links in a group, with user callback routine, according to the order within an index.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Literate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) grp_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op, void *op_data)  
| Iterates over links in a group, with user callback routine, according to the order within an index.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Literate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga0f56b7243d036cdeb5280d8c41f2436e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx_p, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Literate_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga87e036da0c8d1146a073f3ee08e0fedc) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [H5L_iterate1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#aa92ad6ac7f9720520690785ad53d8b08) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Iterates through links in a group by its name.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Literate_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga745a65eb516ce40a3be43490aaeb5c5e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Iterates through links in a group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_loc, const char *src_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_loc, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Moves a link within an HDF5 file.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lunpack_elink_val](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gade0c3b274c185d148f000172fbdc3220) (const void *ext_linkval, size_t link_size, unsigned *flags, const char **filename, const char **obj_path)  
| Decodes external link information.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lvisit1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga5424ef7043c82147490d027a0e8a59ef) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) grp_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5L_iterate1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#aa92ad6ac7f9720520690785ad53d8b08) op, void *op_data)  
| Recursively visits all links starting from a specified group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lvisit2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gae1c6f963892a5f4e8922a66fbe338f66) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) grp_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op, void *op_data)  
| Recursively visits all links starting from a specified group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lvisit_by_name1](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1f1ba1bb4d44f2c111990024809417ac) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5L_iterate1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#aa92ad6ac7f9720520690785ad53d8b08) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Recursively visits all links starting from a specified group.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lvisit_by_name2](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gafee93792c7e27a7e78b1ec221876b173) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Recursively visits all links starting from a specified group.   
  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a81e20e71d0708535b22f679128bcdda2)H5L_MAX_LINK_NAME_LEN
#define H5L_MAX_LINK_NAME_LEN UINT32_MAX  
---  
Maximum length of a link's name  

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a484bbac16dab421b9e32c4d9c2128658)H5L_SAME_LOC
#define H5L_SAME_LOC 0 /* ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)) */  
---  
Macro to indicate operation occurs on same location  

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#adc65173efbfc9b6029391d6de808b2e6)H5L_TYPE_BUILTIN_MAX
#define H5L_TYPE_BUILTIN_MAX [H5L_TYPE_SOFT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a38eb885df3f43f179b973f576fe996ed)  
---  
Maximum value link value for "built-in" link types  

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a6f9109ae61b6bd1b6668237cce13c75f)H5L_TYPE_UD_MAX
#define H5L_TYPE_UD_MAX [H5L_TYPE_MAX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a810ce4b5dddcd41521557af0273dd5cd)  
---  
Maximum link id value for "user-defined" link types.  

Since
    1.12.1 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#afd6fe1db1c142b14a7f879b8d277aa3f)H5L_TYPE_UD_MIN
#define H5L_TYPE_UD_MIN [H5L_TYPE_EXTERNAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48a1b8b636539eab03587c22735ec84ea06)  
---  
Link ids at or above this value are "user-defined" link types.  

Since
    1.8.0 
## Typedef Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a96345937d04b66a13baec1ef1a6cff76)H5L_elink_traverse_t
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(* H5L_elink_traverse_t) (const char *parent_file_name, const char *parent_group_name, const char *child_file_name, const char *child_object_name, unsigned *acc_flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, void *op_data)  
---  
Callback for external link traversal. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#aa92ad6ac7f9720520690785ad53d8b08)H5L_iterate1_t
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(* H5L_iterate1_t) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group, const char *name, const [H5L_info1_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info1__t.html) *info, void *op_data)  
---  
Prototype for [H5Literate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga1e7c0a8cf17699563c02e128f27042f1 "Iterates over links in a group, with user callback routine, according to the order within an index.") / [H5Literate_by_name1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga87e036da0c8d1146a073f3ee08e0fedc "Iterates through links in a group by its name.") operator 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9)H5L_iterate2_t
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(* H5L_iterate2_t) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group, const char *name, const [H5L_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_l__info2__t.html) *info, void *op_data)  
---  
Prototype for [H5Literate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gad7ca4206f06b5ada85b6ec5867ec6c73 "Iterates over links in a group, with user callback routine, according to the order within an index."), [H5Literate_by_name2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga745a65eb516ce40a3be43490aaeb5c5e "Iterates through links in a group.") operator. 
The [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html) version is used in the VOL layer and future public API calls. 
## Enumeration Type Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48)H5L_type_t
enum [H5L_type_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a1e3c5d37c60e7a59b0179e26a4094a48)  
---  
Link class types. 
Values less than 64 are reserved for the HDF5 library's internal use. Values 64 to 255 are for "user-defined" link class types; these types are defined by HDF5 but their behavior can be overridden by users. Users who want to create new classes of links should contact the HDF5 development team at [help@.nosp@m.hdfg.nosp@m.roup..nosp@m.org](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html). These values can never change because they appear in HDF5 files. 
Enumerator  
---  
H5L_TYPE_ERROR  |  Invalid link type id   
H5L_TYPE_HARD  |  Hard link id   
H5L_TYPE_SOFT  |  Soft link id   
H5L_TYPE_EXTERNAL  |  External link id   
H5L_TYPE_MAX  |  Maximum link type id   
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5Lpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


