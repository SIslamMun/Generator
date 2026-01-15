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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#title0 "H2")
  * [ Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#title2 "H2")
  * [Typedef Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#title3 "H2")
  * [◆ H5A_operator1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#title4 "H2")
  * [◆ H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#nested-classes) | [Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#typedef-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#func-members)
H5Apublic.h File Reference
`#include "H5public.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html)"`  
`#include "H5Ipublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html)"`  
`#include "H5Opublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_opublic_8h.html)"`  
`#include "H5Tpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html)"`  

##  Data Structures  
---  
struct  | [H5A_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a__info__t.html)  
##  Typedefs  
---  
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(*  |  [H5A_operator1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#ae42c937252ed79a1ad4672f04adba750)) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) location_id, const char *attr_name, void *operator_data)  
| Typedef for [H5Aiterate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabdb2cf7368eec0ad998cbe6a3f61aa41 "Calls a user's function for each attribute on an object.") callbacks.   
  
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(*  |  [H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74)) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) location_id, const char *attr_name, const [H5A_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a__info__t.html) *ainfo, void *op_data)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id)  
| Closes the specified attribute.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aclose_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga121fb36f6571bfaf17eb0a92f3275560) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Acreate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaa30f5f6c277d6c46f8aa31e89cdba085) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id)  
| Creates an attribute attached to a specified object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Acreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id)  
| Creates an attribute attached to a specified object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Acreate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga9b3ce1af9b445a6381479e8f3b58e6d9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Acreate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga004160c28e281455ec48aa7fe557ef8a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Creates an attribute attached to a specified object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Acreate_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga13dd8779381c7834b59a4ac5521c83ba) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Adelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gada9fa3d6db52329f1fd55662de6ff6ba) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *attr_name)  
| Deletes an attribute from a specified location.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Adelete_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga06711a4e77ff8ab49e427010fd38ac9e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Deletes an attribute from an object according to index order.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Adelete_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gacbf689308f851428dd641b64f5f94feb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Removes an attribute from a specified location.   
  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Aexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga293b5be270d90cd5e47f782ca9aec80b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const char *attr_name)  
| Determines whether an attribute with a given name exists on an object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aexists_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga126bf57a571c018286e67dbc14873925) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const char *attr_name, bool *exists, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5Aexists_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaa1d2305651a4524f6aa0f8b56eec1a37) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Determines whether an attribute with a given name exists on an object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aexists_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga2277f2644e0c0fe5ea03b523dfba0b1d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, bool *exists, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0f6b545850bd21f128904eff51df226d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id)  
| Gets an attribute creation property list identifier.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gae3f1b7b87240b461f7827a8783acc08a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [H5A_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a__info__t.html) *ainfo)  
| Retrieves attribute information by attribute identifier.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gad110910cb227c15fdca938a642714fe9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [H5A_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a__info__t.html) *ainfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Retrieves attribute information by attribute index position.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga258f03e12b4f49ad33ba72d17a9e2faf) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [H5A_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a__info__t.html) *ainfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Retrieves attribute information by attribute name.   
  
ssize_t  |  [H5Aget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga05e195aabab8c623b1c52009aeb99674) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, size_t buf_size, char *buf)  
| Gets an attribute name.   
  
ssize_t  |  [H5Aget_name_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4c552b2db32371f8ea20d87475313fb6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, char *name, size_t size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Gets an attribute name by attribute index position.   
  
int  |  [H5Aget_num_attrs](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaadd809fc16238754105bbddd20bcdde1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id)  
| Determines the number of attributes attached to an object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9e21e544119d03f9342530b45a71d74d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id)  
| Gets a copy of the dataspace for an attribute.   
  
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  [H5Aget_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabd11c8e11db0adde706e41a24a832f06) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id)  
| Returns the amount of storage used to store an attribute.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id)  
| Gets an attribute's datatype.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aiterate1](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabdb2cf7368eec0ad998cbe6a3f61aa41) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, unsigned *idx, [H5A_operator1_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#ae42c937252ed79a1ad4672f04adba750) op, void *op_data)  
| Calls a user's function for each attribute on an object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aiterate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9315a22b60468b6e996559b1b8a77251) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74) op, void *op_data)  
| Calls a user-defined function for each attribute on an object.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aiterate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga75db973d69b61f673f5cdf21ac624cef) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx, [H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Calls user-defined function for each attribute on an object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga59863b205b6d93b2145f0fbca49656f7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id)  
| Opens an attribute for an object specified by object identifier and attribute name.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga8167b57603377742ae78a278dda27634) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Opens the nth attribute attached to an object.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_by_idx_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga1bef4a2dce9cfc0ddaa7472ac1e2d1dd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
| Opens an attribute for an object by object name and attribute name.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga42e2c5c72201f05b32e1c9dda6df0e30) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadaa85276f2731ad78462a6fd27118470) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, unsigned idx)  
| Opens the attribute specified by its index.   
  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga5c05fade96b6b7e2299f56a5b1edb1c1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name)  
| Opens an attribute specified by name.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, void *buf)  
| Reads the value of an attribute.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aread_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7a74c8bd935164e2d19d4b253cde815a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype_id, void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Arename](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga490dcd6db246c1fda7295badfce28203) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *old_name, const char *new_name)  
| Renames an attribute.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Arename_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga5eb1ec49226fd0ec8e6dedc608f134f8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *old_name, const char *new_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Arename_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga21f8483c935d72187b98f5e7c2056140) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *old_attr_name, const char *new_attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Arename_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga27e7336cbf2182a85d8fba3cdf476d8e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *old_attr_name, const char *new_attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, const void *buf)  
| Writes data to an attribute.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Awrite_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gae841dec1e2f4fecd88252307d20c1a59) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, const void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
## Typedef Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#ae42c937252ed79a1ad4672f04adba750)H5A_operator1_t
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(* H5A_operator1_t) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) location_id, const char *attr_name, void *operator_data)  
---  
Typedef for [H5Aiterate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabdb2cf7368eec0ad998cbe6a3f61aa41 "Calls a user's function for each attribute on an object.") callbacks.  

Parameters
     [in] | location_id | The identifier for the group, dataset or named datatype being iterated over   
---|---|---  
[in] | attr_name | The name of the current object attribute   
[in,out] | operator_data | A pointer to the operator data passed into [H5Aiterate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gabdb2cf7368eec0ad998cbe6a3f61aa41 "Calls a user's function for each attribute on an object.") 

Returns
    The return values from an operator are:   
  * Zero causes the iterator to continue, returning zero when all attributes have been processed. 
  * Positive causes the iterator to immediately return that positive value, indicating short-circuit success. The iterator can be restarted at the next attribute. 
  * Negative causes the iterator to immediately return that value, indicating failure. The iterator can be restarted at the next attribute. 


##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74)H5A_operator2_t
typedef [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)(* H5A_operator2_t) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) location_id, const char *attr_name, const [H5A_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a__info__t.html) *ainfo, void *op_data)  
---  
Typedef for [H5Aiterate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9315a22b60468b6e996559b1b8a77251 "Calls a user-defined function for each attribute on an object.") / [H5Aiterate_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga75db973d69b61f673f5cdf21ac624cef "Calls user-defined function for each attribute on an object.") callbacks  

Parameters
     [in] | location_id | The identifier for the group, dataset or named datatype being iterated over   
---|---|---  
[in] | attr_name | The name of the current object attribute   
[in] | ainfo | The attribute's info struct   
[in,out] | op_data | A pointer to the operator data passed into [H5Aiterate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9315a22b60468b6e996559b1b8a77251 "Calls a user-defined function for each attribute on an object.") or [H5Aiterate_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga75db973d69b61f673f5cdf21ac624cef "Calls user-defined function for each attribute on an object.") 

Returns
    The return values from an operator are:   
  * Zero causes the iterator to continue, returning zero when all attributes have been processed. 
  * Positive causes the iterator to immediately return that positive value, indicating short-circuit success. The iterator can be restarted at the next attribute. 
  * Negative causes the iterator to immediately return that value, indicating failure. The iterator can be restarted at the next attribute.



Attention
     **Leaving callback functions:**  
The callback function must return normally, even in the case of error. Returning with H5_ITER_ERROR, instead of leaving by means of exceptions, exit() function, etc... will allow the HDF5 library to manage its resources and maintain a consistent state. See [C++ Developers using HDF5 C-API functions](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html#cpp_c_api_note) warning for detail. 

Since
    1.8.0 
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5Apublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


