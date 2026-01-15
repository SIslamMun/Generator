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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title2 "H2")
  * [◆ H5Aclose_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title3 "H2")
  * [◆ H5Acreate_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title4 "H2")
  * [◆ H5Acreate_by_name_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title5 "H2")
  * [◆ H5Aexists_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title6 "H2")
  * [◆ H5Aexists_by_name_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title7 "H2")
  * [◆ H5Aopen_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title8 "H2")
  * [◆ H5Aopen_by_idx_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title9 "H2")
  * [◆ H5Aopen_by_name_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title10 "H2")
  * [◆ H5Aread_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title11 "H2")
  * [◆ H5Arename_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title12 "H2")
  * [◆ H5Arename_by_name_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title13 "H2")
  * [◆ H5Awrite_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title14 "H2")
  * [◆ H5Dclose_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title15 "H2")
  * [◆ H5Dcreate_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title16 "H2")
  * [◆ H5Dget_space_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title17 "H2")
  * [◆ H5Dopen_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title18 "H2")
  * [◆ H5Dread_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title19 "H2")
  * [◆ H5Dread_multi_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title20 "H2")
  * [◆ H5Dset_extent_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title21 "H2")
  * [◆ H5Dwrite_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title22 "H2")
  * [◆ H5Dwrite_multi_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title23 "H2")
  * [◆ H5Fclose_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title24 "H2")
  * [◆ H5Fcreate_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title25 "H2")
  * [◆ H5Fflush_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title26 "H2")
  * [◆ H5Fopen_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title27 "H2")
  * [◆ H5Freopen_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title28 "H2")
  * [◆ H5Gclose_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title29 "H2")
  * [◆ H5Gcreate_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title30 "H2")
  * [◆ H5Gget_info_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title31 "H2")
  * [◆ H5Gget_info_by_idx_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title32 "H2")
  * [◆ H5Gget_info_by_name_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title33 "H2")
  * [◆ H5Gopen_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title34 "H2")
  * [◆ H5Lcreate_hard_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title35 "H2")
  * [◆ H5Lcreate_soft_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title36 "H2")
  * [◆ H5Ldelete_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title37 "H2")
  * [◆ H5Ldelete_by_idx_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title38 "H2")
  * [◆ H5Lexists_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title39 "H2")
  * [◆ H5Literate_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title40 "H2")
  * [◆ H5Mclose_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title41 "H2")
  * [◆ H5Mcreate_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title42 "H2")
  * [◆ H5Mget_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title43 "H2")
  * [◆ H5Mopen_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title44 "H2")
  * [◆ H5Mput_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title45 "H2")
  * [◆ H5Oclose_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title46 "H2")
  * [◆ H5Ocopy_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title47 "H2")
  * [◆ H5Oflush_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title48 "H2")
  * [◆ H5Oget_info_by_name_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title49 "H2")
  * [◆ H5Oopen_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title50 "H2")
  * [◆ H5Oopen_by_idx_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title51 "H2")
  * [◆ H5Orefresh_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title52 "H2")
  * [◆ H5Ropen_attr_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title53 "H2")
  * [◆ H5Ropen_object_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title54 "H2")
  * [◆ H5Ropen_region_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#title55 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#func-members)
Asynchronous Functions
## Detailed Description
List of the asynchronous functions.  

Note
    The argument `es_id` associated with the asynchronous APIs is the _event set id_. See H5ES for context. 
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aclose_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga121fb36f6571bfaf17eb0a92f3275560) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Acreate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga9b3ce1af9b445a6381479e8f3b58e6d9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Acreate_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga13dd8779381c7834b59a4ac5521c83ba) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) acpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aexists_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga126bf57a571c018286e67dbc14873925) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const char *attr_name, bool *exists, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aexists_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga2277f2644e0c0fe5ea03b523dfba0b1d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, bool *exists, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga8167b57603377742ae78a278dda27634) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_by_idx_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga1bef4a2dce9cfc0ddaa7472ac1e2d1dd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Aopen_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga42e2c5c72201f05b32e1c9dda6df0e30) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Aread_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7a74c8bd935164e2d19d4b253cde815a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype_id, void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Arename_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga5eb1ec49226fd0ec8e6dedc608f134f8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *old_name, const char *new_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Arename_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga27e7336cbf2182a85d8fba3cdf476d8e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *obj_name, const char *old_attr_name, const char *new_attr_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Awrite_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gae841dec1e2f4fecd88252307d20c1a59) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, const void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dclose_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga0933c085c070e86350e3548e337e4e7e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dcreate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gad18a501b7425902947237ec81706182e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dget_space_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga3eb6290902f6d709c762f80d067da3d3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Dopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga6df171aded96ec4926cd46000bf94f7d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dread_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga32856023e22cb8ed2ffa74b1651037b6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dread_multi_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7434c55da618b62d2c20cde4e0e040fc) (size_t count, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, void *buf[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dset_extent_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gab8e6d326fabbad5683ef6d0f669bae75) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dwrite_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7773f0c3503418421bcb535a95ee832e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const void *buf, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Dwrite_multi_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga1efc4f95c82571ce3897002c76469fdc) (size_t count, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_space_id[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const void *buf[], [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fclose_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gac4e2607f507f9b098d3987e7a90b3eeb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Fcreate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaa8e7c2dfa274c41d3ac30678ce77647e) (const char *filename, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Fflush_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaae5544d73d2100e21858ad49c9c53494) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) object_id, [H5F_scope_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#ac9db1b1211555797021daed9b54b8cdf) scope, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Fopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaba13f96a277191516514093a63aa9aee) (const char *filename, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) access_plist, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Freopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga540c3c083deadfb3165fd2c7468e206d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Gclose_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaedca4d2a02bac2dad772dc41dbdd6d68) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Gcreate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaae55331055ebaab0da4dad974b1b0c91) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Gget_info_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga50a254bdae888fcd08fe42465b5f386f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) *ginfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Gget_info_by_idx_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga87d9d82991d998f5f1dd4378e611f46b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) *ginfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Gget_info_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga79226604034b9f7d60aabbe48f53c18a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) *ginfo, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Gopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga51711cab3f8ae549df283aa2ba384527) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) gapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_hard_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaecc69b84cafb71d27dbcc244c35930c7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cur_loc_id, const char *cur_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_loc_id, const char *new_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lcreate_soft_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7ba84c7db6ef67e270fa7bc3413d4def) (const char *link_target, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) link_loc_id, const char *link_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ldelete_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga6074beb8fd1d7761db082fc611519b54) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ldelete_by_idx_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7dc3198d8cfaeb6a780863af8d0af253) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Lexists_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga4fddab130acbcef4dd8c7ccf75a31b2c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, bool *exists, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Literate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga0f56b7243d036cdeb5280d8c41f2436e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) group_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *idx_p, [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) op, void *op_data, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mclose_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaf23243773b9e65741cddafde432bf1ac) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mcreate_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga8bb9c2b6c4e695156c529532e2984184) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) val_type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mget_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gabc2fe75c57eb59f55103cb9d16dcf03c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_mem_type_id, const void *key, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) val_mem_type_id, void *value, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Mopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga128bb64acfc9bf04191424c673da358d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Mput_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaf7339648233e38eacd75eed7e0b6b5e9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) map_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) key_mem_type_id, const void *key, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) val_mem_type_id, const void *value, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Oclose_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga2979a4a45d3cd92c427735db6dcdf431) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) object_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Ocopy_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gae4364b88f8860f680d7782a721bba7af) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) src_loc_id, const char *src_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dst_loc_id, const char *dst_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) ocpypl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lcpl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Oflush_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga90669c99295348e3e169103404d3e61a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Oget_info_by_name_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga4c164275a34ff8208d8bcc9f701d47fe) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *oinfo, unsigned fields, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Oopen_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaefb9550c0b6faeb8c299dc67f92e1494) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Oopen_by_idx_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga3775880873cc64059e71f35f5693b4f6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *group_name, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) idx_type, [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) n, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) lapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Orefresh_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaf3b20ade9902067b99a16e53f171c657) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) oid, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Ropen_attr_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gadd7445737dd9d2d4e519fb7fcf3c3630) ([H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) rapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) aapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Ropen_object_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga4ef748212709bcf2030828dcd9dc66a2) (unsigned app_line, [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) rapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) oapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5Ropen_region_async](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga5f282a3adc66b0d2eafe5d333a6188a9) ([H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *ref_ptr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) rapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) oapl_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) es_id)  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga121fb36f6571bfaf17eb0a92f3275560)H5Aclose_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Aclose_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _attr_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Aclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda "Closes the specified attribute.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga9b3ce1af9b445a6381479e8f3b58e6d9)H5Acreate_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Acreate_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _attr_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _acpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _aapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Acreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga13dd8779381c7834b59a4ac5521c83ba)H5Acreate_by_name_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Acreate_by_name_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _obj_name_ ,   
|  | const char * |  _attr_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _acpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _aapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Acreate_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga004160c28e281455ec48aa7fe557ef8a "Creates an attribute attached to a specified object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga126bf57a571c018286e67dbc14873925)H5Aexists_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Aexists_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | const char * |  _attr_name_ ,   
|  | bool * |  _exists_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Aexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga293b5be270d90cd5e47f782ca9aec80b "Determines whether an attribute with a given name exists on an object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga2277f2644e0c0fe5ea03b523dfba0b1d)H5Aexists_by_name_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Aexists_by_name_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _obj_name_ ,   
|  | const char * |  _attr_name_ ,   
|  | bool * |  _exists_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Aexists_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaa1d2305651a4524f6aa0f8b56eec1a37 "Determines whether an attribute with a given name exists on an object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga8167b57603377742ae78a278dda27634)H5Aopen_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Aopen_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | const char * |  _attr_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _aapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Aopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga59863b205b6d93b2145f0fbca49656f7 "Opens an attribute for an object specified by object identifier and attribute name.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga1bef4a2dce9cfc0ddaa7472ac1e2d1dd)H5Aopen_by_idx_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Aopen_by_idx_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _obj_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _aapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Aopen_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d "Opens the nth attribute attached to an object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga42e2c5c72201f05b32e1c9dda6df0e30)H5Aopen_by_name_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Aopen_by_name_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _obj_name_ ,   
|  | const char * |  _attr_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _aapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Aopen_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10 "Opens an attribute for an object by object name and attribute name.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7a74c8bd935164e2d19d4b253cde815a)H5Aread_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Aread_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _attr_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dtype_id_ ,   
|  | void * |  _buf_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Aread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c "Reads the value of an attribute.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga5eb1ec49226fd0ec8e6dedc608f134f8)H5Arename_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Arename_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _old_name_ ,   
|  | const char * |  _new_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Arename()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga490dcd6db246c1fda7295badfce28203 "Renames an attribute.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga27e7336cbf2182a85d8fba3cdf476d8e)H5Arename_by_name_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Arename_by_name_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _obj_name_ ,   
|  | const char * |  _old_attr_name_ ,   
|  | const char * |  _new_attr_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Arename_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga21f8483c935d72187b98f5e7c2056140)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gae841dec1e2f4fecd88252307d20c1a59)H5Awrite_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Awrite_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _attr_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | const void * |  _buf_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
Asynchronous version of [H5Awrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c "Writes data to an attribute.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga0933c085c070e86350e3548e337e4e7e)H5Dclose_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dclose_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gad18a501b7425902947237ec81706182e)H5Dcreate_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dcreate_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga3eb6290902f6d709c762f80d067da3d3)H5Dget_space_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dget_space_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dget_space()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a "Returns an identifier for a copy of the dataspace for a dataset.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga6df171aded96ec4926cd46000bf94f7d)H5Dopen_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Dopen_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dopen()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga32856023e22cb8ed2ffa74b1651037b6)H5Dread_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dread_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | void * |  _buf_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dread()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7434c55da618b62d2c20cde4e0e040fc)H5Dread_multi_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dread_multi_async  | ( | size_t |  _count_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_space_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_space_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | void * |  _buf_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dread_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8eb1c838aff79a17de385d0707709915 "Reads raw data from a set of datasets into the provided buffers.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gab8e6d326fabbad5683ef6d0f669bae75)H5Dset_extent_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dset_extent_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _size_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7773f0c3503418421bcb535a95ee832e)H5Dwrite_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dwrite_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_space_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | const void * |  _buf_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga1efc4f95c82571ce3897002c76469fdc)H5Dwrite_multi_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dwrite_multi_async  | ( | size_t |  _count_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_type_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mem_space_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_space_id_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | const void * |  _buf_[],   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Dwrite_multi()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaf6213bf3a876c1741810037ff2bb85d8 "Writes raw data from a set buffers to a set of datasets.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gac4e2607f507f9b098d3987e7a90b3eeb)H5Fclose_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fclose_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Fclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaa8e7c2dfa274c41d3ac30678ce77647e)H5Fcreate_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Fcreate_async  | ( | const char * |  _filename_ ,   
---|---|---|---  
|  | unsigned |  _flags_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _fapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Fcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaae5544d73d2100e21858ad49c9c53494)H5Fflush_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Fflush_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _object_id_ ,   
---|---|---|---  
|  | [H5F_scope_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#ac9db1b1211555797021daed9b54b8cdf) |  _scope_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Fflush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124 "Flushes all buffers associated with a file to storage.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaba13f96a277191516514093a63aa9aee)H5Fopen_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Fopen_async  | ( | const char * |  _filename_ ,   
---|---|---|---  
|  | unsigned |  _flags_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _access_plist_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Fopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga540c3c083deadfb3165fd2c7468e206d)H5Freopen_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Freopen_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _file_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Freopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga3f213eb05c5419d63ba168c30036e47b "Returns a new identifier for a previously-opened HDF5 file.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaedca4d2a02bac2dad772dc41dbdd6d68)H5Gclose_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Gclose_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _group_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Gclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221 "Closes the specified group.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaae55331055ebaab0da4dad974b1b0c91)H5Gcreate_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Gcreate_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _gcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _gapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Gcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga50a254bdae888fcd08fe42465b5f386f)H5Gget_info_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Gget_info_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  |  [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) * |  _ginfo_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Gget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gad4be126ab7bbf2001435e8e70089f3d3 "Retrieves information about a group.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga87d9d82991d998f5f1dd4378e611f46b)H5Gget_info_by_idx_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Gget_info_by_idx_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  |  [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) * |  _ginfo_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Gget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga985f27ad1a164d99fa1f58c6de60ab00 "Retrieves information about a group, according to the group's position within an index.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga79226604034b9f7d60aabbe48f53c18a)H5Gget_info_by_name_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Gget_info_by_name_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  |  [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) * |  _ginfo_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Gget_info_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gadedd0c73c98f2ada69305f2992c3300e "Retrieves information about a group by its name.") 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga51711cab3f8ae549df283aa2ba384527)H5Gopen_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Gopen_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _gapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Gopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245) 

Since
    1.12.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaecc69b84cafb71d27dbcc244c35930c7)H5Lcreate_hard_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lcreate_hard_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _cur_loc_id_ ,   
---|---|---|---  
|  | const char * |  _cur_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _new_loc_id_ ,   
|  | const char * |  _new_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Lcreate_hard()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2 "Creates a hard link to an object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7ba84c7db6ef67e270fa7bc3413d4def)H5Lcreate_soft_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lcreate_soft_async  | ( | const char * |  _link_target_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _link_loc_id_ ,   
|  | const char * |  _link_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Lcreate_soft()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga6074beb8fd1d7761db082fc611519b54)H5Ldelete_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Ldelete_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Ldelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga7dc3198d8cfaeb6a780863af8d0af253)H5Ldelete_by_idx_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Ldelete_by_idx_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Ldelete_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gaaf5f820856afdd34f9070a797a246805 "Removes the n-th link in a group.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga4fddab130acbcef4dd8c7ccf75a31b2c)H5Lexists_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Lexists_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | bool * |  _exists_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga171be6e41dc1a464edc402df0ebdf801 "Determines whether a link with the specified name exists in a group.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga0f56b7243d036cdeb5280d8c41f2436e)H5Literate_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Literate_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _group_id_ ,   
---|---|---|---  
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _idx_p_ ,   
|  | [H5L_iterate2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_lpublic_8h.html#a9f26d305724d0969b3b25e100a109fc9) |  _op_ ,   
|  | void * |  _op_data_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * * 

Warning
    The returned value of the callback routine op will not be set in the return value for [H5Literate_async()](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga0f56b7243d036cdeb5280d8c41f2436e), so the `herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)` value should not be used for determining the return state of the callback routine.
Asynchronous version of [H5Literate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaf23243773b9e65741cddafde432bf1ac)H5Mclose_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mclose_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Mclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gad9a6a1199edc0fb4a50d831a213a5f44 "Terminates access to a map object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga8bb9c2b6c4e695156c529532e2984184)H5Mcreate_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mcreate_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _val_type_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Mcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#gac85972f70201429a93184408d40e4b09 "Creates a map object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gabc2fe75c57eb59f55103cb9d16dcf03c)H5Mget_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mget_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_mem_type_id_ ,   
|  | const void * |  _key_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _val_mem_type_id_ ,   
|  | void * |  _value_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Mget()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga585b7c3352cbca858e11bf50d3dd68ed "Retrieves a key-value pair from a map object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga128bb64acfc9bf04191424c673da358d)H5Mopen_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Mopen_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _mapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Mopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga426fbeac4c90849f2ac1855447fb8d43 "Opens a map object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaf7339648233e38eacd75eed7e0b6b5e9)H5Mput_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Mput_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _map_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _key_mem_type_id_ ,   
|  | const void * |  _key_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _val_mem_type_id_ ,   
|  | const void * |  _value_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Mput()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html#ga8cb388b30893cb79ed4d2bec4a8e8c4f "Adds a key-value pair to a map object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga2979a4a45d3cd92c427735db6dcdf431)H5Oclose_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Oclose_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _object_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Oclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga545ad7c54987013ebd50b40fe9e73c61 "Closes an object in an HDF5 file.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gae4364b88f8860f680d7782a721bba7af)H5Ocopy_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Ocopy_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _src_loc_id_ ,   
---|---|---|---  
|  | const char * |  _src_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dst_loc_id_ ,   
|  | const char * |  _dst_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _ocpypl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lcpl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Ocopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa94449be6f67f499be5ddd3fc44f4225 "Copies an object in an HDF5 file.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga90669c99295348e3e169103404d3e61a)H5Oflush_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Oflush_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Oflush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gad99f35048cba4534b6393214684f090f "Flushes all buffers associated with an HDF5 object to disk.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga4c164275a34ff8208d8bcc9f701d47fe)H5Oget_info_by_name_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Oget_info_by_name_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  |  [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) * |  _oinfo_ ,   
|  | unsigned |  _fields_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Oget_info_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga16d8ac07f9244cfccb42b5f309ca6b3c)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaefb9550c0b6faeb8c299dc67f92e1494)H5Oopen_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Oopen_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Oopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga9f635f58c7ddf17f87c253bfbca08bc1 "Opens an object in an HDF5 file by location identifier and path name.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga3775880873cc64059e71f35f5693b4f6)H5Oopen_by_idx_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Oopen_by_idx_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _group_name_ ,   
|  | [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) |  _idx_type_ ,   
|  | [H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) |  _order_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _n_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _lapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Oopen_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaeb66e5cbb3ca79890fc284a0b06762be "Opens the nth object in a group.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gaf3b20ade9902067b99a16e53f171c657)H5Orefresh_async()
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Orefresh_async  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _oid_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Orefresh()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf0318b68be9ab23a92b8a6bee0af9e2f "Refreshes all buffers associated with an HDF5 object.")
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#gadd7445737dd9d2d4e519fb7fcf3c3630)H5Ropen_attr_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Ropen_attr_async  | ( |  [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) * |  _ref_ptr_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _rapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _aapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Ropen_attr()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga2a00d73b9a13b9069d0ad39225dd9f71 "Opens the HDF5 attribute referenced.") 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga4ef748212709bcf2030828dcd9dc66a2)H5Ropen_object_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Ropen_object_async  | ( | unsigned |  _app_line_ ,   
---|---|---|---  
|  |  [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) * |  _ref_ptr_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _rapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _oapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Ropen_object()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd "Opens the HDF5 object referenced.") 

Since
    1.14.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_s_y_n_c.html#ga5f282a3adc66b0d2eafe5d333a6188a9)H5Ropen_region_async()
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5Ropen_region_async  | ( |  [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) * |  _ref_ptr_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _rapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _oapl_id_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _es_id_ )  
* * *
Asynchronous version of [H5Ropen_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga400154b014acd1736bae9a127f4e9a1a "Sets up a dataspace and selection as specified by a region reference.") 

Since
    1.14.0 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


