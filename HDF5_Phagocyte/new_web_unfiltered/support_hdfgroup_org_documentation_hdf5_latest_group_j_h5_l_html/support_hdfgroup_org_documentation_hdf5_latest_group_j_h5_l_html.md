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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title2 "H2")
  * [◆ H5Lcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title3 "H2")
  * [◆ H5Lcreate_external()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title4 "H2")
  * [◆ H5Lcreate_hard()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title5 "H2")
  * [◆ H5Lcreate_soft()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title6 "H2")
  * [◆ H5Ldelete()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title7 "H2")
  * [◆ H5Ldelete_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title8 "H2")
  * [◆ H5Lexists()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title9 "H2")
  * [◆ H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title10 "H2")
  * [◆ H5Lget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title11 "H2")
  * [◆ H5Lget_name_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title12 "H2")
  * [◆ H5Lget_value()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title13 "H2")
  * [◆ H5Lget_value_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title14 "H2")
  * [◆ H5Lis_registered()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title15 "H2")
  * [◆ H5Literate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title16 "H2")
  * [◆ H5Literate_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title17 "H2")
  * [◆ H5Lmove()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title18 "H2")
  * [◆ H5Lunregister()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title19 "H2")
  * [◆ H5Lvisit()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title20 "H2")
  * [◆ H5Lvisit_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#title21 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#func-members)
Java Link (H5L) Interface
## Detailed Description 

See also
     [Links (H5L)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html), C-API      [HDF5 Links](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l__u_g.html), User Guide 
##  Functions  
---  
static synchronized native void  |  [H5Lcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga2de23394679acee98bf6f2a292d4789b) (long src_loc, String src_name, long dst_loc, String dst_name, long lcpl_id, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native void  |  [H5Lcreate_external](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gac26ce5f756ea65daff378846f31e7d3a) (String file_name, String obj_name, long link_loc_id, String link_name, long lcpl_id, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native void  |  [H5Lcreate_hard](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga7a0a7e781b84d6e050fcc4e69f7431b7) (long cur_loc, String cur_name, long dst_loc, String dst_name, long lcpl_id, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native void  |  [H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga567aefdb95d7bc654203e0b6ffd0023f) (String link_target, long link_loc_id, String link_name, long lcpl_id, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native void  |  [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gaf7ede853f1657b08e3cfa7183d090278) (long loc_id, String name, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native void  |  [H5Ldelete_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga9946b71899c9af52b2eca7fb410d7a1e) (long loc_id, String group_name, int idx_type, int order, long n, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native boolean  |  [H5Lexists](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gaded9ce23610fa48d144ee051ea7d8ade) (long loc_id, String name, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) |  [H5Lget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga305046a96e35e869e627628ec65ab64c) (long loc_id, String name, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) |  [H5Lget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga203519895f271531b7895be853ced6c8) (long loc_id, String group_name, int idx_type, int order, long n, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native String  |  [H5Lget_name_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gad6c0b661047ef53098b691000d6f1b67) (long loc_id, String group_name, int idx_type, int order, long n, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native int  |  [H5Lget_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga3ced0a3bc07e1980e2fe2482156f27eb) (long loc_id, String name, String[] link_value, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native int  |  [H5Lget_value_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gaee15384cee1aa493fda3d1d1b4a6caef) (long loc_id, String group_name, int idx_type, int order, long n, String[] link_value, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native int  |  [H5Lis_registered](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gae88edc88cbe61b097164219030bce5ee) (int link_cls_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga4e9e84159546db7f17d3d0c6ee709371) (long grp_id, int idx_type, int order, long idx, [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) op, H5L_iterate_opdata_t op_data) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Literate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga52ab1fc0a9ed00a65b079b80885b2581) (long grp_id, String group_name, int idx_type, int order, long idx, [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) op, H5L_iterate_opdata_t op_data, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native void  |  [H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga29676bd4cc42bb548028eb35401e1a1d) (long src_loc, String src_name, long dst_loc, String dst_name, long lcpl_id, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native void  |  [H5Lunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga7a76061d8d0284c2fa987faecdc624db) (int link_cls_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga3248e35827ce50c36a48327579a53665) (long grp_id, int idx_type, int order, [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) op, H5L_iterate_opdata_t op_data) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Lvisit_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga55024d6616fd96dd5c5c0dc4bfbd67a8) (long loc_id, String group_name, int idx_type, int order, [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) op, H5L_iterate_opdata_t op_data, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga2de23394679acee98bf6f2a292d4789b)H5Lcopy()
| static synchronized native void H5Lcopy  | ( | long |  _src_loc_ ,   
---|---|---|---  
|  | String |  _src_name_ ,   
|  | long |  _dst_loc_ ,   
|  | String |  _dst_name_ ,   
|  | long |  _lcpl_id_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lcopy copies a link from one location to another. 

Parameters
     src_loc | IN: Location identifier of the source link   
---|---  
src_name | IN: Name of the link to be copied   
dst_loc | IN: Location identifier specifying the destination of the copy   
dst_name | IN: Name to be assigned to the new copy   
lcpl_id | IN: Link creation property list identifier   
lapl_id | IN: Link access property list identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gac26ce5f756ea65daff378846f31e7d3a)H5Lcreate_external()
| static synchronized native void H5Lcreate_external  | ( | String |  _file_name_ ,   
---|---|---|---  
|  | String |  _obj_name_ ,   
|  | long |  _link_loc_id_ ,   
|  | String |  _link_name_ ,   
|  | long |  _lcpl_id_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lcreate_external creates a new soft link to an external object, which is an object in a different HDF5 file from the location of the link. 

Parameters
     file_name | IN: Name of the target file containing the target object.   
---|---  
obj_name | IN: Path within the target file to the target object.   
link_loc_id | IN: The file or group identifier for the new link.   
link_name | IN: The name of the new link.   
lcpl_id | IN: Link creation property list identifier   
lapl_id | IN: Link access property list identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga7a0a7e781b84d6e050fcc4e69f7431b7)H5Lcreate_hard()
| static synchronized native void H5Lcreate_hard  | ( | long |  _cur_loc_ ,   
---|---|---|---  
|  | String |  _cur_name_ ,   
|  | long |  _dst_loc_ ,   
|  | String |  _dst_name_ ,   
|  | long |  _lcpl_id_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lcreate_hard creates a new hard link to a pre-existing object in an HDF5 file. 

Parameters
     cur_loc | IN: The file or group identifier for the target object.   
---|---  
cur_name | IN: Name of the target object, which must already exist.   
dst_loc | IN: The file or group identifier for the new link.   
dst_name | IN: The name of the new link.   
lcpl_id | IN: Link creation property list identifier   
lapl_id | IN: Link access property list identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | cur_name or dst_name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga567aefdb95d7bc654203e0b6ffd0023f)H5Lcreate_soft()
| static synchronized native void H5Lcreate_soft  | ( | String |  _link_target_ ,   
---|---|---|---  
|  | long |  _link_loc_id_ ,   
|  | String |  _link_name_ ,   
|  | long |  _lcpl_id_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lcreate_soft creates a new soft link to an object in an HDF5 file. 

Parameters
     link_target | IN: Path to the target object, which is not required to exist.   
---|---  
link_loc_id | IN: The file or group identifier for the new link.   
link_name | IN: The name of the new link.   
lcpl_id | IN: Link creation property list identifier   
lapl_id | IN: Link access property list identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | link_name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gaf7ede853f1657b08e3cfa7183d090278)H5Ldelete()
| static synchronized native void H5Ldelete  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Ldelete removes the link specified from a group. 

Parameters
     loc_id | IN: Identifier of the file or group containing the object.   
---|---  
name | IN: Name of the link to delete.   
lapl_id | IN: Link access property list identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga9946b71899c9af52b2eca7fb410d7a1e)H5Ldelete_by_idx()
| static synchronized native void H5Ldelete_by_idx  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _group_name_ ,   
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | long |  _n_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Ldelete_by_idx removes the nth link in a group according to the specified order and in the specified index. 

Parameters
     loc_id | IN: File or group identifier specifying location of subject group   
---|---  
group_name | IN: Name of subject group   
idx_type | IN: Index or field which determines the order   
order | IN: Order within field or index   
n | IN: Link for which to retrieve information   
lapl_id | IN: Link access property list identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | group_name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gaded9ce23610fa48d144ee051ea7d8ade)H5Lexists()
| static synchronized native boolean H5Lexists  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lexists checks if a link with a particular name exists in a group. 

Parameters
     loc_id | IN: Identifier of the file or group to query.   
---|---  
name | IN: The name of the link to check.   
lapl_id | IN: Link access property list identifier 

Returns
    a boolean, true if the name exists, otherwise false. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga305046a96e35e869e627628ec65ab64c)H5Lget_info()
| static synchronized native [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) H5Lget_info  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lget_info returns information about the specified link. 

Parameters
     loc_id | IN: Identifier of the file or group.   
---|---  
name | IN: Name of the link for which information is being sought.   
lapl_id | IN: Link access property list identifier 

Returns
    a buffer(H5L_info_t) for the link information. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga203519895f271531b7895be853ced6c8)H5Lget_info_by_idx()
| static synchronized native [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) H5Lget_info_by_idx  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _group_name_ ,   
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | long |  _n_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lget_info_by_idx opens a named datatype at the location specified by loc_id and return an identifier for the datatype. 

Parameters
     loc_id | IN: File or group identifier specifying location of subject group   
---|---  
group_name | IN: Name of subject group   
idx_type | IN: Type of index   
order | IN: Order within field or index   
n | IN: Link for which to retrieve information   
lapl_id | IN: Link access property list identifier 

Returns
    a buffer(H5L_info_t) for the link information. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | group_name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gad6c0b661047ef53098b691000d6f1b67)H5Lget_name_by_idx()
| static synchronized native String H5Lget_name_by_idx  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _group_name_ ,   
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | long |  _n_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lget_name_by_idx retrieves name of the nth link in a group, according to the order within a specified field or index. 

Parameters
     loc_id | IN: File or group identifier specifying location of subject group   
---|---  
group_name | IN: Name of subject group   
idx_type | IN: Type of index   
order | IN: Order within field or index   
n | IN: Link for which to retrieve information   
lapl_id | IN: Link access property list identifier 

Returns
    a String for the link name. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | group_name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga3ced0a3bc07e1980e2fe2482156f27eb)H5Lget_value()
| static synchronized native int H5Lget_value  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | String[] |  _link_value_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lget_value returns the link value of a symbolic link. Note that this function is a combination of [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698), [H5Lget_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.") and for external links, H5Lunpack_elink_val. 

Parameters
     loc_id | IN: Identifier of the file or group containing the object.   
---|---  
name | IN: Name of the symbolic link.   
link_value | OUT: Path of the symbolic link, or the file_name and path of an external file.   
lapl_id | IN: Link access property list identifier 

Returns
    the link type 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gaee15384cee1aa493fda3d1d1b4a6caef)H5Lget_value_by_idx()
| static synchronized native int H5Lget_value_by_idx  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _group_name_ ,   
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | long |  _n_ ,   
|  | String[] |  _link_value_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lget_value_by_idx retrieves value of the nth link in a group, according to the order within an index. Note that this function is a combination of [H5Lget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698), [H5Lget_val()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.") and for external links, H5Lunpack_elink_val. 

Parameters
     loc_id | IN: File or group identifier specifying location of subject group   
---|---  
group_name | IN: Name of subject group   
idx_type | IN: Type of index   
order | IN: Order within field or index   
n | IN: Link for which to retrieve information   
link_value | OUT: Path of the symbolic link, or the file_name and path of an external file.   
lapl_id | IN: Link access property list identifier 

Returns
    the link type 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | group_name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#gae88edc88cbe61b097164219030bce5ee)H5Lis_registered()
| static synchronized native int H5Lis_registered  | ( | int | _link_cls_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Lis_registered tests whether a user-defined link class is currently registered, either by the HDF5 Library or by the user through the use of H5Lregister. 

Parameters
     link_cls_id | IN: User-defined link class identifier  
---|--- 

Returns
    Returns a positive value if the link class has been registered and zero if it is unregistered. Otherwise returns a negative value; this may mean that the identifier is not a valid user-defined class identifier. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga4e9e84159546db7f17d3d0c6ee709371)H5Literate()
| static synchronized native int H5Literate  | ( | long |  _grp_id_ ,   
---|---|---|---  
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | long |  _idx_ ,   
|  | [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) |  _op_ ,   
|  | H5L_iterate_opdata_t |  _op_data_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Literate iterates through links in a group. 

Parameters
     grp_id | IN: Identifier specifying subject group   
---|---  
idx_type | IN: Type of index   
order | IN: Order of iteration within index   
idx | IN: Iteration position at which to start   
op | IN: Callback function passing data regarding the link to the calling application   
op_data | IN: User-defined pointer to data required by the application for its processing of the link 

Returns
    returns the return value of the first operator that returns a positive value, or zero if all members were processed with no operator returning non-zero. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga52ab1fc0a9ed00a65b079b80885b2581)H5Literate_by_name()
| static synchronized native int H5Literate_by_name  | ( | long |  _grp_id_ ,   
---|---|---|---  
|  | String |  _group_name_ ,   
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | long |  _idx_ ,   
|  | [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) |  _op_ ,   
|  | H5L_iterate_opdata_t |  _op_data_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Literate_by_name iterates through links in a group. 

Parameters
     grp_id | IN: Identifier specifying subject group   
---|---  
group_name | IN: Name of subject group   
idx_type | IN: Type of index   
order | IN: Order of iteration within index   
idx | IN: Iteration position at which to start   
op | IN: Callback function passing data regarding the link to the calling application   
op_data | IN: User-defined pointer to data required by the application for its processing of the link   
lapl_id | IN: Link access property list identifier 

Returns
    returns the return value of the first operator that returns a positive value, or zero if all members were processed with no operator returning non-zero. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | group_name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga29676bd4cc42bb548028eb35401e1a1d)H5Lmove()
| static synchronized native void H5Lmove  | ( | long |  _src_loc_ ,   
---|---|---|---  
|  | String |  _src_name_ ,   
|  | long |  _dst_loc_ ,   
|  | String |  _dst_name_ ,   
|  | long |  _lcpl_id_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lmove renames a link within an HDF5 file. 

Parameters
     src_loc | IN: Original file or group identifier.   
---|---  
src_name | IN: Original link name.   
dst_loc | IN: Destination file or group identifier.   
dst_name | IN: New link name.   
lcpl_id | IN: Link creation property list identifier to be associated with the new link.   
lapl_id | IN: Link access property list identifier to be associated with the new link. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga7a76061d8d0284c2fa987faecdc624db)H5Lunregister()
| static synchronized native void H5Lunregister  | ( | int | _link_cls_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Lunregister unregisters a class of user-defined links, preventing them from being traversed, queried, moved, etc. 

Parameters
     link_cls_id | IN: User-defined link class identifier  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga3248e35827ce50c36a48327579a53665)H5Lvisit()
| static synchronized native int H5Lvisit  | ( | long |  _grp_id_ ,   
---|---|---|---  
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) |  _op_ ,   
|  | H5L_iterate_opdata_t |  _op_data_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Lvisit recursively visits all links starting from a specified group. 

Parameters
     grp_id | IN: Identifier specifying subject group   
---|---  
idx_type | IN: Type of index   
order | IN: Order of iteration within index   
op | IN: Callback function passing data regarding the link to the calling application   
op_data | IN: User-defined pointer to data required by the application for its processing of the link 

Returns
    returns the return value of the first operator that returns a positive value, or zero if all members were processed with no operator returning non-zero. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_l.html#ga55024d6616fd96dd5c5c0dc4bfbd67a8)H5Lvisit_by_name()
| static synchronized native int H5Lvisit_by_name  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _group_name_ ,   
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | [H5L_iterate_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a9bdbaf7f70be7e08015af52af9247407) |  _op_ ,   
|  | H5L_iterate_opdata_t |  _op_data_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Lvisit_by_name recursively visits all links starting from a specified group. 

Parameters
     loc_id | IN: Identifier specifying subject group   
---|---  
group_name | IN: Name of subject group   
idx_type | IN: Type of index   
order | IN: Order of iteration within index   
op | IN: Callback function passing data regarding the link to the calling application   
op_data | IN: User-defined pointer to data required by the application for its processing of the link   
lapl_id | IN: link access property 

Returns
    returns the return value of the first operator that returns a positive value, or zero if all members were processed with no operator returning non-zero. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | group_name is null.   
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


