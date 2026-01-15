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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title2 "H2")
  * [◆ H5Gclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title3 "H2")
  * [◆ H5Gcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title4 "H2")
  * [◆ H5Gcreate_anon()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title5 "H2")
  * [◆ H5Gflush()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title6 "H2")
  * [◆ H5Gget_create_plist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title7 "H2")
  * [◆ H5Gget_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title8 "H2")
  * [◆ H5Gget_info_by_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title9 "H2")
  * [◆ H5Gget_info_by_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title10 "H2")
  * [◆ H5Gget_obj_info_all() [1/3]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title11 "H2")
  * [◆ H5Gget_obj_info_all() [2/3]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title12 "H2")
  * [◆ H5Gget_obj_info_all() [3/3]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title13 "H2")
  * [◆ H5Gget_obj_info_full()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title14 "H2")
  * [◆ H5Gget_obj_info_idx()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title15 "H2")
  * [◆ H5Gget_obj_info_max()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title16 "H2")
  * [◆ H5Gn_members()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title17 "H2")
  * [◆ H5Gopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title18 "H2")
  * [◆ H5Grefresh()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#title19 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#func-members)
Java Group (H5G) Interface
## Detailed Description 

See also
     [Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html), C-API      [HDF5 Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html), User Guide 
##  Functions  
---  
static int  |  [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga433f956d1b0c5fa81cefc1e390808668) (long group_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static long  |  [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gaf8763b10fec82a1066c37eda26570b37) (long loc_id, String name, long lcpl_id, long gcpl_id, long gapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static long  |  [H5Gcreate_anon](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga3be292a6445c23dad1373ca5bb9b227c) (long loc_id, long gcpl_id, long gapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5Gflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga4185fb6d933bc02871e35b029047a797) (long group_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5Gget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga0e889e318db767ec0554794d05818581) (long group_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) |  [H5Gget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gaece6d1057c42630d336dad0f3915b337) (long group_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) |  [H5Gget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga0ca8f9ea2f9e175f1b63dc8c71a63bbf) (long group_id, String group_name, int idx_type, int order, long n, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) |  [H5Gget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga27f51405659ca1bf6503b03a802fed04) (long group_id, String name, long lapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized int  |  [H5Gget_obj_info_all](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga87b461e7a741e2bb4a5c7513ed370a5d) (long loc_id, String name, String[] objNames, int[] objTypes, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] tokens) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized int  |  [H5Gget_obj_info_all](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga5079a661cd7979ad5b7246c63f492e31) (long loc_id, String name, String[] objNames, int[] objTypes, int[] ltype, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] tokens, int indx_type) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized int  |  [H5Gget_obj_info_all](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga46a64aa2f58e3c1982191b4e8c7dc7a6) (long loc_id, String name, String[] objNames, int[] objTypes, int[] ltype, long[] fno, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] tokens, int indx_type) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized int  |  [H5Gget_obj_info_full](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga3bf5679e85d3590e0c84b7317717e671) (long loc_id, String name, String[] objNames, int[] objTypes, int[] ltype, long[] fno, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] tokens, int indx_type, int indx_order) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized int  |  [H5Gget_obj_info_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gabecf95e812876d3bb286f9aac115b2b7) (long loc_id, String name, int idx, String[] oname, int[] type) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized int  |  [H5Gget_obj_info_max](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga5ba0732cadba0407ab34cb65ce7ab7f7) (long loc_id, String[] objNames, int[] objTypes, int[] lnkTypes, [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] objToken, long objMax) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized long  |  [H5Gn_members](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga8c944a4facf4977e490f540fb95edcc5) (long loc_id, String name) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static long  |  [H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gace85ef3beec51c23830a13ef48b675a5) (long loc_id, String name, long gapl_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized native void  |  [H5Grefresh](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gaa00f4af3b70ba95e672cc066e81a7b4b) (long group_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga433f956d1b0c5fa81cefc1e390808668)H5Gclose()
| static int H5Gclose  | ( | long | _group_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Gclose releases resources used by a group which was opened by a call to [H5Gcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) or [H5Gopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245). 

Parameters
     group_id | Group identifier to release.  
---|--- 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gaf8763b10fec82a1066c37eda26570b37)H5Gcreate()
| static long H5Gcreate  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | long |  _lcpl_id_ ,   
|  | long |  _gcpl_id_ ,   
|  | long |  _gapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Gcreate creates a new group with the specified name at the specified location, loc_id. 

Parameters
     loc_id | IN: The file or group identifier.   
---|---  
name | IN: The absolute or relative name of the new group.   
lcpl_id | IN: Identifier of link creation property list.   
gcpl_id | IN: Identifier of group creation property list.   
gapl_id | IN: Identifier of group access property list. (No group access properties have been implemented at this time; use H5P_DEFAULT.) 

Returns
    a valid group identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga3be292a6445c23dad1373ca5bb9b227c)H5Gcreate_anon()
| static long H5Gcreate_anon  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | long |  _gcpl_id_ ,   
|  | long |  _gapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Gcreate_anon creates a new empty group in the file specified by loc_id. 

Parameters
     loc_id | IN: File or group identifier specifying the file in which the new group is to be created.   
---|---  
gcpl_id | IN: Identifier of group creation property list.   
gapl_id | IN: Identifier of group access property list. (No group access properties have been implemented at this time; use H5P_DEFAULT.) 

Returns
    a valid group identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga4185fb6d933bc02871e35b029047a797)H5Gflush()
| static synchronized native void H5Gflush  | ( | long | _group_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Gflush causes all buffers associated with a group to be immediately flushed to disk without removing the data from the cache. 

Parameters
     group_id | IN: Identifier of the group to be flushed.  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga0e889e318db767ec0554794d05818581)H5Gget_create_plist()
| static synchronized native long H5Gget_create_plist  | ( | long | _group_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Gget_create_plist returns an identifier for the group creation property list associated with the group specified by group_id. 

Parameters
     group_id | IN: Identifier of the group.  
---|--- 

Returns
    an identifier for the group's creation property list 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gaece6d1057c42630d336dad0f3915b337)H5Gget_info()
| static synchronized native [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) H5Gget_info  | ( | long | _group_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Gget_info retrieves information about the group specified by group_id. The information is returned in the group_info struct. 

Parameters
     group_id | IN: Identifier of the group.  
---|--- 

Returns
    a structure in which group information is returned 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga0ca8f9ea2f9e175f1b63dc8c71a63bbf)H5Gget_info_by_idx()
| static synchronized native [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) H5Gget_info_by_idx  | ( | long |  _group_id_ ,   
---|---|---|---  
|  | String |  _group_name_ ,   
|  | int |  _idx_type_ ,   
|  | int |  _order_ ,   
|  | long |  _n_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Gget_info_by_idx retrieves information about a group, according to the group's position within an index. 

Parameters
     group_id | IN: File or group identifier.   
---|---  
group_name | IN: Name of group for which information is to be retrieved.   
idx_type | IN: Type of index by which objects are ordered   
order | IN: Order of iteration within index   
n | IN: Attribute's position in index   
lapl_id | IN: Link access property list. 

Returns
    a structure in which group information is returned 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga27f51405659ca1bf6503b03a802fed04)H5Gget_info_by_name()
| static synchronized native [H5G_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_g__info__t.html) H5Gget_info_by_name  | ( | long |  _group_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | long |  _lapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Gget_info_by_name retrieves information about the group group_name located in the file or group specified by loc_id. 

Parameters
     group_id | IN: File or group identifier.   
---|---  
name | IN: Name of group for which information is to be retrieved.   
lapl_id | IN: Link access property list. 

Returns
    a structure in which group information is returned 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga87b461e7a741e2bb4a5c7513ed370a5d)H5Gget_obj_info_all() [1/3]
| static synchronized int H5Gget_obj_info_all  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | String[] |  _objNames_ ,   
|  | int[] |  _objTypes_ ,   
|  |  [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] |  _tokens_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
retrieves information of all objects under the group (name) located in the file or group specified by loc_id. 

Parameters
     loc_id | IN: File or group identifier   
---|---  
name | IN: Name of group for which information is to be retrieved   
objNames | OUT: Names of all objects under the group, name.   
objTypes | OUT: Types of all objects under the group, name.   
tokens | OUT: Object token of all objects under the group, name. 

Returns
    the number of items found 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga5079a661cd7979ad5b7246c63f492e31)H5Gget_obj_info_all() [2/3]
| static synchronized int H5Gget_obj_info_all  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | String[] |  _objNames_ ,   
|  | int[] |  _objTypes_ ,   
|  | int[] |  _ltype_ ,   
|  |  [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] |  _tokens_ ,   
|  | int |  _indx_type_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
retrieves information of all objects under the group (name) located in the file or group specified by loc_id. 

Parameters
     loc_id | IN: File or group identifier   
---|---  
name | IN: Name of group for which information is to be retrieved   
objNames | OUT: Names of all objects under the group, name.   
objTypes | OUT: Types of all objects under the group, name.   
ltype | OUT: Link type   
tokens | OUT: Object token of all objects under the group, name.   
indx_type | IN: Index type for iterate 

Returns
    the number of items found 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga46a64aa2f58e3c1982191b4e8c7dc7a6)H5Gget_obj_info_all() [3/3]
| static synchronized int H5Gget_obj_info_all  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | String[] |  _objNames_ ,   
|  | int[] |  _objTypes_ ,   
|  | int[] |  _ltype_ ,   
|  | long[] |  _fno_ ,   
|  |  [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] |  _tokens_ ,   
|  | int |  _indx_type_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
retrieves information of all objects under the group (name) located in the file or group specified by loc_id. 

Parameters
     loc_id | IN: File or group identifier   
---|---  
name | IN: Name of group for which information is to be retrieved   
objNames | OUT: Names of all objects under the group, name.   
objTypes | OUT: Types of all objects under the group, name.   
ltype | OUT: Link type   
fno | OUT: File number   
tokens | OUT: Object token of all objects under the group, name.   
indx_type | IN: Index type for iterate 

Returns
    the number of items found 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga3bf5679e85d3590e0c84b7317717e671)H5Gget_obj_info_full()
| static synchronized int H5Gget_obj_info_full  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | String[] |  _objNames_ ,   
|  | int[] |  _objTypes_ ,   
|  | int[] |  _ltype_ ,   
|  | long[] |  _fno_ ,   
|  |  [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] |  _tokens_ ,   
|  | int |  _indx_type_ ,   
|  | int |  _indx_order_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
retrieves information of all objects under the group (name) located in the file or group specified by loc_id. 

Parameters
     loc_id | IN: File or group identifier   
---|---  
name | IN: Name of group for which information is to be retrieved   
objNames | OUT: Names of all objects under the group, name.   
objTypes | OUT: Types of all objects under the group, name.   
ltype | OUT: Link type   
fno | OUT: File number   
tokens | OUT: Object token of all objects under the group, name.   
indx_type | IN: Index type for iterate   
indx_order | IN: Index order for iterate 

Returns
    the number of items found 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gabecf95e812876d3bb286f9aac115b2b7)H5Gget_obj_info_idx()
| static synchronized int H5Gget_obj_info_idx  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | int |  _idx_ ,   
|  | String[] |  _oname_ ,   
|  | int[] |  _type_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Gget_obj_info_idx report the name and type of object with index 'idx' in a Group. The 'idx' corresponds to the index maintained by H5Giterate. Each link is returned, so objects with multiple links will be counted once for each link. 

Parameters
     loc_id | IN: file or group ID.   
---|---  
name | IN: name of the group to iterate, relative to the loc_id   
idx | IN: the index of the object to iterate.   
oname | OUT: the name of the object   
type | OUT: the type of the object 

Returns
    non-negative if successful, -1 if not. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga5ba0732cadba0407ab34cb65ce7ab7f7)H5Gget_obj_info_max()
| static synchronized int H5Gget_obj_info_max  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String[] |  _objNames_ ,   
|  | int[] |  _objTypes_ ,   
|  | int[] |  _lnkTypes_ ,   
|  |  [H5O_token_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__token__t.html)[] |  _objToken_ ,   
|  | long |  _objMax_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
retrieves information of all objects (recurvisely) under the group (name) located in the file or group specified by loc_id up to maximum specified by objMax. 

Parameters
     loc_id | IN: File or group identifier   
---|---  
objNames | OUT: Names of all objects under the group, name.   
objTypes | OUT: Types of all objects under the group, name.   
lnkTypes | OUT: Types of all links under the group, name.   
objToken | OUT: Object token of all objects under the group, name.   
objMax | IN: Maximum number of all objects under the group, name. 

Returns
    the number of items found 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#ga8c944a4facf4977e490f540fb95edcc5)H5Gn_members()
| static synchronized long H5Gn_members  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Gn_members report the number of objects in a Group. The 'objects' include everything that will be visited by H5Giterate. Each link is returned, so objects with multiple links will be counted once for each link. 

Parameters
     loc_id | file or group ID.   
---|---  
name | name of the group to iterate, relative to the loc_id 

Returns
    the number of members in the group or -1 if error. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gace85ef3beec51c23830a13ef48b675a5)H5Gopen()
| static long H5Gopen  | ( | long |  _loc_id_ ,   
---|---|---|---  
|  | String |  _name_ ,   
|  | long |  _gapl_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Gopen opens an existing group, name, at the location specified by loc_id. 

Parameters
     loc_id | IN: File or group identifier specifying the location of the group to be opened.   
---|---  
name | IN: Name of group to open.   
gapl_id | IN: Identifier of group access property list. 

Returns
    a valid group identifier if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | name is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_g.html#gaa00f4af3b70ba95e672cc066e81a7b4b)H5Grefresh()
| static synchronized native void H5Grefresh  | ( | long | _group_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Grefresh causes all buffers associated with a group to be cleared and immediately re-loaded with updated contents from disk. This function essentially closes the group, evicts all metadata associated with it from the cache, and then re-opens the group. The reopened group is automatically re-registered with the same ID. 

Parameters
     group_id | IN: Identifier of the group to be refreshed.  
---|--- 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


