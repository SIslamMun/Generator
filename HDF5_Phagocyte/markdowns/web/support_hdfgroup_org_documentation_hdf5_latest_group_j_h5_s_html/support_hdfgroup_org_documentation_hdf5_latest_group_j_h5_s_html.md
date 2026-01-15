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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title2 "H2")
  * [◆ H5Sclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title3 "H2")
  * [◆ H5Scombine_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title4 "H2")
  * [◆ H5Scombine_select()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title5 "H2")
  * [◆ H5Scopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title6 "H2")
  * [◆ H5Screate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title7 "H2")
  * [◆ H5Screate_simple()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title8 "H2")
  * [◆ H5Sdecode()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title9 "H2")
  * [◆ H5Sencode()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title10 "H2")
  * [◆ H5Sextent_copy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title11 "H2")
  * [◆ H5Sextent_equal()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title12 "H2")
  * [◆ H5Sget_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title13 "H2")
  * [◆ H5Sget_select_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title14 "H2")
  * [◆ H5Sget_select_elem_npoints()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title15 "H2")
  * [◆ H5Sget_select_elem_pointlist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title16 "H2")
  * [◆ H5Sget_select_hyper_blocklist()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title17 "H2")
  * [◆ H5Sget_select_hyper_nblocks()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title18 "H2")
  * [◆ H5Sget_select_npoints()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title19 "H2")
  * [◆ H5Sget_select_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title20 "H2")
  * [◆ H5Sget_simple_extent_dims()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title21 "H2")
  * [◆ H5Sget_simple_extent_ndims()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title22 "H2")
  * [◆ H5Sget_simple_extent_npoints()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title23 "H2")
  * [◆ H5Sget_simple_extent_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title24 "H2")
  * [◆ H5Sis_regular_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title25 "H2")
  * [◆ H5Sis_simple()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title26 "H2")
  * [◆ H5Smodify_select()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title27 "H2")
  * [◆ H5Soffset_simple() [1/2]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title28 "H2")
  * [◆ H5Soffset_simple() [2/2]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title29 "H2")
  * [◆ H5Sselect_adjust()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title30 "H2")
  * [◆ H5Sselect_all()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title31 "H2")
  * [◆ H5Sselect_copy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title32 "H2")
  * [◆ H5Sselect_elements()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title33 "H2")
  * [◆ H5Sselect_hyperslab() [1/2]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title34 "H2")
  * [◆ H5Sselect_hyperslab() [2/2]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title35 "H2")
  * [◆ H5Sselect_intersect_block()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title36 "H2")
  * [◆ H5Sselect_none()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title37 "H2")
  * [◆ H5Sselect_project_intersection()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title38 "H2")
  * [◆ H5Sselect_shape_same()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title39 "H2")
  * [◆ H5Sselect_valid()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title40 "H2")
  * [◆ H5Sset_extent_none()](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title41 "H2")
  * [◆ H5Sset_extent_simple() [1/2]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title42 "H2")
  * [◆ H5Sset_extent_simple() [2/2]](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#title43 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#func-members)
Java Dataspace (H5S) Interface
## Detailed Description 

See also
     [Dataspaces (H5S)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html), C-API      [Dataspaces and Partial I/O](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html), User Guide 
##  Functions  
---  
static int  |  [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga44b145f5c6977c082ac2c2dc121641fa) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5Scombine_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga2c791ebfb610b1e9ffb58c98b8fdd767) (long space_id, int op, long[] start, long[] stride, long[] count, long[] block) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException, IllegalArgumentException  
static synchronized native long  |  [H5Scombine_select](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga949ea42a300e23747d5e5d118edb4d6a) (long space1_id, int op, long space2_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static long  |  [H5Scopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac71b1379b6bb70b44922e77919982618) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static long  |  [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac68bfe5d0d1908e068d3195cbce1a0c2) (int type) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static long  |  [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac218f1e0de73b3030e9f6c3a9fd7aa76) (int rank, long[] dims, long[] maxdims) throws [HDF5Exception](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_exception.html), NullPointerException   
static synchronized native long  |  [H5Sdecode](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga59e8ee2d4af576d0583059e5b8c0daac) (byte[] buf) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native byte[]  |  [H5Sencode](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gadaf43806ad73fad7cae1ce6906aca5a5) (long obj_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native int  |  [H5Sextent_copy](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gada935bae1c757e30b622c763b0d37989) (long dest_space_id, long source_space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native boolean  |  [H5Sextent_equal](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaad5717635c3436220a048dc708660229) (long first_space_id, long second_space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5Sget_regular_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga0c1eba9dc9ce1a722cd04fbb55b2baa9) (long space_id, long[] start, long[] stride, long[] count, long[] block) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException, IllegalArgumentException  
static synchronized native int  |  [H5Sget_select_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga0c33054801e3bfb81bfa54d6f221ca7d) (long space_id, long[] start, long[] end) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native long  |  [H5Sget_select_elem_npoints](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga82a4c4e96511d68c158d3c98c02e1047) (long spaceid) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Sget_select_elem_pointlist](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac857ca6a8d083e0a9193a2393f6ddf2e) (long spaceid, long startpoint, long numpoints, long[] buf) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native int  |  [H5Sget_select_hyper_blocklist](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga9eac2ae4f81d7c3707bb143303e7e8ca) (long spaceid, long startblock, long numblocks, long[] buf) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native long  |  [H5Sget_select_hyper_nblocks](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gab6049c2e10acc3f5d7a1757ec3ddf0ea) (long spaceid) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5Sget_select_npoints](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga9a000ab27396eb7915eb4e80596a6568) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Sget_select_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga7c596592c53df4822bdbfbb136deb0a6) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga614f8c7dd6efb4e19eaf72eb98c16bfa) (long space_id, long[] dims, long[] maxdims) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native int  |  [H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga6f209df592e689994410a985e8e589d9) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5Sget_simple_extent_npoints](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaa28e8a692e81522c86bb333a8f4218c1) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Sget_simple_extent_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac86a5a8512e2bb2d30fea4326c232e91) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native boolean  |  [H5Sis_regular_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga304768df8054389de063f5e31d39e5a6) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native boolean  |  [H5Sis_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gae2f182e245b2671dcc663a950095ce6a) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5Smodify_select](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga814b2cb29fcdfe79892737f4337d0ef9) (long space1_id, int op, long space2_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Soffset_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga9d6693e01e9fa8ed222e7f249b509bce) (long space_id, byte[] offset) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized int  |  [H5Soffset_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga5c565a466431f21a585346bf9e6ea60a) (long space_id, long[] offset) throws [HDF5Exception](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_exception.html), NullPointerException   
static synchronized native void  |  [H5Sselect_adjust](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga5665d54df4c44293dd69be370e8f0587) (long space_id, long[][] offset) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native int  |  [H5Sselect_all](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga54bbdb68dc94192c2e81e30c3acc5c0f) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native void  |  [H5Sselect_copy](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaf60022ebf0dc60b2977487e6e65c36db) (long dst_id, long src_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized int  |  [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gacbe6a9818fe0027974c8b8d5be28f7b1) (long space_id, int op, int num_elements, long[][] coord2D) throws [HDF5Exception](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_exception.html), [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized int  |  [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga53b3c4e59745615f8820693e06b4e995) (long space_id, int op, byte[] start, byte[] stride, byte[] count, byte[] block) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException, IllegalArgumentException   
static synchronized native int  |  [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaf282f746c637aac32f13893a3d0d8a28) (long space_id, int op, long[] start, long[] stride, long[] count, long[] block) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException, IllegalArgumentException  
static synchronized native boolean  |  [H5Sselect_intersect_block](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga692f21b8ac83bb691854b05d50cebd5f) (long space_id, long[] start, long[] end) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static synchronized native int  |  [H5Sselect_none](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga7ff426a847637ee4d76c35cadd3db2e7) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native long  |  [H5Sselect_project_intersection](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac03a31452b876ba28a900a76780d02fd) (long src_space_id, long dst_space_id, long src_intersect_space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native boolean  |  [H5Sselect_shape_same](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga58b042bb96cb58ad68310a3ac33442f4) (long space1_id, long space2_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native boolean  |  [H5Sselect_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga0e952ca42ea362722bcbec0b2a449b44) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized native int  |  [H5Sset_extent_none](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga60e3c3978f0922e98d0570276f3d9d14) (long space_id) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static synchronized long  |  [H5Sset_extent_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaba1a98814211dbfdf4215bf9121a23a2) (long space_id, int rank, byte[] current_size, byte[] maximum_size) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException   
static synchronized native long  |  [H5Sset_extent_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga62604840228dc27655a192cd5a300555) (long space_id, int rank, long[] current_size, long[] maximum_size) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga44b145f5c6977c082ac2c2dc121641fa)H5Sclose()
| static int H5Sclose  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sclose releases a dataspace. 

Parameters
     space_id | Identifier of dataspace to release.  
---|--- 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga2c791ebfb610b1e9ffb58c98b8fdd767)H5Scombine_hyperslab()
| static synchronized native long H5Scombine_hyperslab  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | int |  _op_ ,   
|  | long[] |  _start_ ,   
|  | long[] |  _stride_ ,   
|  | long[] |  _count_ ,   
|  | long[] |  _block_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException, IllegalArgumentException  
static  
H5Scombine_hyperslab combines a hyperslab selection with the current selection for a dataspace, creating a new dataspace to return the generated selection. If the current selection is not a hyperslab, it is freed and the hyperslab parameters passed in are combined with the H5S_SEL_ALL hyperslab (ie. a selection composing the entire current extent). If STRIDE or BLOCK is NULL, they are assumed to be set to all '1'. 

Parameters
     space_id | IN: Dataspace ID of selection to use   
---|---  
op | IN: Operation to perform on current selection.   
start | IN: Offset of start of hyperslab   
stride | IN: Hyperslab stride.   
count | IN: Number of blocks included in hyperslab.   
block | IN: Size of block in hyperslab. 

Returns
    a dataspace ID on success / H5I_INVALID_HID on failure 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | an input array is null.   
IllegalArgumentException | an input array is invalid.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga949ea42a300e23747d5e5d118edb4d6a)H5Scombine_select()
| static synchronized native long H5Scombine_select  | ( | long |  _space1_id_ ,   
---|---|---|---  
|  | int |  _op_ ,   
|  | long |  _space2_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Scombine_select combines two existing hyperslab selections with an operation, returning a new dataspace with the resulting selection. The dataspace extent from space1 is copied for the dataspace extent of the newly created dataspace. 

Parameters
     space1_id | ID of the first dataspace   
---|---  
op | Operation to perform on current selection.   
space2_id | ID of the second dataspace 

Returns
    a dataspace ID on success / H5I_INVALID_HID on failure 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac71b1379b6bb70b44922e77919982618)H5Scopy()
| static long H5Scopy  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Scopy creates a new dataspace which is an exact copy of the dataspace identified by space_id. 

Parameters
     space_id | Identifier of dataspace to copy.   
---|--- 

Returns
    a dataspace identifier if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac68bfe5d0d1908e068d3195cbce1a0c2)H5Screate()
| static long H5Screate  | ( | int | _type_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Screate creates a new dataspace of a particular type. 

Parameters
     type | IN: The type of dataspace to be created.  
---|--- 

Returns
    a dataspace identifier 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac218f1e0de73b3030e9f6c3a9fd7aa76)H5Screate_simple()
| static long H5Screate_simple  | ( | int |  _rank_ ,   
---|---|---|---  
|  | long[] |  _dims_ ,   
|  | long[] |  _maxdims_ ) throws [HDF5Exception](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_exception.html), NullPointerException  
static  
H5Screate_simple creates a new simple data space and opens it for access. 

Parameters
     rank | IN: Number of dimensions of dataspace.   
---|---  
dims | IN: An array of the size of each dimension.   
maxdims | IN: An array of the maximum size of each dimension. 

Returns
    a dataspace identifier 

Exceptions
     [HDF5Exception](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | dims or maxdims is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga59e8ee2d4af576d0583059e5b8c0daac)H5Sdecode()
| static synchronized native long H5Sdecode  | ( | byte[] | _buf_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
---|---|---|---|---|---  
static  
H5Sdecode reconstructs the HDF5 data space object and returns a new object handle for it. 

Parameters
     buf | IN: Buffer for the data space object to be decoded.  
---|--- 

Returns
    a new object handle 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | buf is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gadaf43806ad73fad7cae1ce6906aca5a5)H5Sencode()
| static synchronized native byte[] H5Sencode  | ( | long | _obj_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
---|---|---|---|---|---  
static  
H5Sencode converts a data space description into binary form in a buffer. 

Parameters
     obj_id | IN: Identifier of the object to be encoded.  
---|--- 

Returns
    the buffer for the object to be encoded into. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gada935bae1c757e30b622c763b0d37989)H5Sextent_copy()
| static synchronized native int H5Sextent_copy  | ( | long |  _dest_space_id_ ,   
---|---|---|---  
|  | long |  _source_space_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Sextent_copy copies the extent from source_space_id to dest_space_id. This action may change the type of the dataspace. 

Parameters
     dest_space_id | IN: The identifier for the dataspace from which the extent is copied.   
---|---  
source_space_id | IN: The identifier for the dataspace to which the extent is copied. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaad5717635c3436220a048dc708660229)H5Sextent_equal()
| static synchronized native boolean H5Sextent_equal  | ( | long |  _first_space_id_ ,   
---|---|---|---  
|  | long |  _second_space_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Sextent_equal determines whether the dataspace extents of two dataspaces, space1_id and space2_id, are equal. 

Parameters
     first_space_id | IN: The identifier for the first dataspace.   
---|---  
second_space_id | IN: The identifier for the seconddataspace. 

Returns
    true if successful, else false 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga0c1eba9dc9ce1a722cd04fbb55b2baa9)H5Sget_regular_hyperslab()
| static synchronized native void H5Sget_regular_hyperslab  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | long[] |  _start_ ,   
|  | long[] |  _stride_ ,   
|  | long[] |  _count_ ,   
|  | long[] |  _block_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException, IllegalArgumentException  
static  
H5Sget_regular_hyperslab determines if a hyperslab selection is regular for the dataspace specified by space_id. The start, stride, count, and block arrays must be the same size as the rank of the dataspace. 

Parameters
     space_id | IN: Identifier of dataspace selection to modify   
---|---  
start | OUT: Offset of start of hyperslab   
stride | OUT: Hyperslab stride.   
count | OUT: Number of blocks included in hyperslab.   
block | OUT: Size of block in hyperslab. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | an output array is null.   
IllegalArgumentException | an output array is invalid.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga0c33054801e3bfb81bfa54d6f221ca7d)H5Sget_select_bounds()
| static synchronized native int H5Sget_select_bounds  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | long[] |  _start_ ,   
|  | long[] |  _end_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sget_select_bounds retrieves the coordinates of the bounding box containing the current selection and places them into user-supplied buffers. 
The start and end buffers must be large enough to hold the dataspace rank number of coordinates. 

Parameters
     space_id | Identifier of dataspace to release.   
---|---  
start | coordinates of lowest corner of bounding box.   
end | coordinates of highest corner of bounding box. 

Returns
    a non-negative value if successful,with start and end initialized. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | start or end is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga82a4c4e96511d68c158d3c98c02e1047)H5Sget_select_elem_npoints()
| static synchronized native long H5Sget_select_elem_npoints  | ( | long | _spaceid_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sget_select_elem_npoints returns the number of element points in the current dataspace selection. 

Parameters
     spaceid | Identifier of dataspace to release.  
---|--- 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac857ca6a8d083e0a9193a2393f6ddf2e)H5Sget_select_elem_pointlist()
| static synchronized native int H5Sget_select_elem_pointlist  | ( | long |  _spaceid_ ,   
---|---|---|---  
|  | long |  _startpoint_ ,   
|  | long |  _numpoints_ ,   
|  | long[] |  _buf_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sget_select_elem_pointlist returns an array of of element points in the current dataspace selection. The point coordinates have the same dimensionality (rank) as the dataspace they are located within, one coordinate per point. 

Parameters
     spaceid | Identifier of dataspace to release.   
---|---  
startpoint | first point to retrieve   
numpoints | number of points to retrieve   
buf | returns points startblock to startblock+num-1, each points is _rank_ longs. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | buf is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga9eac2ae4f81d7c3707bb143303e7e8ca)H5Sget_select_hyper_blocklist()
| static synchronized native int H5Sget_select_hyper_blocklist  | ( | long |  _spaceid_ ,   
---|---|---|---  
|  | long |  _startblock_ ,   
|  | long |  _numblocks_ ,   
|  | long[] |  _buf_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sget_select_hyper_blocklist returns an array of hyperslab blocks. The block coordinates have the same dimensionality (rank) as the dataspace they are located within. The list of blocks is formatted as follows:
```
   <"start" coordinate>, immediately followed by
   <"opposite" corner coordinate>, followed by
  the next "start" and "opposite" coordinates,
  etc.
  until all of the selected blocks have been listed.

```


Parameters
     spaceid | Identifier of dataspace to release.   
---|---  
startblock | first block to retrieve   
numblocks | number of blocks to retrieve   
buf | returns blocks startblock to startblock+num-1, each block is _rank_ * 2 (corners) longs. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | buf is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gab6049c2e10acc3f5d7a1757ec3ddf0ea)H5Sget_select_hyper_nblocks()
| static synchronized native long H5Sget_select_hyper_nblocks  | ( | long | _spaceid_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sget_select_hyper_nblocks returns the number of hyperslab blocks in the current dataspace selection. 

Parameters
     spaceid | Identifier of dataspace to release.  
---|--- 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga9a000ab27396eb7915eb4e80596a6568)H5Sget_select_npoints()
| static synchronized native long H5Sget_select_npoints  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sget_select_npoints determines the number of elements in the current selection of a dataspace. 

Parameters
     space_id | IN: Identifier of the dataspace object to query  
---|--- 

Returns
    the number of elements in the selection if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga7c596592c53df4822bdbfbb136deb0a6)H5Sget_select_type()
| static synchronized native int H5Sget_select_type  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sget_select_type retrieves the type of selection currently defined for the dataspace space_id. 

Parameters
     space_id | IN: Identifier of the dataspace object to query  
---|--- 

Returns
    the dataspace selection type if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga614f8c7dd6efb4e19eaf72eb98c16bfa)H5Sget_simple_extent_dims()
| static synchronized native int H5Sget_simple_extent_dims  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | long[] |  _dims_ ,   
|  | long[] |  _maxdims_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sget_simple_extent_dims returns the size and maximum sizes of each dimension of a dataspace through the dims and maxdims parameters. 

Parameters
     space_id | IN: Identifier of the dataspace object to query   
---|---  
dims | OUT: Pointer to array to store the size of each dimension.   
maxdims | OUT: Pointer to array to store the maximum size of each dimension. 

Returns
    the number of dimensions in the dataspace if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | dims or maxdims is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga6f209df592e689994410a985e8e589d9)H5Sget_simple_extent_ndims()
| static synchronized native int H5Sget_simple_extent_ndims  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sget_simple_extent_ndims determines the dimensionality (or rank) of a dataspace. 

Parameters
     space_id | IN: Identifier of the dataspace  
---|--- 

Returns
    the number of dimensions in the dataspace if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaa28e8a692e81522c86bb333a8f4218c1)H5Sget_simple_extent_npoints()
| static synchronized native long H5Sget_simple_extent_npoints  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sget_simple_extent_npoints determines the number of elements in a dataspace. 

Parameters
     space_id | ID of the dataspace object to query  
---|--- 

Returns
    the number of elements in the dataspace if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac86a5a8512e2bb2d30fea4326c232e91)H5Sget_simple_extent_type()
| static synchronized native int H5Sget_simple_extent_type  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sget_simple_extent_type queries a dataspace to determine the current class of a dataspace. 

Parameters
     space_id | Dataspace identifier.  
---|--- 

Returns
    a dataspace class name if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga304768df8054389de063f5e31d39e5a6)H5Sis_regular_hyperslab()
| static synchronized native boolean H5Sis_regular_hyperslab  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sis_regular_hyperslab retrieves a regular hyperslab selection for the dataspace specified by space_id. 

Parameters
     space_id | IN: Identifier of dataspace selection to query  
---|--- 

Returns
    a TRUE/FALSE for hyperslab selection if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gae2f182e245b2671dcc663a950095ce6a)H5Sis_simple()
| static synchronized native boolean H5Sis_simple  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sis_simple determines whether a dataspace is a simple dataspace. 

Parameters
     space_id | Identifier of the dataspace to query  
---|--- 

Returns
    true if is a simple dataspace 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga814b2cb29fcdfe79892737f4337d0ef9)H5Smodify_select()
| static synchronized native void H5Smodify_select  | ( | long |  _space1_id_ ,   
---|---|---|---  
|  | int |  _op_ ,   
|  | long |  _space2_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Smodify_select refine an existing hyperslab selection with an operation, using a second hyperslab. The first selection is modified to contain the result of space1 operated on by space2. 

Parameters
     space1_id | ID of the destination dataspace   
---|---  
op | Operation to perform on current selection.   
space2_id | ID of the source dataspace 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga9d6693e01e9fa8ed222e7f249b509bce)H5Soffset_simple() [1/2]
| static synchronized native int H5Soffset_simple  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | byte[] |  _offset_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Soffset_simple sets the offset of a simple dataspace space_id. 

Parameters
     space_id | IN: The identifier for the dataspace object to reset.   
---|---  
offset | IN: The offset at which to position the selection. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | offset array is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga5c565a466431f21a585346bf9e6ea60a)H5Soffset_simple() [2/2]
| static synchronized int H5Soffset_simple  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | long[] |  _offset_ ) throws [HDF5Exception](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_exception.html), NullPointerException  
static  
H5Soffset_simple sets the offset of a simple dataspace space_id. 

Parameters
     space_id | IN: The identifier for the dataspace object to reset.   
---|---  
offset | IN: The offset at which to position the selection. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | offset array is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga5665d54df4c44293dd69be370e8f0587)H5Sselect_adjust()
| static synchronized native void H5Sselect_adjust  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | long |  _offset_[][] ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sselect_adjust moves a selection by subtracting an offset from it. 

Parameters
     space_id | ID of dataspace to adjust   
---|---  
offset | Offset to subtract 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | offset is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga54bbdb68dc94192c2e81e30c3acc5c0f)H5Sselect_all()
| static synchronized native int H5Sselect_all  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sselect_all selects the entire extent of the dataspace space_id. 

Parameters
     space_id | IN: The identifier of the dataspace to be selected.  
---|--- 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaf60022ebf0dc60b2977487e6e65c36db)H5Sselect_copy()
| static synchronized native void H5Sselect_copy  | ( | long |  _dst_id_ ,   
---|---|---|---  
|  | long |  _src_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Sselect_copy copies all the selection information (including offset) from the source dataspace to the destination dataspace. 

Parameters
     dst_id | ID of the destination dataspace   
---|---  
src_id | ID of the source dataspace 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gacbe6a9818fe0027974c8b8d5be28f7b1)H5Sselect_elements()
| static synchronized int H5Sselect_elements  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | int |  _op_ ,   
|  | int |  _num_elements_ ,   
|  | long |  _coord2D_[][] ) throws [HDF5Exception](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_exception.html), [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sselect_elements selects array elements to be included in the selection for the space_id dataspace. 

Parameters
     space_id | Identifier of the dataspace.   
---|---  
op | operator specifying how the new selection is combined.   
num_elements | Number of elements to be selected.   
coord2D | A 2-dimensional array specifying the coordinates of the elements. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5Exception](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_exception.html) | Error in the data conversion   
---|---  
[HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
NullPointerException | cord array is   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga53b3c4e59745615f8820693e06b4e995)H5Sselect_hyperslab() [1/2]
| static synchronized int H5Sselect_hyperslab  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | int |  _op_ ,   
|  | byte[] |  _start_ ,   
|  | byte[] |  _stride_ ,   
|  | byte[] |  _count_ ,   
|  | byte[] |  _block_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException, IllegalArgumentException  
static  
H5Sselect_hyperslab selects a hyperslab region to add to the current selected region for the dataspace specified by space_id. The start, stride, count, and block arrays must be the same size as the rank of the dataspace. 

Parameters
     space_id | IN: Identifier of dataspace selection to modify   
---|---  
op | IN: Operation to perform on current selection.   
start | IN: Offset of start of hyperslab   
stride | IN: Hyperslab stride.   
count | IN: Number of blocks included in hyperslab.   
block | IN: Size of block in hyperslab. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | an input array is null.   
IllegalArgumentException | an input array is invalid.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaf282f746c637aac32f13893a3d0d8a28)H5Sselect_hyperslab() [2/2]
| static synchronized native int H5Sselect_hyperslab  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | int |  _op_ ,   
|  | long[] |  _start_ ,   
|  | long[] |  _stride_ ,   
|  | long[] |  _count_ ,   
|  | long[] |  _block_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException, IllegalArgumentException  
static  
H5Sselect_hyperslab selects a hyperslab region to add to the current selected region for the dataspace specified by space_id. The start, stride, count, and block arrays must be the same size as the rank of the dataspace. 

Parameters
     space_id | IN: Identifier of dataspace selection to modify   
---|---  
op | IN: Operation to perform on current selection.   
start | IN: Offset of start of hyperslab   
stride | IN: Hyperslab stride.   
count | IN: Number of blocks included in hyperslab.   
block | IN: Size of block in hyperslab. 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | an input array is null.   
IllegalArgumentException | an input array is invalid.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga692f21b8ac83bb691854b05d50cebd5f)H5Sselect_intersect_block()
| static synchronized native boolean H5Sselect_intersect_block  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | long[] |  _start_ ,   
|  | long[] |  _end_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sselect_intersect_block checks to see if the current selection in the dataspace intersects with the block given. 

Parameters
     space_id | ID of dataspace pointer to compare   
---|---  
start | Starting coordinate of block   
end | Opposite ("ending") coordinate of block 

Returns
    a TRUE if the current selection in the dataspace intersects with the block given FALSE otherwise 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
NullPointerException | offset is null.   
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga7ff426a847637ee4d76c35cadd3db2e7)H5Sselect_none()
| static synchronized native int H5Sselect_none  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sselect_none resets the selection region for the dataspace space_id to include no elements. 

Parameters
     space_id | IN: The identifier of the dataspace to be reset.  
---|--- 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gac03a31452b876ba28a900a76780d02fd)H5Sselect_project_intersection()
| static synchronized native long H5Sselect_project_intersection  | ( | long |  _src_space_id_ ,   
---|---|---|---  
|  | long |  _dst_space_id_ ,   
|  | long |  _src_intersect_space_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Sselect_project_intersection projects the intersection of the selections of src_space_id and src_intersect_space_id within the selection of src_space_id as a selection within the selection of dst_space_id. 

Parameters
     src_space_id | Selection that is mapped to dst_space_id, and intersected with src_intersect_space_id   
---|---  
dst_space_id | Selection that is mapped to src_space_id   
src_intersect_space_id | Selection whose intersection with src_space_id is projected to dst_space_id to obtain the result 

Returns
    a dataspace with a selection equal to the intersection of src_intersect_space_id and src_space_id projected from src_space to dst_space on success 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga58b042bb96cb58ad68310a3ac33442f4)H5Sselect_shape_same()
| static synchronized native boolean H5Sselect_shape_same  | ( | long |  _space1_id_ ,   
---|---|---|---  
|  | long |  _space2_id_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
static  
H5Sselect_shape_same checks to see if the current selection in the dataspaces are the same dimensionality and shape. This is primarily used for reading the entire selection in one swoop. 

Parameters
     space1_id | ID of 1st Dataspace pointer to compare   
---|---  
space2_id | ID of 2nd Dataspace pointer to compare 

Returns
    true if the selection is the same dimensionality and shape; false otherwise 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga0e952ca42ea362722bcbec0b2a449b44)H5Sselect_valid()
| static synchronized native boolean H5Sselect_valid  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sselect_valid verifies that the selection for the dataspace. 

Parameters
     space_id | The identifier for the dataspace in which the selection is being reset.  
---|--- 

Returns
    true if the selection is contained within the extent and FALSE if it is not or is an error. 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga60e3c3978f0922e98d0570276f3d9d14)H5Sset_extent_none()
| static synchronized native int H5Sset_extent_none  | ( | long | _space_id_ | ) |  throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html)  
---|---|---|---|---|---  
static  
H5Sset_extent_none removes the extent from a dataspace and sets the type to H5S_NONE. 

Parameters
     space_id | The identifier for the dataspace from which the extent is to be removed.  
---|--- 

Returns
    a non-negative value if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#gaba1a98814211dbfdf4215bf9121a23a2)H5Sset_extent_simple() [1/2]
| static synchronized long H5Sset_extent_simple  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | int |  _rank_ ,   
|  | byte[] |  _current_size_ ,   
|  | byte[] |  _maximum_size_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sset_extent_simple sets or resets the size of an existing dataspace. 

Parameters
     space_id | Dataspace identifier.   
---|---  
rank | Rank, or dimensionality, of the dataspace.   
current_size | Array containing current size of dataspace.   
maximum_size | Array containing maximum size of dataspace. 

Returns
    a dataspace identifier if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___j_h5_s.html#ga62604840228dc27655a192cd5a300555)H5Sset_extent_simple() [2/2]
| static synchronized native long H5Sset_extent_simple  | ( | long |  _space_id_ ,   
---|---|---|---  
|  | int |  _rank_ ,   
|  | long[] |  _current_size_ ,   
|  | long[] |  _maximum_size_ ) throws [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html), NullPointerException  
static  
H5Sset_extent_simple sets or resets the size of an existing dataspace. 

Parameters
     space_id | Dataspace identifier.   
---|---  
rank | Rank, or dimensionality, of the dataspace.   
current_size | Array containing current size of dataspace.   
maximum_size | Array containing maximum size of dataspace. 

Returns
    a dataspace identifier if successful 

Exceptions
     [HDF5LibraryException](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1exceptions_1_1_h_d_f5_library_exception.html) | Error from the HDF5 Library.   
---|---  
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


