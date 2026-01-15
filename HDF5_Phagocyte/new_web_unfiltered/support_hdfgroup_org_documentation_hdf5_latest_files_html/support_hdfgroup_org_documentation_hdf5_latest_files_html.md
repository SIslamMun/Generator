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
  * [ Files](https://support.hdfgroup.org/documentation/hdf5/latest/_files.html#title0 "H1")
  * [ Tracking Free Space in HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_files.html#title1 "H2")
  * [ Removing Unused Space from HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_files.html#title2 "H2")
  * [ Creating an HDF5 File User Block](https://support.hdfgroup.org/documentation/hdf5/latest/_files.html#title3 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Files
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
* * *
#  Files
##  Tracking Free Space in HDF5 Files 

Problem
    You sometimes delete objects in HDF5 files and don't have new content to use the free space, but would like to reuse it in the future. 

Solution
    At file creation time, set the file space management strategy and persistence of free space tracking information via [H5Pset_file_space_strategy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l..."). 

Note
    This feature is only supported in HDF5 1.10.1+.
15 {
16 __label__ fail_fcpl, fail_fapl, fail_file;
17 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl, fapl, file;
18
19 if ((fcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977))) < 0) {
20 ret_val = EXIT_FAILURE;
21 goto fail_fcpl;
22 }
23#if H5_VERSION_GE(1, 10, 1)
24 if ([H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e)(fcpl, [H5F_FSPACE_STRATEGY_FSM_AGGR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a9cc492c4b5c936e48716a8dab3691bcca4eb2323fa7feed13452676d57cc27a87), 1, 4096) < 0) {
25#else
26#error HDF5 1.10.1+ required
27#endif
28 ret_val = EXIT_FAILURE;
29 goto fail_fapl;
30 }
31
32 if ((fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e))) < 0) {
33 ret_val = EXIT_FAILURE;
34 goto fail_fapl;
35 }
36#if H5_VERSION_GE(1, 10, 1)
37 if ([H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)(fapl, [H5F_LIBVER_V110](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a33c6cdc401a3a32dbf63d74019fad4b3), [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)) < 0) {
38#else
39#error HDF5 1.10.x+ required
40#endif
41 ret_val = EXIT_FAILURE;
42 goto fail_file;
43 }
44
45 if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("free_space.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), fcpl, fapl)) < 0) {
46 ret_val = EXIT_FAILURE;
47 goto fail_file;
48 }
49 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
50
51fail_file:
52 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
53fail_fapl:
54 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fcpl);
55fail_fcpl:;
56 } 

Discussion
    Free space tracking is supported only in HDF5 versions 1.10.x and higher. This has implications for the accessibility of your HDF5 files and should be considered carefully. If compatibility with previous versions of HDF5 must be maintained, space reclamation via `h5repack` might be an option.  
The file space strategy [H5F_FSPACE_STRATEGY_FSM_AGGR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a9cc492c4b5c936e48716a8dab3691bcca4eb2323fa7feed13452676d57cc27a87) is not the only option that supports free-space tracking. [H5F_FSPACE_STRATEGY_PAGE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a9cc492c4b5c936e48716a8dab3691bccacd625bd864903e71132c9098929f5a0a) is another option, which adds paged allocation and is used most effectively with page buffering.  
For an in-depth account of HDF5 file space management, paged-allocation, and page buffering, see the following documents: 
  * [HDF5 File Space Management](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/FileSpaceManagement.pdf)
  * [HDF5 File Space Management: Paged Aggregation](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/paged_aggregation.pdf)
  * [Page Buffering](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/RFC-Page_Buffering.pdf)



See Also
    See [Maintaining Compatibility with other HDF5 Library Versions](https://support.hdfgroup.org/documentation/hdf5/latest/_accessibility.html#CB_MaintainCompat) for HDF5 compatibility implications.
##  Removing Unused Space from HDF5 Files 

Problem
    Based on estimates or `h5stat` output you know that a large portion of an HDF5 file consists of free or unaccounted space, and you would like to remove it.
##  Creating an HDF5 File User Block 

Problem
    You would like to include certain ancillary, non-HDF5 content in an HDF5 file such that it can be accessed without the HDF5 library. 

Solution
    Use a file creation property list in which the user block size is set via [H5Pset_userblock()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga403bd982a2976c932237b186ed1cff4d "Sets user block size."). In the example below, we create an 8 MiB user block. 
60 {
61 __label__ fail_fcpl, fail_file;
62 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl, file;
63
64 if ((fcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977))) < 0) {
65 ret_val = EXIT_FAILURE;
66 goto fail_fcpl;
67 }
68 if ([H5Pset_userblock](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#ga403bd982a2976c932237b186ed1cff4d)(fcpl, 8192 * 1024) < 0) {
69 ret_val = EXIT_FAILURE;
70 goto fail_file;
71 }
72 if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("userblock.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), fcpl, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0) {
73 ret_val = EXIT_FAILURE;
74 goto fail_file;
75 }
76 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
77
78fail_file:
79 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fcpl);
80fail_fcpl:;
81 } 

Discussion
    The user block begins at offset 0 and must be at least 512 bytes and a power of 2. The HDF5 library ignores any content between the beginning of the file and the end of the user block.  
You can add or strip a user block to/from an existing HDF5 file with the `h5jam`/`h5unjam` tool, respectively.  

Warning
    If you try to embed content into the user block for use by other applications, pay close attention to how they handle space beyond the last used byte in the user block or the user block in general. In the worst case, applications might try to truncate the rest of the file and destroy the HDF5 portion of the file. 

See Also
    References to related recipes
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


