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
  * [ Accessibility](https://support.hdfgroup.org/documentation/hdf5/latest/_accessibility.html#title0 "H1")
  * [ Maintaining Compatibility with other HDF5 Library Versions](https://support.hdfgroup.org/documentation/hdf5/latest/_accessibility.html#title1 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Accessibility
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
* * *
#  Accessibility
##  Maintaining Compatibility with other HDF5 Library Versions 

Problem
    You want to ensure that the HDF5 files you produce or modify are accessible by all releavnt tools and applications 

Solution
    For HDF5 items (objects, attributes, etc.) that you would like to create in new or existing HDF5 files, ascertain the supported range of HDF5 library versions as lower and upper bounds. When creating new or opening existing HDF5 files, use a file access property list and configure the supported range via the [H5Pset_libver_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.") function.  
In the example below, we restrict HDF5 item creation to the HDF5 1.8.x family of library versions. 
15 {
16 __label__ fail_fapl, fail_file;
17 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl, file, aspace, attr;
18
19 if ((fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e))) < 0) {
20 ret_val = EXIT_FAILURE;
21 goto fail_fapl;
22 }
23#if H5_VERSION_GE(1, 10, 0)
24 if ([H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)(fapl, [H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8), [H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8)) < 0) {
25#elif H5_VERSION_GE(1, 8, 0)
26 if ([H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)(fapl, [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0), [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)) < 0) {
27#else
28#error Only HDF5 1.8.x and later supported.
29#endif
30 ret_val = EXIT_FAILURE;
31 goto fail_file;
32 }
33 if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("set_libver_bounds.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl)) < 0) {
34 ret_val = EXIT_FAILURE;
35 goto fail_file;
36 }
37
38 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
39fail_file:
40 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
41fail_fapl:;
42 } 

Discussion
    See RFC [Setting Bounds for Object Creation in HDF5 1.10.0](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/RFC-bounds.pdf) for a detailed and comprehensive account of HDF5 versioning (library, file format spec., etc.) and the [H5Pset_libver_bounds()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.") function.  
The default range [H5F_LIBVER_EARLIEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2abed98059b4a02d048b1eb3985fba5fa1) (low) - [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0) (high) offers the widest compatibility range, but may not be suitable for certain (feature-)use cases.  
The HDF5 library comes with a _forward-_ and _backward-compatibility_ guarantee: This means that the latest version of the library can always read HDF5 files created by a version realesed earlier (backward compatibility). It also means that a given release of the library can read the contents of HDF5 files created with later versions of the library as long as the files do not contain features introduced in later versions (forward compatibility). 

See Also
    See the recipe [Creating "Large" HDF5 Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_attributes.html#CB_LargeAttributes) for an example where we use HDF5 compatibility settings to enable the creation of large HDF5 attributes.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


