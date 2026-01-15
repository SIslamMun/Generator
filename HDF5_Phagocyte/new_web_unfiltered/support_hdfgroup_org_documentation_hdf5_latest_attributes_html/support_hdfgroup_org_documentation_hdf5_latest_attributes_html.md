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
  * [ Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_attributes.html#title0 "H1")
  * [ Creating "Large" HDF5 Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_attributes.html#title1 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Attributes
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
* * *
#  Attributes
##  Creating "Large" HDF5 Attributes 

Problem
    You would like to use HDF5 attributes the size of whose values exceeds a few dozen kilobytes 

Solution
    A file format change in the HDF5 1.8.x family of library releases made it possible to have attributes larger than about 64 KiB. Ensure that the lower library version bound for new HDF5 item creation is at least 1.8.x, and create larger attributes as usual.  
In the example below, we create an attribute whose value occupies 4 MiB. 

Note
    This feature is only supported in HDF5 1.8.x+
15 {
16 __label__ fail_attr, fail_aspace, fail_fapl, fail_file;
17 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl, file, aspace, attr;
18
19 if ((fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e))) < 0) {
20 ret_val = EXIT_FAILURE;
21 goto fail_fapl;
22 }
23#if H5_VERSION_GE(1, 10, 0)
24 if ([H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)(fapl, [H5F_LIBVER_V18](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2a434ca8def77a117013577c8cec6af0d8), [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)) < 0) {
25#elif H5_VERSION_GE(1, 8, 0)
26 if ([H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)(fapl, [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0), [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)) < 0) {
27#else
28#error Only HDF5 1.8.x and later supported.
29#endif
30 ret_val = EXIT_FAILURE;
31 goto fail_file;
32 }
33 if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("large_attribute.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl)) < 0) {
34 ret_val = EXIT_FAILURE;
35 goto fail_file;
36 }
37
38 if ((aspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)[]){1024 * 1024}, NULL)) < 0) {
39 ret_val = EXIT_FAILURE;
40 goto fail_aspace;
41 }
42 if ((attr = [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24)(file, "4MiB", [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea), aspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0) {
43 ret_val = EXIT_FAILURE;
44 goto fail_attr;
45 }
46
47 [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda)(attr);
48fail_attr:
49 [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(aspace);
50fail_aspace:
51 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
52fail_file:
53 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
54fail_fapl:;
55 } 

Discussion
    Large attributes are supported only in HDF5 versions 1.8.x and higher. This has implications for the accessibility of your HDF5 files and is your call.  
Since there are no size limitations for large attributes, it might seem tempting to mistake them for dataset stand-ins. This is not the case, for at least two reasons:
  1. Attributes decorate HDF5 objects, have their own local namespace, and can't be link targets.
  2. Attribute I/O treats the attribute value as atomic, i.e., there is no support for partial I/O. A large attribute will always be read or written in its entirety.



See Also
    See [Maintaining Compatibility with other HDF5 Library Versions](https://support.hdfgroup.org/documentation/hdf5/latest/_accessibility.html#CB_MaintainCompat) for HDF5 compatibility implications.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


