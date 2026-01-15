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
  * [ HDF5 Filter Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#title1 "H2")
  * [ Programming Model for Applications](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#title2 "H2")
  * [ Programming Model for HDF5 Filter Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#title3 "H2")
  * [ Design](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#title4 "H2")
  * [ Building an HDF5 bzip2 Plugin Example](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Plugins
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 Filter Plugins
##  Introduction
HDF5 supports compression of data using a stackable pipeline of filters which can be implemented for reading and writing datasets, both at runtime and post‐process. These filters are supported as dynamically loadable plugins, and users can even implement custom filters of their own design.
##  Programming Model for Applications
This section describes the programming model for an application that uses a third-party HDF5 filter plugin to write or read data. For simplicity of presentation, it is assumed that the HDF5 filter plugin is available on the system in a default location. The HDF5 filter plugin is discussed in detail in the [Programming Model for HDF5 Filter Plugins](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#subsec_filter_plugins_prog) section.
###  Applying a Third-party Filter When Creating and Writing a Dataset
A third-party filter can be added to the HDF5 filter pipeline by using the H5Pset_filter function, as a user would do in the past. The identification number and the filter parameters should be available to the application. For example, if the application intends to apply the HDF5 bzip2 compression filter that was registered with The HDF Group and has an identification number 307 ([Registered Filters](https://github.com/HDFGroup/hdf5_plugins/blob/master/docs/RegisteredFilterPlugins.md)) then the application would follow the steps as outlined below: 
dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
status = [H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3) (dcpl, ([H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237))307, [H5Z_FLAG_MANDATORY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a7c50fa9c17a3e84280b3624e3e2515db), (size_t)6, cd_values);
dset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (file, DATASET, [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b), space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dcpl, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) (dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), wdata[0]);
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)
#define H5P_DATASET_CREATE
**Definition** H5Ppublic.h:60
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484)
#define H5S_ALL
**Definition** H5Spublic.h:33
[H5Z_FLAG_MANDATORY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a7c50fa9c17a3e84280b3624e3e2515db)
#define H5Z_FLAG_MANDATORY
**Definition** H5Zpublic.h:109
[H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237)
int H5Z_filter_t
Filter identifiers.
**Definition** H5Zpublic.h:55
[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
#define H5Dcreate
**Definition** H5version.h:1124
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
[H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3)
herr_t H5Pset_filter(hid_t plist_id, H5Z_filter_t filter, unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[])
Adds a filter to the filter pipeline.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
#define H5T_STD_I32LE
**Definition** H5Tpublic.h:471
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
###  Reading Data with an Applied Third-party Filter
An application does not need to do anything special to read the data with a third-party filter applied. For example, if one wants to read data written in the previous example, the following regular steps should be taken: 
file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc) (FILE, [H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dset = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) (file, DATASET, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c) (dset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), rdata[0]);
[H5F_ACC_RDONLY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a1c406ffa89f4acf5a332144a2683d394)
#define H5F_ACC_RDONLY
**Definition** H5Fpublic.h:28
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)
herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void *buf)
Reads raw data from a dataset into a provided buffer.
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
The command-line utility [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump), for example, will read and display the data as shown: 
HDF5 "h5ex_d_bzip2.h5" {
GROUP "/" {
DATASET "DS1" {
DATATYPE [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
DATASPACE SIMPLE { ( 32, 64 ) / ( 32, 64 ) }
STORAGE_LAYOUT {
CHUNKED ( 4, 8 )
SIZE 6410 (1.278:1 COMPRESSION)
}
FILTERS {
USER_DEFINED_FILTER {
FILTER_ID 307
COMMENT HDF5 bzip2 filter; see
https://\PLURL/docs/RegisteredFilterPlugins.md
PARAMS { 2 }
}
}
FILLVALUE {
FILL_TIME [H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72)
VALUE [H5D_FILL_VALUE_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a322c8263bd76fa5af8ff7636de5dfa23aa117582dd3ab4b104ce04029f0c7756a)
}
ALLOCATION_TIME {
[H5D_ALLOC_TIME_INCR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2ac898a96931fd3402d9e5646690c77636)
}
DATA {
...
}
}
}
}
[H5D_FILL_VALUE_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a322c8263bd76fa5af8ff7636de5dfa23aa117582dd3ab4b104ce04029f0c7756a)
@ H5D_FILL_VALUE_DEFAULT
**Definition** H5Dpublic.h:130
[H5D_FILL_TIME_IFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aa85b225308b0a277c4dd6fed7ee465a72)
@ H5D_FILL_TIME_IFSET
**Definition** H5Dpublic.h:119
[H5D_ALLOC_TIME_INCR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aab70b464cf3c5fc931dce0c4fe98b3d2ac898a96931fd3402d9e5646690c77636)
@ H5D_ALLOC_TIME_INCR
**Definition** H5Dpublic.h:94
If the filter can not be loaded then [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) will show the following: 
...
}
DATA {h5dump error: unable to print data
}
...
###  A Word of Caution When Using Custom Filters
Data goes through the HDF5 filter pipeline only when it is written to the file or read into application memory space from the file. For example, the I/O operation is triggered with a call to [H5Fflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124 "Flushes all buffers associated with a file to storage."), or when a data item (HDF5 metadata or a raw data chunk) is evicted from the cache or brought into the cache. Please notice that [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.")/[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") calls on the chunked datasets do not necessarily trigger I/O since the HDF5 Library uses a separate chunk cache.
A data item may remain in the cache until the HDF5 Library is closed. If the HDF5 plugin that has to be applied to the data item becomes unavailable before the file and all objects in the file are closed, an error will occur. The following example demonstrates the issue. Please notice the position of the [H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8 "Unregisters a filter.") call:
// Create a new group using compression.
gcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_GROUP_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a8330a95b257d45d6347a2daa96f261e9));
status = [H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3)(gcpl,H5Z_FILTER_BZIP2,[H5Z_FLAG_MANDATORY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#a7c50fa9c17a3e84280b3624e3e2515db),(size_t)1, cd_values);
group = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) (file, GNAME, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), gcpl, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
for (i=0; i < NGROUPS; i++) {
sprintf(name, "group_%d", i);
tmp_id = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) (group, name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
status = [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221)(tmp_id);
}
status = [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb) (gcpl);
status = [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221) (group);
// Unregister the filter. Call to H5Fclose will fail because the library tries
// to apply the filter that is not available anymore. This has a cascade effect
// on H5Fclose.
[H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8)(H5Z_FILTER_BZIP2);
status = [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124) (file);
[H5P_GROUP_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a8330a95b257d45d6347a2daa96f261e9)
#define H5P_GROUP_CREATE
**Definition** H5Ppublic.h:76
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
[H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)
#define H5Gcreate
**Definition** H5version.h:1240
[H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221)
herr_t H5Gclose(hid_t group_id)
Closes the specified group.
[H5Zunregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga6b8bcdde70c9256c50c7c62ba66380f8)
herr_t H5Zunregister(H5Z_filter_t id)
Unregisters a filter.
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)
herr_t H5Pclose(hid_t plist_id)
Terminates access to a property list.
Here is an error stack produced by the program: 
HDF5-DIAG: Error detected in HDF5 (xx.yy.zz) thread 0:
#000: H5F.c line **** in H5Fclose(): decrementing file ID failed
major: Object atom
minor: Unable to close file
#001: H5I.c line **** in H5I_dec_app_ref(): can't decrement ID ref count
major: Object atom
minor: Unable to decrement reference count
#002: H5F.c line **** in H5F_close(): can't close file
major: File accessibility
minor: Unable to close file
...
#026: H5Z.c line **** in H5Z_find(): required filter is not registered
major: Data filters
minor: Object not found
To avoid the problem make sure to close all objects to which the filter is applied and flush them using the [H5Fflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124 "Flushes all buffers associated with a file to storage.") call before unregistering the filter.
##  Programming Model for HDF5 Filter Plugins
This section describes how to create an HDF5 filter, an HDF5 filter plugin, and how to install the HDF5 plugin on the system.
###  Writing a Filter Function
The HDF5 filter function for the dynamically loaded filter feature should be written as a custom filter. This example shows how to define and register a simple filter that adds a checksum capability to the data stream.
The function that acts as the filter always returns zero (failure) if the `md5()` function was not detected at configuration time (left as an exercise for the reader). Otherwise the function is broken down to an input and output half. The output half calculates a checksum, increases the size of the output buffer if necessary, and appends the checksum to the end of the buffer. The input half calculates the checksum on the first part of the buffer and compares it to the checksum already stored at the end of the buffer. If the two differ then zero (failure) is returned, otherwise the buffer size is reduced to exclude the checksum. 
size_t md5_filter(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[],
size_t nbytes, size_t *buf_size, void **buf)
{
#ifdef HAVE_MD5
unsigned char cksum[16];
if (flags & H5Z_REVERSE) {
// Input
assert(nbytes >= 16);
md5(nbytes-16, *buf, cksum);
// Compare
if (memcmp(cksum, (char*)(*buf)+ nbytes- 16, 16)) {
return 0; // fail
}
// Strip off checksum
return nbytes - 16;
}
else {
// Output
md5(nbytes, *buf, cksum);
// Increase buffer size if necessary
if (nbytes + 16 > *buf_size) {
*buf_size = nbytes + 16;
*buf = realloc(*buf, *buf_size);
}
// Append checksum
memcpy((char*)(*buf)+nbytes, cksum, 16);
return nbytes+16;
}
#else
return 0; // fail
#endif
}
Once the filter function is defined it must be registered so the HDF5 library knows about it. Since we're testing this filter we choose one of the [H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237 "Filter identifiers.") numbers from the reserved range. We'll randomly choose 305.
#define FILTER_MD5 305
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) status = [H5Zregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b)(FILTER_MD5, "md5 checksum", md5_filter);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Zregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b)
herr_t H5Zregister(const void *cls)
Registers a new filter with the HDF5 library.
Now we can use the filter in a pipeline. We could have added the filter to the pipeline before defining or registering the filter as long as the filter was defined and registered by time we tried to use it (if the filter is marked as optional then we could have used it without defining it and the library would have automatically removed it from the pipeline for each chunk written before the filter was defined and registered).
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) chunk_size[3] = {10,10,10};
[H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)(dcpl, 3, chunk_size);
[H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3)(dcpl, FILTER_MD5, 0, 0, NULL);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, "dset", [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), space, dcpl);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)
herr_t H5Pset_chunk(hid_t plist_id, int ndims, const hsize_t dim[])
Sets the size of the chunks used to store a chunked layout dataset.
[H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09)
#define H5T_NATIVE_DOUBLE
**Definition** H5Tpublic.h:1036
See the example of a more sophisticated HDF5 bzip2 filter function in the [Building an HDF5 bzip2 Plugin Example](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#subsec_filter_plugins_build) section. The HDF5 bzip2 filter function is also available for download from [Filter Plugin Repository](https://github.com/HDFGroup/hdf5_plugins).
The user has to remember a few things when writing an HDF5 filter function. 
  1. An HDF5 filter is bidirectional.  
The filter handles both input and output to the file; a flag is passed to the filter to indicate the direction. 
  2. An HDF5 filter operates on a buffer.  
The filter reads data from a buffer, performs some sort of transformation on the data, places he result in the same or new buffer, and returns the buffer pointer and size to the caller. 
  3. An HDF5 filter should return zero in the case of failure. 


The signature of the HDF5 filter function and the accompanying filter structure (see the section below) are described in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html) [H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237 "Filter identifiers.").
###  Registering a Filter with The HDF Group
If you are writing a filter that will be used by others, it would be a good idea to request a filter identification number and register it with The HDF Group. Please follow the procedure described at [Registered Filters](https://github.com/HDFGroup/hdf5_plugins/blob/master/docs/RegisteredFilterPlugins.md).
The HDF Group anticipates that developers of HDF5 filter plugins will not only register new filters, but will also provide links to the source code and/or binaries for the corresponding HDF5 filter plugins.
It is very important for the users of the filter that developers provide filter information in the “name” field of the filter structure, for example: 
const [H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html) H5Z_BZIP2[1] = {{
[H5Z_CLASS_T_VERS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#acec2c757b38aefdb817ba7c7915778a9), // H5Z_class_t version
([H5Z_filter_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zpublic_8h.html#afae8461c70d47e63be2163af23362237))H5Z_FILTER_BZIP2, // Filter id number
1, // encoder_present flag (set to true)
1, // decoder_present flag (set to true)
"HDF5 bzip2 filter; see
https://\PLURL/docs/RegisteredFilterPlugins.md",
// Filter name for debugging
NULL, // The "can apply" callback
NULL, // The "set local" callback
([H5Z_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a130d8964a46667029c7d3c14572577c6))H5Z_filter_bzip2, // The actual filter function
}};
[H5Z_func_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#a130d8964a46667029c7d3c14572577c6)
size_t(* H5Z_func_t)(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes, size_t *buf_size, void **buf)
The filter operation callback function, defining a filter's operation on data.
**Definition** H5Zdevelop.h:153
[H5Z_CLASS_T_VERS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_zdevelop_8h.html#acec2c757b38aefdb817ba7c7915778a9)
#define H5Z_CLASS_T_VERS
**Definition** H5Zdevelop.h:31
[H5Z_class2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_z__class2__t.html)
**Definition** H5Zdevelop.h:162
The HDF5 Library and command-line tools have access to the “name” field. An application can use the H5Pget_filter<*> functions to retrieve information about the filters.
Using the example of the structure above, the [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) tool will print the string “HDF5 bzip2 filter found at …” pointing users to the applied filter (see the example in the [Reading Data with an Applied Third-party Filter](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_l__u_g.html#subsubsec_filter_plugins_model_read) section) thus solving the problem of the filter’s origin.
###  Creating an HDF5 Filter Plugin
The HDF5 filter plugin source should include: 
  1. The [H5PLextern.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html) header file from the HDF5 distribution. 
  2. The definition of the filter structure (see the example shown in the section above). 
  3. The filter function (for example, H5Z_filter_bzip2). 
  4. The two functions necessary for the HDF5 Library to find the correct type of the plugin library while loading it at runtime and to get information about the filter function:  
`H5PL_type_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) H5PLget_plugin_type(void)[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5);`  
---  
`const void* H5PLget_plugin_info(void)[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd);`  
  
Here is an example of the functions above for the HDF5 bzip2 filter:  
`H5PL_type_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lpublic_8h.html#a8d48cb770a80a3f84c969ec03b34d86f) H5PLget_plugin_type(void)[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a11942aa6c2beef4a76faa83b77feacd5) {return H5PL_TYPE_FILTER;}`  
---  
`const void* H5PLget_plugin_info(void)[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p_lextern_8h.html#a12bc51a0ed6df8d85b2fb6c4410cd1cd) {return H5Z_BZIP2;}`  
  5. Other functions such as the source of the compression library may also be included. 


Build the HDF5 filter plugin as a shared library. The following steps should be taken: 
  1. When compiling, point to the HDF5 header files. 
  2. Use the appropriate linking flags. 
  3. Link with any required external libraries. 
  4. For example, if libbz2.so is installed on a Linux system, the HDF5 bzip2 plugin library libH5Zbzip2.so may be linked with libbz2.so instead of including bzip2 source into the plugin library.  
The complete example of the HDF5 bzip2 plugin library is provided at [BZIP2 Filter Plugin](https://github.com/HDFGroup/hdf5_plugins/tree/master/BZIP2) and can be adopted for other plugins. 


###  Installing an HDF5 Filter Plugin
The default directory for an HDF5 filter plugin library is defined on UNIX-like systems as 
“/usr/local/[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)/lib/plugin”
[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)
**Definition** HDF5.F90:26
and on Windows systems as 
"%ALLUSERSPROFILE%/hdf5/lib/plugin".
The default path can be overwritten by a user with the [HDF5_PLUGIN_PATH](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a020d112b3c3251e7518461466777611d) environment variable. Several directories can be specified for the search path using “:” as a path separator for UNIX-like systems and “;” for Windows.
Readers are encouraged to try the example in the “Building an HDF5 bzip2 Plugin Example” section.
##  Design
Dynamic loading of the HDF5 filter plugin (or filter library) is triggered only by two events: when an application calls the [H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline.") function to set the filter for the first time, or when the data to which the filter is applied is read for the first time.
##  Building an HDF5 bzip2 Plugin Example
The HDF Group provides an repository of the HDF5 filter plugin that can be checked out from [BZIP2 Filter Plugin](https://github.com/HDFGroup/hdf5_plugins/tree/master/BZIP2).
It contains the source code for the bzip2 plugin library and an example that uses the plugin. It requires the HDF5 Library with the dynamically loaded feature and the bzip2 library being available on the system. The plugin and the example can be built using configure or CMake commands. For instructions on how to build with CMake, see the README.txt file in the source code distribution. The bzip2 library that can be built with CMake is available from: 
GIT_URL: "https://github.com/libarchive/bzip2.git"
GIT_BRANCH: "master"
See the documentation at [hdf5_plugins/docs](https://github.com/HDFGroup/hdf5_plugins/tree/master/docs) folder. In particular: [INSTALL_With_CMake](https://github.com/HDFGroup/hdf5_plugins/blob/master/docs/INSTALL_With_CMake.txt) [USING_HDF5_AND_CMake](https://github.com/HDFGroup/hdf5_plugins/blob/master/docs/USING_HDF5_AND_CMake.txt)
Previous Chapter [HDF5 References](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_r__u_g.html#sec_reference) - Next Chapter [Additional Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_a_r__u_g.html#sec_addition)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


