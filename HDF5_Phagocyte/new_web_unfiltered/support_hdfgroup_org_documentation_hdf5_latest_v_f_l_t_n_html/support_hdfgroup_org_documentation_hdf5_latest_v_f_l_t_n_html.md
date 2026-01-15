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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title0 "H1")
  * [ Using a File Driver](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title1 "H1")
  * [ Driver Header Files](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title2 "H2")
  * [ Creating and Opening Files](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title3 "H2")
  * [ Performing I/O](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title4 "H2")
  * [ File Driver Interchangeability](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title5 "H2")
  * [ Implementation of a Driver](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title6 "H1")
  * [ Mode Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title7 "H2")
  * [ File Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title8 "H2")
  * [ Open Files](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title9 "H2")
  * [ Closing Files](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title10 "H2")
  * [ File Keys](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title11 "H2")
  * [ Saving Modes Across Opens](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title12 "H2")
  * [ Address Space Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title13 "H1")
  * [ Userblock and Superblock](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title14 "H2")
  * [ Allocatiion of Format Regions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title15 "H2")
  * [ Freeing Format Regions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title16 "H2")
  * [ Querying the Address Range](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title17 "H2")
  * [ Data Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title18 "H1")
  * [ Contiguous I/O Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title19 "H2")
  * [ Flushing Cached Data](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title20 "H2")
  * [ Optimization Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title21 "H1")
  * [ Registration of a Driver](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title22 "H1")
  * [ Programming Note for C++ Developers Using C Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title23 "H2")
  * [ Querying Driver Information](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title24 "H1")
  * [ Miscellaneous](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html#title25 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Virtual File Layer
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Introduction
The HDF5 file format describes how HDF5 data structures and dataset raw data are mapped to a linear format address space and the HDF5 library implements that bidirectional mapping in terms of an API. However, the HDF5 format specifications do not indicate how the format address space is mapped onto storage and HDF (version 5 and earlier) simply mapped the format address space directly onto a single file by convention.
Since early versions of HDF5 it became apparent that users want the ability to map the format address space onto different types of storage (a single file, multiple files, local memory, global memory, network distributed global memory, a network protocol, etc.) with various types of maps. For instance, some users want to be able to handle very large format address spaces on operating systems that support only 2GB files by partitioning the format address space into equal-sized parts each served by a separate file. Other users want the same multi-file storage capability but want to partition the address space according to purpose (raw data in one file, object headers in another, global heap in a third, etc.) in order to improve I/O speeds.
In fact, the number of storage variations is probably larger than the number of methods that the HDF5 team is capable of implementing and supporting. Therefore, a Virtual File Layer API is being implemented which will allow application teams or departments to design and implement their own mapping between the HDF5 format address space and storage, with each mapping being a separate file driver (possibly written in terms of other file drivers). The HDF5 team will provide a small set of useful file drivers which will also serve as examples for those who which to write their own: 
[H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606) | This is the default driver which uses Posix file-system functions like read and write to perform I/O to a single file. All I/O requests are unbuffered although the driver does optimize file seeking operations to some extent.   
---|---  
[H5FD_STDIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dstdio_8h.html#a030a03b96a9f6e46035ce64e25389085) | This driver uses functions from 'stdio.h' to perform buffered I/O to a single file.   
[H5FD_CORE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dcore_8h.html#ae449696f6b86abcd1120beab21fff76a) | This driver performs I/O directly to memory and can be used to create small temporary files that never exist on permanent storage. This type of storage is generally very fast since the I/O consists only of memory-to-memory copy operations.   
[H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf) | This is the driver of choice for accessing files in parallel using MPI and MPI-IO. It is only predefined if the library is compiled with parallel I/O support.   
[H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6) | Large format address spaces are partitioned into more manageable pieces and sent to separate storage locations using an underlying driver of the user's choice. [The HDF5 h5repart Tool](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_t__u_g.html) can be used to change the sizes of the family members when stored as files or to convert a family of files to a single file or vice versa.   
#  Using a File Driver
Most application writers will use a driver defined by the HDF5 library or contributed by another programming team. This chapter describes how existing drivers are used.
##  Driver Header Files
Each file driver is defined in its own public header file which should be included by any application which plans to use that driver. The predefined drivers are in header files whose names begin with 'H5FD' followed by the driver name and '.h'. The 'hdf5.h' header file includes all the predefined driver header files.
Once the appropriate header file is included a symbol of the form 'H5FD_' followed by the upper-case driver name will be the driver identification number.(The driver name is by convention and might not apply to drivers which are not distributed with HDF5.) However, the value may change if the library is closed (e.g., by calling [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.")) and the symbol is referenced again.
##  Creating and Opening Files
In order to create or open a file one must define the method by which the storage is accessed(The access method also indicates how to translate the storage name to a storage server such as a file, network protocol, or memory.) and does so by creating a file access property list(The term "file access property list" is a misnomer since storage isn't required to be a file.) which is passed to the [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") function. A default file access property list is created by calling [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class.") and then the file driver information is inserted by calling a driver initialization function such as [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver."): 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
size_t member_size = 100*1024*1024; /*100MB*/
[H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d)(fapl, member_size, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("foo%05d.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e)
#define H5P_FILE_ACCESS
**Definition** H5Ppublic.h:56
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d)
herr_t H5Pset_fapl_family(hid_t fapl_id, hsize_t memb_size, hid_t memb_fapl_id)
Sets the file access property list to use the family driver.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)
herr_t H5Pclose(hid_t plist_id)
Terminates access to a property list.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
Each file driver will have its own initialization function whose name is H5Pset_fapl_ followed by the driver name and which takes a file access property list as the first argument followed by additional driver-dependent arguments.
An alternative to using the driver initialization function is to set the driver directly using the [H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e "Sets a file driver.") function.(This function is overloaded to operate on data transfer property lists also, as described below.) Its second argument is the file driver identifier, which may have a different numeric value from run to run depending on the order in which the file drivers are registered with the library. The third argument encapsulates the additional arguments of the driver initialization function. This method only works if the file driver writer has made the driver-specific property list structure a public datatype, which is often not the case. 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
static H5FD_family_fapl_t fa = {100*1024*1024, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)};
[H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e)(fapl, [H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6), &fa);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("foo.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
[H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6)
#define H5FD_FAMILY
**Definition** H5FDfamily.h:23
[H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e)
herr_t H5Pset_driver(hid_t plist_id, hid_t driver_id, const void *driver_info)
Sets a file driver.
It is also possible to query the file driver information from a file access property list by calling [H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") to determine the driver and then calling a driver-defined query function to obtain the driver information: 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) driver = [H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8)(fapl);
if ([H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606)==driver) {
/*nothing further to get*/
} else if ([H5FD_FAMILY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dfamily_8h.html#adbf24f060712550a2a9649589a6060c6)==driver) {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) member_fapl;
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) member_size;
[H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375)(fapl, &member_size, &member_fapl);
} else if (....) {
....
}
[H5FD_SEC2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dsec2_8h.html#a15ae1f958e1cf11cb239916d76b10606)
#define H5FD_SEC2
**Definition** H5FDsec2.h:24
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f)
uint64_t haddr_t
**Definition** H5public.h:355
[H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8)
hid_t H5Pget_driver(hid_t plist_id)
Returns low-lever driver identifier.
[H5Pget_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9195cb7b6c33ed8947486669702a4375)
herr_t H5Pget_fapl_family(hid_t fapl_id, hsize_t *memb_size, hid_t *memb_fapl_id)
Returns file access property list information.
##  Performing I/O
The [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") and [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") functions transfer data between application memory and the file. They both take an optional data transfer property list which has some general driver-independent properties and optional driver-defined properties. An application will typically perform I/O in one of three styles via the [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") or [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") function:
Like file access properties in the previous section, data transfer properties can be set using a driver initialization function or a general purpose function. For example, to set the MPI-IO driver to use independent access for I/O operations one would say: 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)(H5P_DATA_XFER);
[H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5)(dxpl, [H5FD_MPIO_INDEPENDENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a6c67820a8798cd75a6f0ebbb44e9a2af));
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, type, mspace, fspace, buffer, dxpl);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(dxpl);
[H5FD_MPIO_INDEPENDENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a6c67820a8798cd75a6f0ebbb44e9a2af)
@ H5FD_MPIO_INDEPENDENT
**Definition** H5FDmpi.h:37
[H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5)
herr_t H5Pset_dxpl_mpio(hid_t dxpl_id, H5FD_mpio_xfer_t xfer_mode)
Sets data transfer mode.
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)
herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void *buf)
Reads raw data from a dataset into a provided buffer.
The alternative is to initialize a driver defined C struct and pass it to the [H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e "Sets a file driver.") function: 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)(H5P_DATA_XFER);
static H5FD_mpio_dxpl_t dx = {[H5FD_MPIO_INDEPENDENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a6c67820a8798cd75a6f0ebbb44e9a2af)};
[H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e)(dxpl, [H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf), &dx);
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, type, mspace, fspace, buffer, dxpl);
[H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf)
#define H5FD_MPIO
**Definition** H5FDmpio.h:25
The transfer property list can be queried in a manner similar to the file access property list: the driver provides a function (or functions) to return various information about the transfer property list: 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) driver = [H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8)(dxpl);
if ([H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf)==driver) {
[H5FD_mpio_xfer_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16) xfer_mode;
[H5Pget_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga3f1295b2ee6dcb59288ac885c5562aa8)(dxpl, &xfer_mode);
} else {
....
}
[H5FD_mpio_xfer_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16)
H5FD_mpio_xfer_t
**Definition** H5FDmpi.h:36
[H5Pget_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga3f1295b2ee6dcb59288ac885c5562aa8)
herr_t H5Pget_dxpl_mpio(hid_t dxpl_id, H5FD_mpio_xfer_t *xfer_mode)
Returns the data transfer mode.
##  File Driver Interchangeability
The HDF5 specifications describe two things: the mapping of data onto a linear format address space and the C API which performs the mapping. However, the mapping of the format address space onto storage intentionally falls outside the scope of the HDF5 specs. This is a direct result of the fact that it is not generally possible to store information about how to access storage inside the storage itself. For instance, given only the file name '/arborea/1225/work/f%03d' the HDF5 library is unable to tell whether the name refers to a file on the local file system, a family of files on the local file system, a file on host 'arborea' port 1225, a family of files on a remote system, etc.
Two ways which library could figure out where the storage is located are: storage access information can be provided by the user, or the library can try all known file access methods. This implementation uses the former method.
In general, if a file was created with one driver then it isn't possible to open it with another driver. There are of course exceptions: a file created with MPIO could probably be opened with the sec2 driver, any file created by the sec2 driver could be opened as a family of files with one member, etc. In fact, sometimes a file must not only be opened with the same driver but also with the same driver properties. The predefined drivers are written in such a way that specifying the correct driver is sufficient for opening a file.
#  Implementation of a Driver
A driver is simply a collection of functions and data structures which are registered with the HDF5 library at runtime. The functions fall into these categories: 
  * Functions which operate on modes 
  * Functions which operate on files 
  * Functions which operate on the address space 
  * Functions which operate on data 
  * Functions for driver initialization 
  * Optimization functions


##  Mode Functions
Some drivers need information about file access and data transfers which are very specific to the driver. The information is usually implemented as a pair of pointers to C structs which are allocated and initialized as part of an HDF5 property list and passed down to various driver functions. There are two classes of settings: file access modes that describe how to access the file through the driver, and data transfer modes which are settings that control I/O operations. Each file opened by a particular driver may have a different access mode; each dataset I/O request for a particular file may have a different data transfer mode.
Since each driver has its own particular requirements for various settings, each driver is responsible for defining the mode structures that it needs. Higher layers of the library treat the structures as opaque but must be able to copy and free them. Thus, the driver provides either the size of the structure or a pair of function pointers for each of the mode types.
Example: The family driver needs to know how the format address space is partitioned and the file access property list to use for the family members. 
// Driver-specific file access properties
typedef struct H5FD_family_fapl_t {
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) memb_size; // size of each family member
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) memb_fapl; // file access property list for each family member
} H5FD_family_fapl_t;
// Driver specific data transfer properties
typedef struct H5FD_family_dxpl_t {
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) memb_dxpl_id; //data xfer property list of each member
} H5FD_family_dxpl_t;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
n order to copy or free one of these structures the member file access or data transfer properties must also be copied or freed. This is done by providing a copy and close function for each structure:
Example: The file access property list copy and close functions for the family driver: 
static void *
H5FD_family_fapl_copy(const void *_old_fa)
{
const H5FD_family_fapl_t *old_fa = (const H5FD_family_fapl_t*)_old_fa;
H5FD_family_fapl_t *new_fa = malloc(sizeof(H5FD_family_fapl_t));
assert(new_fa);
memcpy(new_fa, old_fa, sizeof(H5FD_family_fapl_t));
new_fa->memb_fapl_id = [H5Pcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b)(old_fa->memb_fapl_id);
return new_fa;
}
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
H5FD_family_fapl_free(void *_fa)
{
H5FD_family_fapl_t *fa = (H5FD_family_fapl_t*)_fa;
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fa->memb_fapl_id);
free(fa);
return 0;
}
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Pcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b)
hid_t H5Pcopy(hid_t plist_id)
Copies an existing property list to create a new property list.
Generally when a file is created or opened the file access properties for the driver are copied into the file pointer which is returned and they may be modified from their original value (for instance, the file family driver modifies the member size property when opening an existing family). In order to support the [H5Fget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga359585c49f82f5199178777b39e780f4 "Returns a file access property list identifier.") function the driver must provide a fapl_get callback which creates a copy of the driver-specific properties based on a particular file.
Example: The file family driver copies the member size file access property list into the return value: 
static void *
H5FD_family_fapl_get([H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *_file)
{
H5FD_family_t *file = (H5FD_family_t*)_file;
H5FD_family_fapl_t *fa = calloc(1, sizeof(H5FD_family_fapl_t*));
fa->memb_size = file->memb_size;
fa->memb_fapl_id = [H5Pcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b)(file->memb_fapl_id);
return fa;
}
[H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html)
**Definition** H5FDdevelop.h:340
##  File Functions
The higher layers of the library expect files to have a name and allow the file to be accessed in various modes. The driver must be able to create a new file, replace an existing file, or open an existing file. Opening or creating a file should return a handle, a pointer to a specialization of the [H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) struct, which allows read-only or read-write access and which will be passed to the other driver functions as they are called.(Read-only access is only appropriate when opening an existing file.) 
typedef struct {
// Public fields
[H5FD_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__class__t.html) *cls; //class data defined below
// Private fields -- driver-defined
} [H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html);
[H5FD_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__class__t.html)
**Definition** H5FDdevelop.h:199
Example: The family driver requires handles to the underlying storage, the size of the members for this particular file (which might be different than the member size specified in the file access property list if an existing file family is being opened), the name used to open the file in case additional members must be created, and the flags to use for creating those additional members. The eoa member caches the size of the format address space so the family members don't have to be queried in order to find it. 
// The description of a file belonging to this driver.
typedef struct H5FD_family_t {
[H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) pub; // public stuff, must be first
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) memb_fapl_id; // file access property list for members
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) memb_size; // maximum size of each member file
int nmembs; // number of family members
int amembs; // number of member slots allocated
[H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) **memb; // dynamic array of member pointers
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) eoa; // end of allocated addresses
char *name; // name generator printf format
unsigned flags; // flags for opening additional members
} H5FD_family_t;
Example: The sec2 driver needs to keep track of the underlying Unix file descriptor and also the end of format address space and current Unix file size. It also keeps track of the current file position and last operation (read, write, or unknown) in order to optimize calls to lseek. The device and inode fields are defined on Unix in order to uniquely identify the file and will be discussed below. 
typedef struct H5FD_sec2_t {
[H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) pub; // public stuff, must be first
int fd; // the unix file
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) eoa; // end of allocated region
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) eof; // end of file; current file size
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) pos; // current file I/O position
int op; // last operation
dev_t device; // file device number
ino_t inode; // file i-node number
} H5FD_sec2_t;
##  Open Files
All drivers must define a function for opening/creating a file. This function should have a prototype which is: 
`static H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) * open (const char *name, unsigned flags, hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) maxaddr)` | The file name name and file access property list fapl are the same as were specified in the [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") call. The flags are the same as in those calls also except the flag [H5F_ACC_CREAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5f311bcbc21086812e8b754f6354262b) is also present if the call was to H5Fcreate and they are documented in the '[H5Fpublic.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html)' file. The maxaddr argument is the maximum format address that the driver should be prepared to handle (the minimum address is always zero).   
---|---  
Example: The sec2 driver opens a Unix file with the requested name and saves information which uniquely identifies the file (the Unix device number and inode). 
static [H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *
H5FD_sec2_open(const char *name, unsigned flags, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id/*unused*/,
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) maxaddr)
{
unsigned o_flags;
int fd;
struct stat sb;
H5FD_sec2_t *file=NULL;
// Check arguments
if (!name || !*name) return NULL;
if (0==maxaddr || [HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c)==maxaddr) return NULL;
if (ADDR_OVERFLOW(maxaddr)) return NULL;
// Build the open flags
o_flags = ([H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4) & flags) ? O_RDWR : O_RDONLY;
if ([H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) & flags) o_flags |= O_TRUNC;
if ([H5F_ACC_CREAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5f311bcbc21086812e8b754f6354262b) & flags) o_flags |= O_CREAT;
if ([H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90) & flags) o_flags |= O_EXCL;
// Open the file
if ((fd=open(name, o_flags, 0666))<0) return NULL;
if (fstat(fd, &sb)<0) {
close(fd);
return NULL;
}
// Create the new file struct
file = calloc(1, sizeof(H5FD_sec2_t));
file->fd = fd;
file->eof = sb.st_size;
file->pos = [HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c);
file->op = [OP_UNKNOWN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628a3070ec4253e2531351d7aeb586069a54);
file->device = sb.st_dev;
file->inode = sb.st_ino;
return ([H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html)*)file;
}
[OP_UNKNOWN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628a3070ec4253e2531351d7aeb586069a54)
@ OP_UNKNOWN
**Definition** H5FDprivate.h:52
[H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4)
#define H5F_ACC_RDWR
**Definition** H5Fpublic.h:29
[H5F_ACC_CREAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5f311bcbc21086812e8b754f6354262b)
#define H5F_ACC_CREAT
**Definition** H5Fpublic.h:33
[H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90)
#define H5F_ACC_EXCL
**Definition** H5Fpublic.h:31
[HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c)
#define HADDR_UNDEF
**Definition** H5public.h:367
##  Closing Files
Closing a file simply means that all cached data should be flushed to the next lower layer, the file should be closed at the next lower layer, and all file-related data structures should be freed. All information needed by the close function is already present in the file handle. 
`static herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) close (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file)` | The file argument is the handle which was returned by the open function, and the close should free only memory associated with the driver-specific part of the handle (the public parts will have already been released by HDF5's virtual file layer).   
---|---  
Example: The sec2 driver just closes the underlying Unix file, making sure that the actual file size is the same as that known to the library by writing a zero to the last file position it hasn't been written by some previous operation (which happens in the same code which flushes the file contents and is shown below). 
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
H5FD_sec2_close([H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *_file)
{
H5FD_sec2_t *file = (H5FD_sec2_t*)_file;
if (H5FD_sec2_flush(_file)<0) return -1;
if (close(file->fd)<0) return -1;
free(file);
return 0;
}
##  File Keys
Occasionally an application will attempt to open a single file more than one time in order to obtain multiple handles to the file. HDF5 allows the files to share information(For instance, writing data to one handle will cause the data to be immediately visible on the other handle.) but in order to accomplish this HDF5 must be able to tell when two names refer to the same file. It does this by associating a driver-defined key with each file opened by a driver and comparing the key for an open request with the keys for all other files currently open by the same driver. 
`const int cmp (const H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *f1, const H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *f2)` | The driver may provide a function which compares two files f1 and f2 belonging to the same driver and returns a negative, positive, or zero value a la the strcmp function.(The ordering is arbitrary as long as it's consistent within a particular file driver.) If this function is not provided then HDF5 assumes that all calls to the open callback return unique files regardless of the arguments and it is up to the application to avoid doing this if that assumption is incorrect.   
---|---  
Each time a file is opened the library calls the cmp function to compare that file with all other files currently open by the same driver and if one of them matches (at most one can match) then the file which was just opened is closed and the previously opened file is used instead.
Opening a file twice with incompatible flags will result in failure. For instance, opening a file with the truncate flag is a two step process which first opens the file without truncation so keys can be compared, and if no matching file is found already open then the file is closed and immediately reopened with the truncation flag set (if a matching file is already open then the truncating open will fail).
Example: The sec2 driver uses the Unix device and i-node as the key. They were initialized when the file was opened. 
static int
H5FD_sec2_cmp(const [H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *_f1, const [H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *_f2)
{
const H5FD_sec2_t *f1 = (const H5FD_sec2_t*)_f1;
const H5FD_sec2_t *f2 = (const H5FD_sec2_t*)_f2;
if (f1->device < f2->device) return -1;
if (f1->device > f2->device) return 1;
if (f1->inode < f2->inode) return -1;
if (f1->inode > f2->inode) return 1;
return 0;
}
##  Saving Modes Across Opens
Some drivers may also need to store certain information in the file superblock in order to be able to reliably open the file at a later date. This is done by three functions: one to determine how much space will be necessary to store the information in the superblock, one to encode the information, and one to decode the information. These functions are optional, but if any one is defined then the other two must also be defined. 
Function  | Description   
---|---  
`static hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) sb_size (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file)` | The sb_size function returns the number of bytes necessary to encode information needed later if the file is reopened.   
`static herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) sb_encode (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, char *name, unsigned char *buf)` | The sb_encode function encodes information from the file into buffer buf allocated by the caller. It also writes an 8-character (plus null termination) into the name argument, which should be a unique identification for the driver.   
`static herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) sb_decode (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, const char *name, const unsigned char *buf)` | The sb_decode function looks at the name decodes data from the buffer buf and updates the file argument with the new information, advancing *p in the process.   
The part of this which is somewhat tricky is that the file must be readable before the superblock information is decoded. File access modes fall outside the scope of the HDF5 file format, but they are placed inside the boot block for convenience.(File access modes do not describe data, but rather describe how the HDF5 format address space is mapped to the underlying file(s). Thus, in general the mapping must be known before the file superblock can be read. However, the user usually knows enough about the mapping for the superblock to be readable and once the superblock is read the library can fill in the missing parts of the mapping.)
#  Address Space Functions
HDF5 does not assume that a file is a linear address space of bytes. Instead, the library will call functions to allocate and free portions of the HDF5 format address space, which in turn map onto functions in the file driver to allocate and free portions of file address space. The library tells the file driver how much format address space it wants to allocate and the driver decides what format address to use and how that format address is mapped onto the file address space. Usually the format address is chosen so that the file address can be calculated in constant time for data I/O operations (which are always specified by format addresses).
##  Userblock and Superblock
The HDF5 format allows an optional userblock to appear before the actual HDF5 data in such a way that if the userblock is sucked out of the file and everything remaining is shifted downward in the file address space, then the file is still a valid HDF5 file. The userblock size can be zero or any multiple of two greater than or equal to 512 and the file superblock begins immediately after the userblock.
HDF5 allocates space for the userblock and superblock by calling an allocation function defined below, which must return a chunk of memory at format address zero on the first call.
##  Allocatiion of Format Regions
The library makes many types of allocation requests: 
[H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce) | userblock   
---|---  
[H5FD_MEM_BTREE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a29b8528e16990fbe265682559b917fa3) | An allocation request for a node of a B-tree.   
[H5FD_MEM_DRAW](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ab435b061fede0393691acbe7cac2cb2e) | An allocation request for the raw data of a dataset.   
[H5FD_MEM_GHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a020d6245f874e8262058c3278fefe58e) | An allocation request for a global heap collection. Global heaps are used to store certain types of references such as dataset region references. The set of all global heap collections can become quite large.   
[H5FD_MEM_LHEAP](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae7536174d3ae2a842a71d6c192b43a13) | An allocation request for a local heap. Local heaps are used to store the names which are members of a group. The combined size of all local heaps is a function of the number of object names in the file.   
[H5FD_MEM_OHDR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a4337f7056fb57717e82fa1081f496d75) | An allocation request for (part of) an object header. Object headers are relatively small and include meta information about objects (like the data space and type of a dataset) and attributes.   
When a chunk of memory is freed the library adds it to a free list and allocation requests are satisfied from the free list before requesting memory from the file driver. Each type of allocation request enumerated above has its own free list, but the file driver can specify that certain object types can share a free list. It does so by providing an array which maps a request type to a free list. If any value of the map is H5MF_DEFAULT (zero) then the object's own free list is used. The special value H5MF_NOLIST indicates that the library should not attempt to maintain a free list for that particular object type, instead calling the file driver each time an object of that type is freed.
Mappings predefined in the '[H5FDdevelop.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html)' file are: 
[H5FD_FLMAP_SINGLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a4c63846487a43d7a3d589ac2852c9078) | All memory usage types are mapped to a single free list.   
---|---  
[H5FD_FLMAP_DICHOTOMY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#ac828cc26131c2df37fdb84ca13272540) | Memory usage is segregated into meta data and raw data for the purposes of memory management.   
[H5FD_FLMAP_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a49f2128fbc831c73ec19f7e24ec6ee24) | Each memory usage type has its own free list.   
Example: To make a map that manages object headers on one free list and everything else on another free list one might initialize the map with the following code: (the use of [H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce) is arbitrary) 
[H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) mt, map[[H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566)];
for (mt = 0; mt < [H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566); mt++) {
map[mt] = ([H5FD_MEM_OHDR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a4337f7056fb57717e82fa1081f496d75)== mt) ? mt : [H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce);
}
[H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b)
enum H5F_mem_t H5FD_mem_t
**Definition** H5FDpublic.h:273
[H5FD_MEM_NTYPES](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a3be201777da432df4a8e2c1b618a7566)
@ H5FD_MEM_NTYPES
**Definition** H5Fpublic.h:163
[H5FD_MEM_OHDR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5a4337f7056fb57717e82fa1081f496d75)
@ H5FD_MEM_OHDR
**Definition** H5Fpublic.h:161
[H5FD_MEM_SUPER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a51e588cd954ea2388816bd0818850eb5ae536846ac8d6ecd1a2a8479409df1bce)
@ H5FD_MEM_SUPER
**Definition** H5Fpublic.h:156
If an allocation request cannot be satisfied from the free list then one of two things happen. If the driver defines an allocation callback then it is used to allocate space; otherwise new memory is allocated from the end of the format address space by incrementing the end-of-address marker. 
`static haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) alloc (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, H5MF_type_t type, hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size)` | The file argument is the file from which space is to be allocated, type is the type of memory being requested (from the list above) without being mapped according to the freelist map and size is the number of bytes being requested. The library is allowed to allocate large chunks of storage and manage them in a layer above the file driver (although the current library doesn't do that). The allocation function should return a format address for the first byte allocated. The allocated region extends from that address for size bytes. If the request cannot be honored then the undefined address value is returned ([HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c)). The first call to this function for a file which has never had memory allocated must return a format address of zero or [HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c) since this is how the library allocates space for the userblock and/or superblock.   
---|---  
##  Freeing Format Regions
When the library is finished using a certain region of the format address space it will return the space to the free list according to the type of memory being freed and the free list map described above. If the free list has been disabled for a particular memory usage type (according to the free list map) and the driver defines a free callback then it will be invoked. The free callback is also invoked for all entries on the free list when the file is closed.
`static herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) free (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, H5MF_type_t type, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size)` | The file argument is the file for which space is being freed; type is the type of object being freed (from the list above) without being mapped according to the freelist map; addr is the first format address to free; and size is the size in bytes of the region being freed. The region being freed may refer to just part of the region originally allocated and/or may cross allocation boundaries provided all regions being freed have the same usage type. However, the library will never attempt to free regions which have already been freed or which have never been allocated.   
---|---  
A driver may choose to not define the free function, in which case format addresses will be leaked. This isn't normally a huge problem since the library contains a simple free list of its own and freeing parts of the format address space is not a common occurrence.
##  Querying the Address Range
Each file driver must have some mechanism for setting and querying the end of address, or EOA, marker. The EOA marker is the first format address after the last format address ever allocated. If the last part of the allocated address range is freed then the driver may optionally decrease the eoa marker. 
`static haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) get_eoa (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file)` | This function returns the current value of the EOA marker for the specified file.   
---|---  
Example: The sec2 driver just returns the current eoa marker value which is cached in the file structure: 
static [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f)
H5FD_sec2_get_eoa([H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *_file)
{
H5FD_sec2_t *file = (H5FD_sec2_t*)_file;
return file->eoa;
}
The eoa marker is initially zero when a file is opened and the library may set it to some other value shortly after the file is opened (after the superblock is read and the saved eoa marker is determined) or when allocating additional memory in the absence of an alloc callback (described above).
Example: The sec2 driver simply caches the eoa marker in the file structure and does not extend the underlying Unix file. When the file is flushed or closed then the Unix file size is extended to match the eoa marker. 
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
H5FD_sec2_set_eoa([H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *_file, [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr)
{
H5FD_sec2_t *file = (H5FD_sec2_t*)_file;
file->eoa = addr;
return 0;
}
#  Data Functions
These functions operate on data, transferring a region of the format address space between memory and files.
##  Contiguous I/O Functions
A driver must specify two functions to transfer data from the library to the file and vice versa. 
`static herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) read (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, H5FD_mem_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) type, hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size, void *buf)` | The read function reads data from file beginning at address addr and continuing for size bytes into the buffer buf supplied by the caller.   
---|---  
`static herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) write (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, H5FD_mem_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) type, hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size, const void *buf)` | The write function transfers data in the opposite direction.   
  * Both functions take a data transfer property list dxpl which indicates the fine points of how the data is to be transferred and which comes directly from the [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") or [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") function. 
  * Both functions receive type of data being written, which may allow a driver to tune it's behavior for different kinds of data. 
  * Both functions should return a negative value if they fail to transfer the requested data, or non-negative if they succeed. The library will never attempt to read from unallocated regions of the format address space.


Example: The sec2 driver just makes system calls. It tries not to call lseek if the current operation is the same as the previous operation and the file position is correct. It also fills the output buffer with zeros when reading between the current EOF and EOA markers and restarts system calls which were interrupted. 
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
H5FD_sec2_read([H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *_file, [H5FD_mem_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) type/*unused*/, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id/*unused*/,
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size, void *buf/*out*/)
{
H5FD_sec2_t *file = (H5FD_sec2_t*)_file;
ssize_t nbytes;
assert(file && file->pub.cls);
assert(buf);
/* Check for overflow conditions */
if (REGION_OVERFLOW(addr, size)) return -1;
if (addr+size>file->eoa) return -1;
/* Seek to the correct location */
if ((addr!=file->pos || [OP_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628ac02b22f7017838ad0c1304b26a8cb2d2)!=file->op) &&
file_seek(file->fd, (file_offset_t)addr, SEEK_SET)<0) {
file->pos = [HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c);
file->op = [OP_UNKNOWN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628a3070ec4253e2531351d7aeb586069a54);
return -1;
}
/*
* Read data, being careful of interrupted system calls, partial results,
* and the end of the file.
*/
while (size>0) {
do nbytes = read(file->fd, buf, size);
while (-1==nbytes && EINTR==errno);
if (-1==nbytes) {
/* error */
file->pos = [HADDR_UNDEF](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a676244a60b85ee8bbd111afd4a8fce3c);
file->op = [OP_UNKNOWN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628a3070ec4253e2531351d7aeb586069a54);
return -1;
}
if (0==nbytes) {
/* end of file but not end of format address space */
memset(buf, 0, size);
size = 0;
}
assert(nbytes>=0);
assert(([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c))nbytes<=size);
size -= ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c))nbytes;
addr += ([haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f))nbytes;
buf = (char*)buf + nbytes;
}
/* Update current position */
file->pos = addr;
file->op = [OP_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628ac02b22f7017838ad0c1304b26a8cb2d2);
return 0;
}
[OP_READ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628ac02b22f7017838ad0c1304b26a8cb2d2)
@ OP_READ
**Definition** H5FDprivate.h:53
Example: The sec2 write callback is similar except it updates the file EOF marker when extending the file.
##  Flushing Cached Data
Some drivers may desire to cache data in memory in order to make larger I/O requests to the underlying file and thus improving bandwidth. Such drivers should register a cache flushing function so that the library can insure that data has been flushed out of the drivers in response to the application calling [H5Fflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124 "Flushes all buffers associated with a file to storage."). 
`static herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) flush (H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file)` | Flush all data for file file to storage.   
---|---  
Example: The sec2 driver doesn't cache any data but it also doesn't extend the Unix file as aggressively as it should. Therefore, when finalizing a file it should write a zero to the last byte of the allocated region so that when reopening the file later the EOF marker will be at least as large as the EOA marker saved in the superblock (otherwise HDF5 will refuse to open the file, claiming that the data appears to be truncated). 
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
H5FD_sec2_flush([H5FD_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *_file)
{
H5FD_sec2_t *file = (H5FD_sec2_t*)_file;
if (file->eoa>file->eof) {
if (-1==file_seek(file->fd, file->eoa-1, SEEK_SET)) return -1;
if (write(file->fd, "", 1)!=1) return -1;
file->eof = file->eoa;
file->pos = file->eoa;
file->op = [OP_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628a17a60143bf77a229ec511cfe1465723e);
}
return 0;
}
[OP_WRITE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dprivate_8h.html#ad04fbad2016010f126b7e16230011628a17a60143bf77a229ec511cfe1465723e)
@ OP_WRITE
**Definition** H5FDprivate.h:54
#  Optimization Functions
The library is capable of performing several generic optimizations on I/O, but these types of optimizations may not be appropriate for a given VFL driver.
Each driver may provide a query function to allow the library to query whether to enable these optimizations. If a driver lacks a query function, the library will disable all types of optimizations which can be queried.
`static herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) query (const H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, unsigned long *flags)` | This function is called by the library to query which optimizations to enable for I/O to this driver.   
---|---  
These are the flags which are currently defined: 
`H5FD_FEAT_AGGREGATE_METADATA (0x00000001)` | Defining the H5FD_FEAT_AGGREGATE_METADATA for a VFL driver means that the library will attempt to allocate a larger block for metadata and then sub-allocate each metadata request from that larger block.   
---|---  
`H5FD_FEAT_ACCUMULATE_METADATA (0x00000002)` | Defining the H5FD_FEAT_ACCUMULATE_METADATA for a VFL driver means that the library will attempt to cache metadata as it is written to the file and build up a larger block of metadata to eventually pass to the VFL 'write' routine.   
`H5FD_FEAT_DATA_SIEVE (0x00000004)` | Defining the H5FD_FEAT_DATA_SIEVE for a VFL driver means that the library will attempt to cache raw data as it is read from/written to a file in a "data sieve" buffer.   
See Rajeev Thakur's papers: <https://web.cels.anl.gov/~thakur/papers/mpio-impl.pdf> <https://web.cels.anl.gov/~thakur/papers/romio-coll.pdf>
#  Registration of a Driver
Before a driver can be used the HDF5 library needs to be told of its existence. This is done by registering the driver, which results in a driver identification number. Instead of passing many arguments to the registration function, the driver information is entered into a structure and the address of the structure is passed to the registration function where it is copied. This allows the HDF5 API to be extended while providing backward compatibility at the source level.
`hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5FDregister (H5FD_class_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__class__t.html) *cls)` | The driver described by struct cls is registered with the library and an ID number for the driver is returned.   
---|---  
The [H5FD_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__class__t.html) type is a struct with the following fields: 
`const char *name` | A pointer to a constant, null-terminated driver name to be used for debugging purposes.   
---|---  
`size_t fapl_size` | The size in bytes of the file access mode structure or zero if the driver supplies a copy function or doesn't define the structure.   
`void *(*fapl_copy)(const void *fapl)` | An optional function which copies a driver-defined file access mode structure. This field takes precedence over fm_size when both are defined.   
`void (*fapl_free)(void *fapl)` | An optional function to free the driver-defined file access mode structure. If null, then the library calls the C free function to free the structure.   
`size_t dxpl_size` | The size in bytes of the data transfer mode structure or zero if the driver supplies a copy function or doesn't define the structure.   
`void *(*dxpl_copy)(const void *dxpl)` | An optional function which copies a driver-defined data transfer mode structure. This field takes precedence over xm_size when both are defined.   
`void (*dxpl_free)(void *dxpl)` | An optional function to free the driver-defined data transfer mode structure. If null, then the library calls the C free function to free the structure.   
`H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *(*open)(const char *name, unsigned flags, hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) maxaddr)` | The function which opens or creates a new file.   
`herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*close)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file)` | The function which ends access to a file.   
`int (*cmp)(const H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *f1, const H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *f2)` | An optional function to determine whether two open files have the same key. If this function is not present then the library assumes that two files will never be the same.   
`int (*query)(const H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *f, unsigned long *flags)` | An optional function to determine which library optimizations a driver can support.   
`haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) (*alloc)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, H5FD_mem_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) type, hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size)` | An optional function to allocate space in the file.   
`herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*free)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, H5FD_mem_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) type, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size)` | An optional function to free space in the file.   
`haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) (*get_eoa)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file)` | A function to query how much of the format address space has been allocated.   
`herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*set_eoa)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f))` | A function to set the end of address space.   
`haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) (*get_eof)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file)` | A function to return the current end-of-file marker value.   
`herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*read)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, H5FD_mem_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) type, hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size, void *buffer)` | A function to read data from a file.   
`herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*write)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file, H5FD_mem_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) type, hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl, haddr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) addr, hsize_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) size, const void *buffer)` | A function to write data to a file.   
`herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) (*flush)(H5FD_t[](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__t.html) *file)` | A function which flushes cached data to the file.   
`H5FD_mem_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html#a02887a6f018be1a0ce7358522095578b) fl_map[H5FD_MEM_NTYPES]` | An array which maps a file allocation request type to a free list.   
Example: The sec2 driver would be registered as: 
static const [H5FD_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__class__t.html) H5FD_sec2_g = {
"sec2", /*name */
MAXADDR, /*maxaddr */
NULL, /*sb_size */
NULL, /*sb_encode */
NULL, /*sb_decode */
0, /*fapl_size */
NULL, /*fapl_get */
NULL, /*fapl_copy */
NULL, /*fapl_free */
0, /*dxpl_size */
NULL, /*dxpl_copy */
NULL, /*dxpl_free */
H5FD_sec2_open, /*open */
H5FD_sec2_close, /*close */
H5FD_sec2_cmp, /*cmp */
H5FD_sec2_query, /*query */
NULL, /*alloc */
NULL, /*free */
H5FD_sec2_get_eoa, /*get_eoa */
H5FD_sec2_set_eoa, /*set_eoa */
H5FD_sec2_get_eof, /*get_eof */
H5FD_sec2_read, /*read */
H5FD_sec2_write, /*write */
H5FD_sec2_flush, /*flush */
[H5FD_FLMAP_SINGLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a4c63846487a43d7a3d589ac2852c9078), /*fl_map */
};
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
H5FD_sec2_init(void)
{
if (!H5FD_SEC2_g) {
H5FD_SEC2_g = [H5FDregister](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a576ba6b6ebc34ae422bf40700a4c372d)(&H5FD_sec2_g);
}
return H5FD_SEC2_g;
}
[H5FD_FLMAP_SINGLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a4c63846487a43d7a3d589ac2852c9078)
#define H5FD_FLMAP_SINGLE
**Definition** H5FDdevelop.h:148
[H5FDregister](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a576ba6b6ebc34ae422bf40700a4c372d)
hid_t H5FDregister(const H5FD_class_t *cls)
A driver can be removed from the library by unregistering it 
`herr_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5Dunregister (hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) driver)` | Where driver is the ID number returned when the driver was registered.   
---|---  
Unregistering a driver makes it unusable for creating new file access or data transfer property lists but doesn't affect any property lists or files that already use that driver.
##  Programming Note for C++ Developers Using C Functions
If a C routine that takes a function pointer as an argument is called from within C++ code, the C routine should be returned from normally.
Examples of this kind of routine include callbacks such as [H5Pset_elink_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga8850042eed51777866d7bd0d050cfdc2 "Sets the external link traversal callback function in a link access property list.") and [H5Pset_type_conv_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga10a80b29444d933da1aa2003f46cf003 "Sets user-defined datatype conversion callback function.") and functions such as [H5Tconvert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a "Converts data from one specified datatype to another.") and [H5Ewalk2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html#ga4ecc0f6a1ea5bb821373a5a7b8070655 "Walks the specified error stack, calling the specified function.").
Exiting the routine in its normal fashion allows the HDF5 C Library to clean up its work properly. In other words, if the C++ application jumps out of the routine back to the C++ “catch” statement, the library is not given the opportunity to close any temporary data structures that were set up when the routine was called. The C++ application should save some state as the routine is started so that any problem that occurs might be diagnosed.
#  Querying Driver Information
`void * H5Pget_driver_data (hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl)  
 void * H5Pget_driver_data (hid_t[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fxpl)` | This function is intended to be used by driver functions, not applications. It returns a pointer directly into the file access property list fapl which is a copy of the driver's file access mode originally provided to the H5Pset_driver function. If its argument is a data transfer property list fxpl then it returns a pointer to the driver-specific data transfer information instead.   
---|---  
#  Miscellaneous
The various private H5F_low_* functions will be replaced by public H5FD* functions so they can be called from drivers.
All private functions H5F_addr_* which operate on addresses will be renamed as public functions by removing the first underscore so they can be called by drivers.
The [haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) address data type will be passed by value throughout the library. The original intent was that this type would eventually be a union of file address types for the various drivers and may become quite large, but that was back when drivers were part of HDF5. It will become an alias for an unsigned integer type (32 or 64 bits depending on how the library was configured).
The various H5F*.c driver files will be renamed H5FD*.c and each will have a corresponding header file. All driver functions except the initializer and API will be declared static.
This documentation didn't cover optimization functions which would be useful to drivers like MPI-IO. Some drivers may be able to perform data pipeline operations more efficiently than HDF5 and need to be given a chance to override those parts of the pipeline. The pipeline would be designed to call various H5FD optimization functions at various points which return one of three values: the operation is not implemented by the driver, the operation is implemented but failed in a non-recoverable manner, the operation is implemented and succeeded.
Various parts of HDF5 check the only the top-level file driver and do something special if it is the MPI-IO driver. However, we might want to be able to put the MPI-IO driver under other drivers such as the raw part of a split driver or under a debug driver whose sole purpose is to accumulate statistics as it passes all requests through to the MPI-IO driver. Therefore we will probably need a function which takes a format address and or object type and returns the driver which would have been used at the lowest level to process the request.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


