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
  * [ The HDF5 Library and Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5__u_g.html#title1 "H2")
  * [ The HDF5 Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5__u_g.html#title2 "H2")
  * [ The Data Transfer Pipeline](https://support.hdfgroup.org/documentation/hdf5/latest/_h5__u_g.html#title3 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Library and Programming Model
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  The HDF5 Library and Programming Model
##  Introduction
The HDF5 library implements the HDF5 abstract data model and storage model. These models were described in the preceding chapter.
Two major objectives of the HDF5 products are to provide tools that can be used on as many computational platforms as possible (portability), and to provide a reasonably object-oriented data model and programming interface.
To be as portable as possible, the HDF5 library is implemented in portable C. C is not an object-oriented language, but the library uses several mechanisms and conventions to implement an object model.
One mechanism the HDF5 library uses is to implement the objects as data structures. To refer to an object, the HDF5 library implements its own pointers. These pointers are called identifiers. An identifier is then used to invoke operations on a specific instance of an object. For example, when a group is opened, the API returns a group identifier. This identifier is a reference to that specific group and will be used to invoke future operations on that group. The identifier is valid only within the context it is created and remains valid until it is closed or the file is closed. This mechanism is essentially the same as the mechanism that C++ or other object-oriented languages use to refer to objects except that the syntax is C.
Similarly, object-oriented languages collect all the methods for an object in a single name space. An example is the methods of a C++ class. The C language does not have any such mechanism, but the HDF5 library simulates this through its API naming convention. API function names begin with a common prefix that is related to the class of objects that the function operates on. The table below lists the HDF5 objects and the standard prefixes used by the corresponding HDF5 APIs. For example, functions that operate on datatype objects all have names beginning with H5T.
Access flags and modes Prefix  | Operates on   
---|---  
[Attributes (H5A)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html) | Attributes   
[Datasets (H5D)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html) | Datasets   
[Error Handling (H5E)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_e.html) | Error reports   
[Files (H5F)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html) | Files   
[Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) | Groups   
[Identifiers (H5I)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i.html) | Identifiers   
[Links (H5L)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html) | Links   
[Objects (H5O)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html) | Objects   
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) | Property lists   
[References (H5R)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html) | References   
[Dataspaces (H5S)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html) | Dataspaces   
[Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html) | Datatypes   
[Filters (H5Z)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html) | Filters   
##  The HDF5 Programming Model
In this section we introduce the HDF5 programming model by means of a series of short code samples. These samples illustrate a broad selection of common HDF5 tasks. More details are provided in the following chapters and in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
###  Creating an HDF5 File
Before an HDF5 file can be used or referred to in any manner, it must be explicitly created or opened. When the need for access to a file ends, the file must be closed. The example below provides a C code fragment illustrating these steps. In this example, the values for the file creation property list and the file access property list are set to the defaults [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f).
_Creating and closing an HDF5 file_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file; // declare file identifier
// Create a new file using H5F_ACC_TRUNC to truncate and overwrite
// any file of the same name, default file creation properties, and
// default file access properties. Then close the file.
file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(FILE, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
status = [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
Note: If there is a possibility that a file of the declared name already exists and you wish to open a new file regardless of that possibility, the flag [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) will cause the operation to overwrite the previous file. If the operation should fail in such a circumstance, use the flag [H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90) instead.
###  Creating and Initializing a Dataset
The essential objects within a dataset are datatype and dataspace. These are independent objects and are created separately from any dataset to which they may be attached. Hence, creating a dataset requires, at a minimum, the following steps: 
  1. Create and initialize a dataspace for the dataset 
  2. Define a datatype for the dataset 
  3. Create and initialize the dataset


The code in the example below illustrates the execution of these steps.
_Create a dataset_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dataset, datatype, dataspace; // declare identifiers
// Create a dataspace: Describe the size of the array and
// create the dataspace for a fixed-size dataset.
dimsf[0] = NX;
dimsf[1] = NY;
dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dimsf, NULL);
// Define a datatype for the data in the dataset.
// We will store little endian integers.
datatype = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
status = [H5Tset_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab1aab76b1214a819281f2156c6d45d71)(datatype, [H5T_ORDER_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0ae5668f73f6c28feddb7af175ac53012d));
// Create a new dataset within the file using the defined
// dataspace and datatype and default dataset creation
// properties.
// NOTE: H5T_NATIVE_INT can be used as the datatype if
// conversion to little endian is not needed.
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, DATASETNAME, datatype, dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5T_ORDER_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0ae5668f73f6c28feddb7af175ac53012d)
@ H5T_ORDER_LE
**Definition** H5Tpublic.h:55
[H5Tset_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab1aab76b1214a819281f2156c6d45d71)
herr_t H5Tset_order(hid_t type_id, H5T_order_t order)
Sets the byte order of a datatype.
[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
#define H5Dcreate
**Definition** H5version.h:1124
[H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
hid_t H5Screate_simple(int rank, const hsize_t dims[], const hsize_t maxdims[])
Creates a new simple dataspace and opens it for access.
[H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)
hid_t H5Tcopy(hid_t type_id)
Copies an existing datatype.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
[RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5)
#define RANK
**Definition** h5import.h:343
###  Closing an Object
An application should close an object such as a datatype, dataspace, or dataset once the object is no longer needed. Since each is an independent object, each must be released (or closed) separately. This action is frequently referred to as releasing the object's identifier. The code in the example below closes the datatype, dataspace, and dataset that were created in the preceding section.
_Close an object_
[H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0)(datatype);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dataset);
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(dataspace);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)
herr_t H5Dclose(hid_t dset_id)
Closes the specified dataset.
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)
herr_t H5Sclose(hid_t space_id)
Releases and terminates access to a dataspace.
[H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0)
herr_t H5Tclose(hid_t type_id)
Releases a datatype.
There is a long list of HDF5 library items that return a unique identifier when the item is created or opened. Each time that one of these items is opened, a unique identifier is returned. Closing a file does not mean that the groups, datasets, or other open items are also closed. Each opened item must be closed separately.
For more information,  

See also
     [Using Identifiers](https://support.hdfgroup.org/documentation/hdf5/latest/_using_identifiers.html) in the HDF5 Application Developer's Guide under General Topics in HDF5.
#### How Closing a File Effects Other Open Structural Elements
Every structural element in an HDF5 file can be opened, and these elements can be opened more than once. Elements range in size from the entire file down to attributes. When an element is opened, the HDF5 library returns a unique identifier to the application. Every element that is opened must be closed. If an element was opened more than once, each identifier that was returned to the application must be closed. For example, if a dataset was opened twice, both dataset identifiers must be released (closed) before the dataset can be considered closed. Suppose an application has opened a file, a group in the file, and two datasets in the group. In order for the file to be totally closed, the file, group, and datasets must each be closed. Closing the file before the group or the datasets will not affect the state of the group or datasets: the group and datasets will still be open.
There are several exceptions to the above general rule. One is when the [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") function is used. [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") causes a general shutdown of the library: all data is written to disk, all identifiers are closed, and all memory used by the library is cleaned up. Another exception occurs on parallel processing systems. Suppose on a parallel system an application has opened a file, a group in the file, and two datasets in the group. If the application uses the [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file.") function to close the file, the call will fail with an error. The open group and datasets must be closed before the file can be closed. A third exception is when the file access property list includes the property [H5F_CLOSE_STRONG](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#aa85fa00d037d2b0401cf72edf9a6475fae6af53249bfe320745828497f28b6390). This property closes any open elements when the file is closed with [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124 "Terminates access to an HDF5 file."). For more information, see the [H5Pset_fclose_degree](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c "Sets the file close degree.") function in the HDF5 Reference Manual.
###  Writing or Reading a Dataset to or from a File
Having created the dataset, the actual data can be written with a call to [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset."). See the example below.
_Writing a dataset_
// Write the data to the dataset using default transfer properties.
status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data);
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484)
#define H5S_ALL
**Definition** H5Spublic.h:33
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
Note that the third and fourth [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") parameters in the above example describe the dataspaces in memory and in the file, respectively. For now, these are both set to [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) which indicates that the entire dataset is to be written. The selection of partial datasets and the use of differing dataspaces in memory and in storage will be discussed later in this chapter and in more detail elsewhere in this guide.
Reading the dataset from storage is similar to writing the dataset to storage. To read an entire dataset, substitute [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") for [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") in the above example.
###  Reading and Writing a Portion of a Dataset
The previous section described writing or reading an entire dataset. HDF5 also supports access to portions of a dataset. These parts of datasets are known as selections.
The simplest type of selection is a simple hyperslab. This is an n-dimensional rectangular sub-set of a dataset where n is equal to the dataset's rank. Other available selections include a more complex hyperslab with user-defined stride and block size, a list of independent points, or the union of any of these.
The figure below shows several sample selections.
Dataset selections ![](https://support.hdfgroup.org/documentation/hdf5/latest/Pmodel_fig5_a.gif)  
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Pmodel_fig5_b.gif)  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Pmodel_fig5_c.gif)  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Pmodel_fig5_d.gif)   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Pmodel_fig5_e.gif)  
Note: In the figure above, selections can take the form of a simple hyperslab, a hyperslab with user-defined stride and block, a selection of points, or a union of any of these forms.
Selections and hyperslabs are portions of a dataset. As described above, a simple hyperslab is a rectangular array of data elements with the same rank as the dataset's dataspace. Thus, a simple hyperslab is a logically contiguous collection of points within the dataset.
The more general case of a hyperslab can also be a regular pattern of points or blocks within the dataspace. Four parameters are required to describe a general hyperslab: the starting coordinates, the block size, the stride or space between blocks, and the number of blocks. These parameters are each expressed as a one-dimensional array with length equal to the rank of the dataspace and are described in the table below.
Parameter  | Definition   
---|---  
start  | The coordinates of the starting location of the hyperslab in the dataset's dataspace.   
block  | The size of each block to be selected from the dataspace. If the block parameter is set to NULL, the block size defaults to a single element in each dimension, as if the block array was set to all 1s (all ones). This will result in the selection of a uniformly spaced set of count points starting at start and on the interval defined by stride.   
stride  | The number of elements separating the starting point of each element or block to be selected. If the stride parameter is set to NULL, the stride size defaults to 1 (one) in each dimension and no elements are skipped.   
count  | The number of elements or blocks to select along each dimension.   
#### Reading Data into a Differently Shaped Memory Block
For maximum flexibility in user applications, a selection in storage can be mapped into a differently-shaped selection in memory. All that is required is that the two selections contain the same number of data elements. In this example, we will first define the selection to be read from the dataset in storage, and then we will define the selection as it will appear in application memory.
Suppose we want to read a 3 x 4 hyperslab from a two-dimensional dataset in a file beginning at the dataset element <1,2>. The first task is to create the dataspace that describes the overall rank and dimensions of the dataset in the file and to specify the position and size of the in-file hyperslab that we are extracting from that dataset. See the code below.
_Define the selection to be read from storage_
// Define dataset dataspace in file.
dataspace = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)(dataset); // dataspace identifier
rank = [H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gae5282a81692b80b5b19dd12d05b9b28e)(dataspace);
status_n = [H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3)(dataspace, dims_out, NULL);
// Define hyperslab in the dataset.
offset[0] = 1;
offset[1] = 2;
count[0] = 3;
count[1] = 4;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(dataspace, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, NULL, count, NULL);
[H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550)
@ H5S_SELECT_SET
**Definition** H5Spublic.h:100
[H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)
hid_t H5Dget_space(hid_t dset_id)
Returns an identifier for a copy of the dataspace for a dataset.
[H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)
herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op, const hsize_t start[], const hsize_t stride[], const hsize_t count[], const hsize_t block[])
Selects a hyperslab region to add to the current selected region.
[H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3)
int H5Sget_simple_extent_dims(hid_t space_id, hsize_t dims[], hsize_t maxdims[])
Retrieves dataspace dimension size and maximum size.
[H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gae5282a81692b80b5b19dd12d05b9b28e)
int H5Sget_simple_extent_ndims(hid_t space_id)
Determines the dimensionality of a dataspace.
The next task is to define a dataspace in memory. Suppose that we have in memory a three-dimensional 7 x 7 x 3 array into which we wish to read the two-dimensional 3 x 4 hyperslab described above and that we want the memory selection to begin at the element <3,0,0> and reside in the plane of the first two dimensions of the array. Since the in-memory dataspace is three-dimensional, we have to describe the in-memory selection as three-dimensional. Since we are keeping the selection in the plane of the first two dimensions of the in-memory dataset, the in-memory selection will be a 3 x 4 x 1 array defined as <3,4,1>.
Notice that we must describe two things: the dimensions of the in-memory array, and the size and position of the hyperslab that we wish to read in. The code below illustrates how this would be done.
_Define the memory dataspace and selection_
// Define memory dataspace.
dimsm[0] = 7;
dimsm[1] = 7;
dimsm[2] = 3;
memspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(RANK_OUT,dimsm,NULL);
// Define memory hyperslab.
offset_out[0] = 3;
offset_out[1] = 0;
offset_out[2] = 0;
count_out[0] = 3;
count_out[1] = 4;
count_out[2] = 1;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(memspace, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset_out, NULL, count_out, NULL);
The hyperslab defined in the code above has the following parameters: start=(3,0,0), count=(3,4,1), stride and block size are NULL.
#### Writing Data into a Differently Shaped Disk Storage Block
Now let's consider the opposite process of writing a selection from memory to a selection in a dataset in a file. Suppose that the source dataspace in memory is a 50-element, one-dimensional array called vector and that the source selection is a 48-element simple hyperslab that starts at the second element of vector. See the figure below.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Pmodel_fig2.gif) A one-dimensional array  
---  
Further suppose that we wish to write this data to the file as a series of 3 x 2-element blocks in a two-dimensional dataset, skipping one row and one column between blocks. Since the source selection contains 48 data elements and each block in the destination selection contains 6 data elements, we must define the destination selection with 8 blocks. We will write 2 blocks in the first dimension and 4 in the second. The code below shows how to achieve this objective.
_The destination selection_
// Select the hyperslab for the dataset in the file, using
// 3 x 2 blocks, a (4,3) stride, a (2,4) count, and starting
// at the position (0,1).
start[0] = 0; start[1] = 1;
stride[0] = 4; stride[1] = 3;
count[0] = 2; count[1] = 4;
block[0] = 3; block[1] = 2;
ret = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(fid, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), start, stride, count, block);
// Create dataspace for the first dataset.
mid1 = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(MSPACE1_RANK, dim1, NULL);
// Select hyperslab.
// We will use 48 elements of the vector buffer starting at the
// second element. Selected elements are 1 2 3 . . . 48
start[0] = 1;
stride[0] = 1;
count[0] = 48;
block[0] = 1;
ret = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(mid1, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), start, stride, count, block);
// Write selection from the vector buffer to the dataset in the file.
ret = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), mid1, fid, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), vector);
###  Getting Information about a Dataset
Although reading is analogous to writing, it is often first necessary to query a file to obtain information about the dataset to be read. For instance, we often need to determine the datatype associated with a dataset, or its dataspace (in other words, rank and dimensions). As illustrated in the code example below, there are several get routines for obtaining this information.
_Routines to get dataset parameters_
// Get datatype and dataspace identifiers,
// then query datatype class, order, and size, and
// then query dataspace rank and dimensions.
datatype = [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb) (dataset); // datatype identifier
class = H5Tget_class (datatype);
if (class == [H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037))
printf("Dataset has INTEGER type \n");
order = [H5Tget_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaeb5bd7ec46787a4b6d33947dc73c2a5f) (datatype);
if (order == [H5T_ORDER_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0ae5668f73f6c28feddb7af175ac53012d))
printf("Little endian order \n");
size = [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e) (datatype);
printf ("Size is %d \n", size);
dataspace = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a) (dataset); // dataspace identifier
// Find rank and retrieve current and maximum dimension sizes.
rank = [H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3) (dataspace, dims, max_dims);
[H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037)
@ H5T_INTEGER
**Definition** H5Tpublic.h:32
[H5Tget_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaeb5bd7ec46787a4b6d33947dc73c2a5f)
H5T_order_t H5Tget_order(hid_t type_id)
Returns the byte order of an atomic datatype.
[H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)
hid_t H5Dget_type(hid_t dset_id)
Returns an identifier for a copy of the datatype for a dataset.
[H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)
size_t H5Tget_size(hid_t type_id)
Returns the size of a datatype.
###  Creating and Defining Compound Datatypes
A compound datatype is a collection of one or more data elements. Each element might be an atomic type, a small array, or another compound datatype.
The provision for nested compound datatypes allows these structures to become quite complex. An HDF5 compound datatype has some similarities to a C struct or a Fortran common block. Though not originally designed with databases in mind, HDF5 compound datatypes are sometimes used in a way that is similar to a database record. Compound datatypes can become either a powerful tool or a complex and difficult-to-debug construct. Reasonable caution is advised.
To create and use a compound datatype, you need to create a datatype with class compound ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6)) and specify the total size of the data element in bytes. A compound datatype consists of zero or more uniquely named members. Members can be defined in any order but must occupy non-overlapping regions within the datum. The table below lists the properties of compound datatype members.
Parameter  | Definition   
---|---  
Index  | An index number between zero and N-1, where N is the number of members in the compound. The elements are indexed in the order of their location in the array of bytes.   
Name  | A string that must be unique within the members of the same datatype.   
Datatype  | An HDF5 datatype.   
Offset  | A fixed byte offset which defines the location of the first byte of that member in the compound datatype.   
Properties of the members of a compound datatype are defined when the member is added to the compound type. These properties cannot be modified later.
#### Defining Compound Datatypes
Compound datatypes must be built out of other datatypes. To do this, you first create an empty compound datatype and specify its total size. Members are then added to the compound datatype in any order.
Each member must have a descriptive name. This is the key used to uniquely identify the member within the compound datatype. A member name in an HDF5 datatype does not necessarily have to be the same as the name of the corresponding member in the C struct in memory although this is often the case. You also do not need to define all the members of the C struct in the HDF5 compound datatype (or vice versa).
Usually a C struct will be defined to hold a data point in memory, and the offsets of the members in memory will be the offsets of the struct members from the beginning of an instance of the struct. The library defines the macro that computes the offset of member m within a struct variable s: 
[HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s,m)
[HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)
#define HOFFSET(S, M)
**Definition** H5Tpublic.h:22
The code below shows an example in which a compound datatype is created to describe complex numbers whose type is defined by the complex_t struct.
_A compound datatype for complex numbers_
Typedef struct {
double re; //real part
double im; //imaginary part
} complex_t;
complex_t tmp; //used only to compute offsets
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) complex_id = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof tmp);
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (complex_id, "real", [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(tmp,re), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (complex_id, "imaginary", [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(tmp,im), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6)
@ H5T_COMPOUND
**Definition** H5Tpublic.h:38
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)
herr_t H5Tinsert(hid_t parent_id, const char *name, size_t offset, hid_t member_id)
Adds a new member to a compound datatype.
[H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65)
hid_t H5Tcreate(H5T_class_t type, size_t size)
Creates a new datatype.
[H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09)
#define H5T_NATIVE_DOUBLE
**Definition** H5Tpublic.h:1036
###  Creating and Writing Extendable Datasets
An extendable dataset is one whose dimensions can grow. One can define an HDF5 dataset to have certain initial dimensions with the capacity to later increase the size of any of the initial dimensions. For example, the figure below shows a 3 x 3 dataset (a) which is later extended to be a 10 x 3 dataset by adding 7 rows (b), and further extended to be a 10 x 5 dataset by adding two columns (c).
![](https://support.hdfgroup.org/documentation/hdf5/latest/Pmodel_fig3.gif) Extending a dataset  
---  
HDF5 requires the use of chunking when defining extendable datasets. [Chunking in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/hdf5_chunking.html) makes it possible to extend datasets efficiently without having to reorganize contiguous storage excessively.
To summarize, an extendable dataset requires two conditions: 
  1. Define the dataspace of the dataset as unlimited in all dimensions that might eventually be extended 
  2. Enable chunking in the dataset creation properties


For example, suppose we wish to create a dataset similar to the one shown in the figure above. We want to start with a 3 x 3 dataset, and then later we will extend it. To do this, go through the steps below.
First, declare the dataspace to have unlimited dimensions. See the code shown below. Note the use of the predefined constant [H5S_UNLIMITED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5af9ab788797b2ea9a4843857674ac18) to specify that a dimension is unlimited.
_Declaring a dataspace with unlimited dimensions_
// dataset dimensions at creation time
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dims[2] = {3, 3};
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) maxdims[2] = {[H5S_UNLIMITED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5af9ab788797b2ea9a4843857674ac18), [H5S_UNLIMITED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5af9ab788797b2ea9a4843857674ac18)};
// Create the data space with unlimited dimensions.
dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dims, maxdims);
[H5S_UNLIMITED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5af9ab788797b2ea9a4843857674ac18)
#define H5S_UNLIMITED
**Definition** H5Spublic.h:52
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
Next, set the dataset creation property list to enable chunking. See the code below.
_Enable chunking_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cparms;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) chunk_dims[2] ={2, 5};
// Modify dataset creation properties to enable chunking.
cparms = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
status = [H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)(cparms, [RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), chunk_dims);
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)
#define H5P_DATASET_CREATE
**Definition** H5Ppublic.h:60
[H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)
herr_t H5Pset_chunk(hid_t plist_id, int ndims, const hsize_t dim[])
Sets the size of the chunks used to store a chunked layout dataset.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
The next step is to create the dataset. See the code below.
_Create a dataset_
// Create a new dataset within the file using cparms creation properties.
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, DATASETNAME, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), cparms, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
Finally, when the time comes to extend the size of the dataset, invoke [H5Dextend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac4c0ff57977b1f39c1055296e39cbe91 "Extends a dataset."). Extending the dataset along the first dimension by seven rows leaves the dataset with new dimensions of <10,3>. See the code below.
_Extend the dataset by seven rows_
// Extend the dataset. Dataset becomes 10 x 3.
dims[0] = dims[0] + 7;
size[0] = dims[0];
size[1] = dims[1];
status = [H5Dextend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac4c0ff57977b1f39c1055296e39cbe91) (dataset, size);
[H5Dextend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac4c0ff57977b1f39c1055296e39cbe91)
herr_t H5Dextend(hid_t dset_id, const hsize_t size[])
Extends a dataset.
###  Creating and Working with Groups
Groups provide a mechanism for organizing meaningful and extendable sets of datasets within an HDF5 file. The [Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) API provides several routines for working with groups.
#### Creating a Group
With no datatype, dataspace, or storage layout to define, creating a group is considerably simpler than creating a dataset. For example, the following code creates a group called Data in the root group of file.
_Create a group_
// Create a group in the file.
grp = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)(file, "/Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)
#define H5Gcreate
**Definition** H5version.h:1240
A group may be created within another group by providing the absolute name of the group to the [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) function or by specifying its location. For example, to create the group Data_new in the group Data, you might use the sequence of calls shown below.
_Create a group within a group_
// Create group "Data_new" in the group "Data" by specifying
// absolute name of the group.
grp_new = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)(file, "/Data/Data_new", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// or
// Create group "Data_new" in the "Data" group.
grp_new = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)(grp, "Data_new", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
This first parameter of [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) is a location identifier. file in the first example specifies only the file. _grp_ in the second example specifies a particular group in a particular file. Note that in this instance, the group identifier _grp_ is used as the first parameter in the [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) call so that the relative name of Data_new can be used.
The third parameter of [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) optionally specifies how much file space to reserve to store the names of objects that will be created in this group. If a non-positive value is supplied, the library provides a default size.
Use [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221 "Closes the specified group.") to close the group and release the group identifier.
#### Creating a Dataset within a Group
As with groups, a dataset can be created in a particular group by specifying either its absolute name in the file or its relative name with respect to that group. The next code excerpt uses the absolute name.
_Create a dataset within a group using a relative name_
// Create the dataset "Compressed_Data" in the group Data using
// the absolute name. The dataset creation property list is
// modified to use GZIP compression with the compression
// effort set to 6. Note that compression can be used only when
// the dataset is chunked.
dims[0] = 1000;
dims[1] = 20;
cdims[0] = 20;
cdims[1] = 20;
dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dims, NULL);
plist = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
[H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)(plist, 2, cdims);
[H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d)(plist, 6);
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, "/Data/Compressed_Data", [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f),
plist, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d)
herr_t H5Pset_deflate(hid_t plist_id, unsigned level)
Sets deflate (GNU gzip) compression method and compression level.
Alternatively, you can first obtain an identifier for the group in which the dataset is to be created, and then create the dataset with a relative name.
_Create a dataset within a group using a relative name_
// Open the group.
grp = [H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)(file, "Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Create the dataset "Compressed_Data" in the "Data" group
// by providing a group identifier and a relative dataset
// name as parameters to the H5Dcreate function.
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(grp, "Compressed_Data", [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), plist, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)
#define H5Gopen
**Definition** H5version.h:1251
#### Accessing an Object in a Group
Any object in a group can be accessed by its absolute or relative name. The first code snippet below illustrates the use of the absolute name to access the dataset _Compressed_Data_ in the group _Data_ created in the examples above. The second code snippet illustrates the use of the relative name.
_Accessing a group using its relative name_
// Open the dataset "Compressed_Data" in the "Data" group.
dataset = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file, "/Data/Compressed_Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Open the group "data" in the file.
grp = [H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)(file, "Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Access the "Compressed_Data" dataset in the group.
dataset = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(grp, "Compressed_Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
###  Working with Attributes
An attribute is a small dataset that is attached to a normal dataset or group. Attributes share many of the characteristics of datasets, so the programming model for working with attributes is similar in many ways to the model for working with datasets. The primary differences are that an attribute must be attached to a dataset or a group and sub-setting operations cannot be performed on attributes.
To create an attribute belonging to a particular dataset or group, first create a dataspace for the attribute with the call to [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083 "Creates a new dataspace of a specified type."), and then create the attribute using [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24). For example, the code shown below creates an attribute called “Integer attribute” that is a member of a dataset whose identifier is dataset. The attribute identifier is attr2. [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c "Writes data to an attribute.") then sets the value of the attribute of that of the integer variable point. [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda "Closes the specified attribute.") then releases the attribute identifier.
_Create an attribute_
int point = 1; // Value of the scalar attribute
// Create scalar attribute.
aid2 = [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)([H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8));
attr2 = [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24)(dataset, "Integer attribute", [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), aid2, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Write scalar attribute.
ret = [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c)(attr2, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), &point);
// Close attribute dataspace.
ret = [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(aid2);
// Close attribute.
ret = [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda)(attr2);
[H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8)
@ H5S_SCALAR
**Definition** H5Spublic.h:90
[H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24)
#define H5Acreate
**Definition** H5version.h:1100
[H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c)
herr_t H5Awrite(hid_t attr_id, hid_t type_id, const void *buf)
Writes data to an attribute.
[H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda)
herr_t H5Aclose(hid_t attr_id)
Closes the specified attribute.
[H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)
hid_t H5Screate(H5S_class_t type)
Creates a new dataspace of a specified type.
_Read a known attribute_
// Attach to the scalar attribute using attribute name, then
// read and display its value.
attr = [H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10)(file_id, dataset_name, "Integer attribute", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
ret = [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c)(attr, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), &point_out);
printf("The value of the attribute \"Integer attribute\" is %d \n", point_out);
ret = [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda)(attr);
[H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c)
herr_t H5Aread(hid_t attr_id, hid_t type_id, void *buf)
Reads the value of an attribute.
[H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10)
hid_t H5Aopen_by_name(hid_t loc_id, const char *obj_name, const char *attr_name, hid_t aapl_id, hid_t lapl_id)
Opens an attribute for an object by object name and attribute name.
To read a scalar attribute whose name and datatype are known, first open the attribute using [H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10 "Opens an attribute for an object by object name and attribute name."), and then use [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c "Reads the value of an attribute.") to get its value. For example, the code shown below reads a scalar attribute called “Integer attribute” whose datatype is a native integer and whose parent dataset has the identifier dataset.
To read an attribute whose characteristics are not known, go through these steps. First, query the file to obtain information about the attribute such as its name, datatype, rank, and dimensions, and then read the attribute. The following code opens an attribute by its index value using [H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d "Opens the nth attribute attached to an object."), and then it reads in information about the datatype with [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c "Reads the value of an attribute.").
_Read an unknown attribute_
// Attach to the string attribute using its index, then read and
// display the value.
attr = [H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d)(file_id, dataset_name, index_type, iter_order, 2, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
atype = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d));
[H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)(atype, 4);
ret = [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c)(attr, atype, string_out);
printf("The value of the attribute with the index 2 is %s \n", string_out);
[H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d)
hid_t H5Aopen_by_idx(hid_t loc_id, const char *obj_name, H5_index_t idx_type, H5_iter_order_t order, hsize_t n, hid_t aapl_id, hid_t lapl_id)
Opens the nth attribute attached to an object.
[H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)
herr_t H5Tset_size(hid_t type_id, size_t size)
Sets size for a datatype.
[H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d)
#define H5T_C_S1
**Definition** H5Tpublic.h:646
In practice, if the characteristics of attributes are not known, the code involved in accessing and processing the attribute can be quite complex. For this reason, HDF5 includes a function called [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c). This function applies a user-supplied function to each of a set of attributes. The user-supplied function can contain the code that interprets, accesses, and processes each attribute.
##  The Data Transfer Pipeline
The HDF5 library implements data transfers between different storage locations. At the lowest levels, the HDF5 Library reads and writes blocks of bytes to and from storage using calls to the virtual file layer (VFL) drivers. In addition to this, the HDF5 library manages caches of metadata and a data I/O pipeline. The data I/O pipeline applies compression to data blocks, transforms data elements, and implements selections.
A substantial portion of the HDF5 library's work is in transferring data from one environment or media to another. This most often involves a transfer between system memory and a storage medium. Data transfers are affected by compression, encryption, machine-dependent differences in numerical representation, and other features. So, the bit-by-bit arrangement of a given dataset is often substantially different in the two environments.
Consider the representation on disk of a compressed and encrypted little-endian array as compared to the same array after it has been read from disk, decrypted, decompressed, and loaded into memory on a big-endian system. HDF5 performs all of the operations necessary to make that transition during the I/O process with many of the operations being handled by the VFL and the data transfer pipeline.
The figure below provides a simplified view of a sample data transfer with four stages. Note that the modules are used only when needed. For example, if the data is not compressed, the compression stage is omitted.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Pmodel_fig6.gif) A data transfer from storage to memory  
---  
For a given I/O request, different combinations of actions may be performed by the pipeline. The library automatically sets up the pipeline and passes data through the processing steps. For example, for a read request (from disk to memory), the library must determine which logical blocks contain the requested data elements and fetch each block into the library's cache. If the data needs to be decompressed, then the compression algorithm is applied to the block after it is read from disk. If the data is a selection, the selected elements are extracted from the data block after it is decompressed. If the data needs to be transformed (for example, byte swapped), then the data elements are transformed after decompression and selection.
While an application must sometimes set up some elements of the pipeline, use of the pipeline is normally transparent to the user program. The library determines what must be done based on the metadata for the file, the object, and the specific request. An example of when an application might be required to set up some elements in the pipeline is if the application used a custom error-checking algorithm.
In some cases, it is necessary to pass parameters to and from modules in the pipeline or among other parts of the library that are not directly called through the programming API. This is accomplished through the use of dataset transfer and data access property lists.
The VFL provides an interface whereby user applications can add custom modules to the data transfer pipeline. For example, a custom compression algorithm can be used with the HDF5 Library by linking an appropriate module into the pipeline through the VFL. This requires creating an appropriate wrapper for the compression module and registering it with the library with [H5Zregister](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_z.html#ga93145acc38c2c60d832b7a9b0123706b "Registers a new filter with the HDF5 library."). The algorithm can then be applied to a dataset with an [H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline.") call which will add the algorithm to the selected dataset's transfer property list.
Previous Chapter [The HDF5 Data Model and File Structure](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_m__u_g.html#sec_data_model) - Next Chapter [The HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#sec_file)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


