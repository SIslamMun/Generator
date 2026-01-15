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
  * [ HDF5 Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#title1 "H2")
  * [ Attribute Function Summaries](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#title2 "H2")
  * [ Programming Model for Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#title3 "H2")
  * [ Working with Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#title4 "H2")
  * [ Special Issues](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Attributes
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 Attributes
An HDF5 attribute is a small metadata object describing the nature and/or intended usage of a primary data object. A primary data object may be a dataset, group, or committed datatype.
##  Introduction
Attributes are assumed to be very small as data objects go, so storing them as standard HDF5 datasets would be quite inefficient. HDF5 attributes are therefore managed through a special attributes interface, [Attributes (H5A)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html), which is designed to easily attach attributes to primary data objects as small datasets containing metadata information and to minimize storage requirements.
Consider, as examples of the simplest case, a set of laboratory readings taken under known temperature and pressure conditions of 18.0 degrees Celsius and 0.5 atmospheres, respectively. The temperature and pressure stored as attributes of the dataset could be described as the following name/value pairs: 
  * temp=18.0 
  * pressure=0.5


While HDF5 attributes are not standard HDF5 datasets, they have much in common: 
  * An attribute has a user-defined dataspace and the included metadata has a user-assigned datatype 
  * Metadata can be of any valid HDF5 datatype 
  * Attributes are addressed by name


But there are some very important differences: 
  * There is no provision for special storage such as compression or chunking 
  * There is no partial I/O or sub-setting capability for attribute data 
  * Attributes cannot be shared 
  * Attributes cannot have attributes 
  * Being small, an attribute is stored in the object header of the object it describes and is thus attached directly to that object


##  Attribute Function Summaries
see [Attributes (H5A)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html) reference manual
##  Programming Model for Attributes
The figure below shows the UML model for an HDF5 attribute and its associated dataspace and datatype. 
![](https://support.hdfgroup.org/documentation/hdf5/latest/UML_Attribute.jpg) The UML model for an HDF5 attribute  
---  
Creating an attribute is similar to creating a dataset. To create an attribute, the application must specify the object to which the attribute is attached, the datatype and dataspace of the attribute data, and the attribute creation property list.
The following steps are required to create and write an HDF5 attribute: 
  * Obtain the object identifier for the attribute's primary data object 
  * Define the characteristics of the attribute and specify the attribute creation property list 
    * Define the datatype 
    * Define the dataspace 
    * Specify the attribute creation property list
  * Create the attribute 
  * Write the attribute data (optional) 
  * Close the attribute (and datatype, dataspace, and attribute creation property list, if necessary) 
  * Close the primary data object (if appropriate)


The following steps are required to open and read/write an existing attribute. Since HDF5 attributes allow no partial I/O, you need specify only the attribute and the attribute's memory datatype to read it: 
  * Obtain the object identifier for the attribute's primary data object 
  * Obtain the attribute's name or index 
  * Open the attribute 
  * Get attribute dataspace and datatype (optional) 
  * Specify the attribute's memory type 
  * Read and/or write the attribute data 
  * Close the attribute 
  * Close the primary data object (if appropriate)


##  Working with Attributes
###  The Structure of an Attribute
An attribute has two parts: name and value(s).
HDF5 attributes are sometimes discussed as name/value pairs in the form name=value.
An attribute's name is a null-terminated ASCII or UTF-8 character string. Each attribute attached to an object has a unique name.
The value portion of the attribute contains one or more data elements of the same datatype.
HDF5 attributes have all the characteristics of HDF5 datasets except that there is no partial I/O capability. In other words, attributes can be written and read only in full with no sub-setting.
###  Creating, Writing, and Reading Attributes
If attributes are used in an HDF5 file, these functions will be employed: [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24), [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c), and [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c). [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24) and [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c) are used together to place the attribute in the file. If an attribute is to be used and is not currently in memory, [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c) generally comes into play usually in concert with one each of the H5Aget_* and H5Aopen_* functions.
To create an attribute, call H5Acreate: 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) create_plist,
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) access_plist)
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24)
#define H5Acreate
**Definition** H5version.h:1100
loc_id identifies the object (dataset, group, or committed datatype) to which the attribute is to be attached. name, type_id, space_id, and create_plist convey, respectively, the attribute's name, datatype, dataspace, and attribute creation property list. The attribute's name must be locally unique: it must be unique within the context of the object to which it is attached.
[H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24) creates the attribute in memory. The attribute does not exist in the file until [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c) writes it there.
To write or read an attribute, call H5Awrite or H5Aread, respectively: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, const void *buf)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) mem_type_id, void *buf)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c)
herr_t H5Aread(hid_t attr_id, hid_t type_id, void *buf)
Reads the value of an attribute.
[H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c)
herr_t H5Awrite(hid_t attr_id, hid_t type_id, const void *buf)
Writes data to an attribute.
attr_id identifies the attribute while mem_type_id identifies the in-memory datatype of the attribute data.
[H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c) writes the attribute data from the buffer buf to the file. [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c) reads attribute data from the file into buf.
The HDF5 Library converts the metadata between the in-memory datatype, mem_type_id, and the in-file datatype, defined when the attribute was created, without user intervention.
###  Accessing Attributes by Name or Index
Attributes can be accessed by name or index value. The use of an index value makes it possible to iterate through all of the attributes associated with a given object.
To access an attribute by its name, use the [H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10) function. [H5Aopen_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gadb49a0b5b9798d2e944d877adba8ae10) returns an attribute identifier that can then be used by any function that must access an attribute such as [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c). Use the function [H5Aget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga05e195aabab8c623b1c52009aeb99674) to determine an attribute's name.
To access an attribute by its index value, use the [H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d) function. To determine an attribute index value when it is not already known, use the H5Oget_info function. [H5Aopen_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab1451cdff4f77dcf9feaee83c8179b2d) is generally used in the course of opening several attributes for later access. Use [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) if the intent is to perform the same operation on every attribute attached to an object.
###  Obtaining Information Regarding an Object's Attributes
In the course of working with HDF5 attributes, one may need to obtain any of several pieces of information: 
  * An attribute name 
  * The dataspace of an attribute 
  * The datatype of an attribute 
  * The number of attributes attached to an object


To obtain an attribute's name, call H5Aget_name with an attribute identifier, attr_id: 
ssize_t [H5Aget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga05e195aabab8c623b1c52009aeb99674) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id, size_t buf_size, char *buf)
[H5Aget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga05e195aabab8c623b1c52009aeb99674)
ssize_t H5Aget_name(hid_t attr_id, size_t buf_size, char *buf)
Gets an attribute name.
As with other attribute functions, attr_id identifies the attribute; buf_size defines the size of the buffer; and buf is the buffer to which the attribute's name will be read.
If the length of the attribute name, and hence the value required for buf_size, is unknown, a first call to [H5Aget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga05e195aabab8c623b1c52009aeb99674) will return that size. If the value of buf_size used in that first call is too small, the name will simply be truncated in buf. A second [H5Aget_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga05e195aabab8c623b1c52009aeb99674) call can then be used to retrieve the name in an appropriately-sized buffer.
To determine the dataspace or datatype of an attribute, call [H5Aget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9e21e544119d03f9342530b45a71d74d) or [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44), respectively: 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Aget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9e21e544119d03f9342530b45a71d74d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id) [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id) 
[H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44)
hid_t H5Aget_type(hid_t attr_id)
Gets an attribute's datatype.
[H5Aget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9e21e544119d03f9342530b45a71d74d)
hid_t H5Aget_space(hid_t attr_id)
Gets a copy of the dataspace for an attribute.
[H5Aget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga9e21e544119d03f9342530b45a71d74d) returns the dataspace identifier for the attribute attr_id. [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44) returns the datatype identifier for the attribute attr_id.
To determine the number of attributes attached to an object, use the [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) function. The function signature is below. 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5)( [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) object_id, [H5O_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a5f76b0cdd6d68d61f11e46d4f06e50d4) *object_info ) 
[H5O_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a5f76b0cdd6d68d61f11e46d4f06e50d4)
#define H5O_info_t
**Definition** H5version.h:1575
[H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5)
#define H5Oget_info
**Definition** H5version.h:1351
The number of attributes will be returned in the object_info buffer. This is generally the preferred first step in determining attribute index values. If the call returns N, the attributes attached to the object object_id have index values of 0 through N-1.
###  Iterating across an Object's Attributes
It is sometimes useful to be able to perform the identical operation across all of the attributes attached to an object. At the simplest level, you might just want to open each attribute. At a higher level, you might wish to perform a rather complex operation on each attribute as you iterate across the set.
To iterate an operation across the attributes attached to an object, one must make a series of calls to [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, [H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3) index_type,
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9) order, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *n, [H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74) op,
void *op_data)
[H5A_operator2_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_apublic_8h.html#a28fef0ded9a6c0eb12334c0d15dc3e74)
herr_t(* H5A_operator2_t)(hid_t location_id, const char *attr_name, const H5A_info_t *ainfo, void *op_data)
**Definition** H5Apublic.h:60
[H5_iter_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a6a6ddd1504d1ed61939d46d91d9441b9)
H5_iter_order_t
**Definition** H5public.h:379
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5_index_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8203c78e10ab2d89d8bce688a31afad3)
H5_index_t
**Definition** H5public.h:402
[H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c)
#define H5Aiterate
**Definition** H5version.h:1111
[H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) successively marches across all of the attributes attached to the object specified in loc_id, performing the operation(s) specified in op_func with the data specified in op_data on each attribute.
When [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) is called, index contains the index of the attribute to be accessed in this call. When [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) returns, index will contain the index of the next attribute. If the returned index is the null pointer, then all attributes have been processed, and the iterative process is complete.
op_func is a user-defined operation that adheres to the [H5A_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a04b85ea47893cb5538a75130533900a9) prototype. This prototype and certain requirements imposed on the operator's behavior are described in the [H5Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab9dcfc543cd4282f32b8ea19e08ffa6c) entry in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
op_data is also user-defined to meet the requirements of op_func. Beyond providing a parameter with which to pass this data, HDF5 provides no tools for its management and imposes no restrictions.
###  Deleting an Attribute
Once an attribute has outlived its usefulness or is no longer appropriate, it may become necessary to delete it.
To delete an attribute, call [H5Adelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gada9fa3d6db52329f1fd55662de6ff6ba)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Adelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gada9fa3d6db52329f1fd55662de6ff6ba) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *name)
[H5Adelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gada9fa3d6db52329f1fd55662de6ff6ba)
herr_t H5Adelete(hid_t loc_id, const char *attr_name)
Deletes an attribute from a specified location.
[H5Adelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gada9fa3d6db52329f1fd55662de6ff6ba) removes the attribute name from the group, dataset, or committed datatype specified in loc_id.
[H5Adelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gada9fa3d6db52329f1fd55662de6ff6ba) must not be called if there are any open attribute identifiers on the object loc_id. Such a call can cause the internal attribute indexes to change; future writes to an open attribute would then produce unintended results.
###  Closing an Attribute
As is the case with all HDF5 objects, once access to an attribute it is no longer needed, that attribute must be closed. It is best practice to close it as soon as practicable; it is mandatory that it be closed prior to the H5close call closing the HDF5 Library.
To close an attribute, call [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda)
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attr_id)
[H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda)
herr_t H5Aclose(hid_t attr_id)
Closes the specified attribute.
[H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda) closes the specified attribute by terminating access to its identifier, attr_id.
##  Special Issues
Some special issues for attributes are discussed below.
#### Large Numbers of Attributes Stored in Dense Attribute Storage
The dense attribute storage scheme was added in version 1.8 so that datasets, groups, and committed datatypes that have large numbers of attributes could be processed more quickly.
Attributes start out being stored in an object's header. This is known as compact storage. For more information, see "Storage Strategies."
As the number of attributes grows, attribute-related performance slows. To improve performance, dense attribute storage can be initiated with the H5Pset_attr_phase_change function. See the HDF5 Reference Manual for more information.
When dense attribute storage is enabled, a threshold is defined for the number of attributes kept in compact storage. When the number is exceeded, the library moves all of the attributes into dense storage at another location. The library handles the movement of attributes and the pointers between the locations automatically. If some of the attributes are deleted so that the number falls below the threshold, then the attributes are moved back to compact storage by the library.
The improvements in performance from using dense attribute storage are the result of holding attributes in a heap and indexing the heap with a B-tree.
Note that there are some disadvantages to using dense attribute storage. One is that this is a new feature. Datasets, groups, and committed datatypes that use dense storage cannot be read by applications built with earlier versions of the library. Another disadvantage is that attributes in dense storage cannot be compressed.
#### Large Attributes Stored in Dense Attribute Storage
We generally consider the maximum size of an attribute to be 64K bytes. The library has two ways of storing attributes larger than 64K bytes: in dense attribute storage or in a separate dataset. Using dense attribute storage is described in this section, and storing in a separate dataset is described in the next section.
To use dense attribute storage to store large attributes, set the number of attributes that will be stored in compact storage to 0 with the H5Pset_attr_phase_change function. This will force all attributes to be put into dense attribute storage and will avoid the 64KB size limitation for a single attribute in compact attribute storage.
The example code below illustrates how to create a large attribute that will be kept in dense storage.
Create   
---  
14 { 15 __label__ fail_acpl, fail_attr, fail_file; 16 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, acpl, fspace, attr; 17 18 unsigned mode = [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a); 19 char file_name[] = "f1.h5"; 20 // attribute names can be arbitrary Unicode strings 21 char attr_name[] = "Χαρακτηριστικό"; 22 23 if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(file_name, mode, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 24 ret_val = EXIT_FAILURE; 25 goto fail_file; 26 } 27 if ((acpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_ATTRIBUTE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa0102211c679e031e2e9831b66c48a12))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 28 ret_val = EXIT_FAILURE; 29 goto fail_acpl; 30 } 31 // use UTF-8 encoding for the attribute name 32 if ([H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc)(acpl, [H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)) < 0) { 33 ret_val = EXIT_FAILURE; 34 goto fail_fspace; 35 } 36 // create a scalar (singleton) attribute 37 if ((fspace = [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)([H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 38 ret_val = EXIT_FAILURE; 39 goto fail_fspace; 40 } 41 // create an attribute on the root group 42 if ((attr = [H5Acreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308)(file, attr_name, [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b), fspace, acpl, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) == 43 [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) { 44 ret_val = EXIT_FAILURE; 45 goto fail_attr; 46 } 47 48 [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda)(attr); 49fail_attr: 50 [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(fspace); 51fail_fspace: 52 [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(acpl); 53fail_acpl: 54 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file); 55fail_file:; 56 }  
#### Large Attributes Stored in a Separate Dataset
In addition to dense attribute storage (see above), a large attribute can be stored in a separate dataset. In the figure below, DatasetA holds an attribute that is too large for the object header in Dataset1. By putting a pointer to DatasetA as an attribute in Dataset1, the attribute becomes available to those working with Dataset1. This way of handling large attributes can be used in situations where backward compatibility is important and where compression is important. Applications built with versions before 1.8.x can read large attributes stored in separate datasets. Datasets can be compressed while attributes cannot. 
![](https://support.hdfgroup.org/documentation/hdf5/latest/Shared_Attribute.jpg) A large or shared HDF5 attribute and its associated dataset(s)  
---  
Note: In the figure above, DatasetA is an attribute of Dataset1 that is too large to store in Dataset1's header. DatasetA is associated with Dataset1 by means of an object reference pointer attached as an attribute to Dataset1. The attribute in DatasetA can be shared among multiple datasets by means of additional object reference pointers attached to additional datasets.
#### Shared Attributes
Attributes written and managed through the [Attributes (H5A)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html) interface cannot be shared. If shared attributes are required, they must be handled in the manner described above for large attributes and illustrated in the figure above.
#### Attribute Names
While any ASCII or UTF-8 character may be used in the name given to an attribute, it is usually wise to avoid the following kinds of characters: 
  * Commonly used separators or delimiters such as slash, backslash, colon, and semi-colon (\, /, :, ;) 
  * Escape characters 
  * Wild cards such as asterisk and question mark (*, ?) NULL can be used within a name, but HDF5 names are terminated with a NULL: whatever comes after the NULL will be ignored by HDF5.


The use of ASCII or UTF-8 characters is determined by the character encoding property. See [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.") in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
#### No Special I/O or Storage
HDF5 attributes have all the characteristics of HDF5 datasets except the following: 
  * Attributes are written and read only in full: there is no provision for partial I/O or sub-setting 
  * No special storage capability is provided for attributes: there is no compression or chunking, and attributes are not extendable


Previous Chapter [HDF5 Dataspaces and Partial I/O](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#sec_dataspace) - Next Chapter [HDF5 Error Handling](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#sec_error)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


