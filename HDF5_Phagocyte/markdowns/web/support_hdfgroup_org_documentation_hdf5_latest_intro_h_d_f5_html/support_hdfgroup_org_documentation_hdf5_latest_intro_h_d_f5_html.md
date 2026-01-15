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
  * [ HDF5 Description](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title0 "H1")
  * [ File Format](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title1 "H2")
  * [ Data Model](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title2 "H2")
  * [ Datatypes, Dataspaces, Properties and Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title3 "H2")
  * [ HDF5 Software](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title4 "H2")
  * [ Introduction to the HDF5 Programming Model and APIs](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title5 "H1")
  * [ Steps to create a file](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title6 "H2")
  * [ Steps to create a dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title7 "H2")
  * [ Writing to or reading from a dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title8 "H2")
  * [ Steps to create a group](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title9 "H2")
  * [ Steps to create and write to an attribute](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html#title10 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Introduction to HDF5
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
* * *
#  HDF5 Description
HDF5 consists of a file format for storing HDF5 data, a data model for logically organizing and accessing HDF5 data from an application, and the software (libraries, language interfaces, and tools) for working with this format.
##  File Format
HDF5 consists of a file format for storing HDF5 data, a data model for logically organizing and accessing HDF5 data from an application, and the software (libraries, language interfaces, and tools) for working with this format.
##  Data Model
The HDF5 Data Model, also known as the HDF5 Abstract (or Logical) Data Model consists of the building blocks for data organization and specification in HDF5.
An HDF5 file (an object in itself) can be thought of as a container (or group) that holds a variety of heterogeneous data objects (or datasets). The datasets can be images, tables, graphs, and even documents, such as PDF or Excel:
![](https://support.hdfgroup.org/documentation/hdf5/latest/fileobj.png)  
---  
The two primary objects in the HDF5 Data Model are groups and datasets.
There are also a variety of other objects in the HDF5 Data Model that support groups and datasets, including datatypes, dataspaces, properties and attributes.
###  Groups
HDF5 groups (and links) organize data objects. Every HDF5 file contains a root group that can contain other groups or be linked to objects in other files.
There are two groups in the HDF5 file depicted above: Viz and SimOut. Under the Viz group are a variety of images and a table that is shared with the SimOut group. The SimOut group contains a 3-dimensional array, a 2-dimensional array and a link to a 2-dimensional array in another HDF5 file. ![](https://support.hdfgroup.org/documentation/hdf5/latest/group.png)  
---  
Working with groups and group members is similar in many ways to working with directories and files in UNIX. As with UNIX directories and files, objects in an HDF5 file are often described by giving their full (or absolute) path names. 
  * / signifies the root group. 
  * /foo signifies a member of the root group called foo. 
  * /foo/zoo signifies a member of the group foo, which in turn is a member of the root group.


###  Datasets
HDF5 datasets organize and contain the “raw” data values. A dataset consists of metadata that describes the data, in addition to the data itself:
In this picture, the data is stored as a three dimensional dataset of size 4 x 5 x 6 with an integer datatype. It contains attributes, Time and Pressure, and the dataset is chunked and compressed. ![](https://support.hdfgroup.org/documentation/hdf5/latest/dataset.png)  
---  
Datatypes, dataspaces, properties and (optional) attributes are HDF5 objects that describe a dataset. The datatype describes the individual data elements.
##  Datatypes, Dataspaces, Properties and Attributes
###  Datatypes
The datatype describes the individual data elements in a dataset. It provides complete information for data conversion to or from that datatype.
In the dataset depicted, each element of the dataset is a 32-bit integer. ![](https://support.hdfgroup.org/documentation/hdf5/latest/datatype.png)  
---  
Datatypes in HDF5 can be grouped into: 
  * **Pre-Defined Datatypes** : These are datatypes that are created by HDF5. They are actually opened (and closed) by HDF5 and can have different values from one HDF5 session to the next. There are two types of pre-defined datatypes: 
    * Standard datatypes are the same on all platforms and are what you see in an HDF5 file. Their names are of the form H5T_ARCH_BASE where ARCH is an architecture name and BASE is a pro­gramming type name. For example, [H5T_IEEE_F32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga71d24a7d4c373ed9a003d7a0d8133f1e) indicates a standard Big Endian floating point type. 
    * Native datatypes are used to simplify memory operations (reading, writing) and are NOT the same on different platforms. For example, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) indicates an int (C). 
  * **Derived Datatypes** : These are datatypes that are created or derived from the pre-defined datatypes. An example of a commonly used derived datatype is a string of more than one character. Compound datatypes are also derived types. A compound datatype can be used to create a simple table, and can also be nested, in which it includes one more other compound datatypes.  This is an example of a dataset with a compound datatype. Each element in the dataset consists of a 16-bit integer, a character, a 32-bit integer, and a 2x3x2 array of 32-bit floats (the datatype). It is a 2-dimensional 5 x 3 array (the dataspace). The datatype should not be confused with the dataspace.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/cmpnddtype.png)  
---  


###  Dataspaces
A dataspace describes the layout of a dataset's data elements. It can consist of no elements (NULL), a single element (scalar), or a simple array.
This image illustrates a dataspace that is an array with dimensions of 5 x 3 and a rank (number of dimensions) of 2. ![](https://support.hdfgroup.org/documentation/hdf5/latest/dataspace1.png)  
---  
A dataspace can have dimensions that are fixed (unchanging) or unlimited, which means they can grow in size (i.e. they are extendible).
There are two roles of a dataspace: 
  * It contains the spatial information (logical layout) of a dataset stored in a file. This includes the rank and dimensions of a dataset, which are a permanent part of the dataset definition. 
  * It describes an application's data buffers and data elements participating in I/O. In other words, it can be used to select a portion or subset of a dataset.

The dataspace is used to describe both the logical layout of a dataset and a subset of a dataset. ![](https://support.hdfgroup.org/documentation/hdf5/latest/dataspace.png)  
---  
###  Properties
A property is a characteristic or feature of an HDF5 object. There are default properties which handle the most common needs. These default properties can be modified using the HDF5 Property List API to take advantage of more powerful or unusual features of HDF5 objects.
![](https://support.hdfgroup.org/documentation/hdf5/latest/properties.png)  
---  
For example, the data storage layout property of a dataset is contiguous by default. For better performance, the layout can be modified to be chunked or chunked and compressed:
###  Attributes
Attributes can optionally be associated with HDF5 objects. They have two parts: a name and a value. Attributes are accessed by opening the object that they are attached to so are not independent objects. Typically an attribute is small in size and contains user metadata about the object that it is attached to.
Attributes look similar to HDF5 datasets in that they have a datatype and dataspace. However, they do not support partial I/O operations, and they cannot be compressed or extended.
##  HDF5 Software
The HDF5 software is written in C and includes optional wrappers for C++, FORTRAN (90 and F2003), and Java. The HDF5 binary distribution consists of the HDF5 libraries, include files, command-line utilities, scripts for compiling applications, and example programs.
###  HDF5 APIs and Libraries
There are APIs for each type of object in HDF5. For example, all C routines in the HDF5 library begin with a prefix of the form H5*, where * is one or two uppercase letters indicating the type of object on which the function operates: 
  * [Attributes (H5A)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html) **A** ttribute Interface 
  * [Datasets (H5D)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html) **D** ataset Interface 
  * [Files (H5F)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html) **F** ile Interface


The HDF5 High Level APIs simplify many of the steps required to create and access objects, as well as providing templates for storing objects. Following is a list of the High Level APIs: 
  * [HDF5 Lite APIs (H5LT,H5LD)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html) – simplifies steps in creating datasets and attributes 
  * [HDF5 Images API (H5IM)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_i_m.html) – defines a standard for storing images in HDF5 
  * [HDF5 Table APIs (H5TB)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html) – condenses the steps required to create tables 
  * [HDF5 Dimension Scales APIs (H5DS)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html) – provides a standard for dimension scale storage 
  * [HDF5 Packet Table APIs (H5PT)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html) – provides a standard for storing packet data


###  Tools
Useful tools for working with HDF5 files include: 
  * [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) : A utility to dump or display the contents of an HDF5 File 
  * h5cc, h5c++, h5fc : pkg-config based shell scripts for compiling applications 
  * HDFView : A java browser to view HDF (HDF4 and HDF5) files


#### h5dump
The [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) utility displays the contents of an HDF5 file in Data Description Language ([DDL in BNF for HDF5 2.0.0 and above](https://support.hdfgroup.org/documentation/hdf5/latest/_d_d_l_b_n_f200.html)). Below is an example of [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) output for an HDF5 file that contains no objects: 
$ h5dump file.h5
HDF5 "file.h5" {
GROUP "/" {
}
}
With large files and datasets the output from [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) can be overwhelming. There are options that can be used to examine specific parts of an HDF5 file. Some useful [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) options are included below: 
-H, --header Display header information only (no data)
-d <name> Display a dataset with a specified path and name
-p Display properties
-n Display the contents of the file
#### h5cc, h5fc, h5c++
The built HDF5 binaries include the h5cc, h5fc, h5c++ compile scripts for compiling applications. When using these scripts there is no need to specify the HDF5 libraries and include files. Compiler options can be passed to the scripts.
#### HDFView
The HDFView tool allows browsing of data in HDF (HDF4 and HDF5) files.
#  Introduction to the HDF5 Programming Model and APIs
The HDF5 Application Programming Interface is extensive, but a few functions do most of the work.
To introduce the programming model, examples in Python and C are included below. The Python examples use the HDF5 Python APIs (h5py). See the Examples from "Learning the Basics" page for complete examples that can be downloaded and run for C, FORTRAN, C++, Java and Python.
The general paradigm for working with objects in HDF5 is to: 
  * Open the object. 
  * Access the object. 
  * Close the object.


The library imposes an order on the operations by argument dependencies. For example, a file must be opened before a dataset because the dataset open call requires a file handle as an argument. Objects can be closed in any order. However, once an object is closed it no longer can be accessed.
Keep the following in mind when looking at the example programs included in this section: 
  *     * C routines begin with the prefix “H5*” where * is a single letter indicating the object on which the operation is to be performed. 
    * FORTRAN routines are similar; they begin with “h5*” and end with “_f”. 
    * Java routines are similar; the routine names begin with “H5*” and are prefixed with “H5.” as the class. Constants are in the HDF5Constants class and are prefixed with "HDF5Constants.". The function arguments are usually similar, see [HDF5 Java API Package](https://support.hdfgroup.org/documentation/hdf5/latest/_h_d_f5_l_i_b.html)
For example: 
    * File Interface:
      * [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.") (C)
      * [h5f.h5fopen_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_f.html#gae9fa6f0b685c0e1516b7924dcb0e8819 "Opens HDF5 file.") (FORTRAN)
      * H5.H5Fopen (Java)
    * Dataset Interface:
      * [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) (C)
      * [h5d.h5dopen_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_d.html#ga56db335650709fe67bbfbc5a46ff4abf "Opens an existing dataset.") (FORTRAN)
      * H5.H5Dopen (Java)
    * Dataspace interface:
      * [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1 "Releases and terminates access to a dataspace.") (C)
      * [h5s.h5sclose_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_s.html#ga4642024f539f37493a8eef6e53552f50 "Releases and terminates access to a dataspace.") (FORTRAN)
      * H5.H5Sclose (Java)
The HDF5 Python APIs use methods associated with specific objects. 
  * For portability, the HDF5 library has its own defined types. Some common types that you will see in the example code are: 
    * [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) is used for object handles 
    * [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) is used for dimensions 
    * [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) is used for many return values 
  * Language specific files must be included in applications: 
    * **Python** :  
` import h5py  
  import numpy `
    * **C** :  
`"#include hdf5.h"`
    * **FORTRAN** :  
`USE HDF5`  
and call [h5lib.h5open_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5.html#ga3ca54ba860cf80a39bcaef71484f5baf "Initializes HDF5 Fortran interface.") and [h5lib.h5close_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5.html#gaaae21753ce0b3ef8eeb85720ac705d88 "Closes HDF5 Fortran interface.") to initialize and close the HDF5 FORTRAN interface 
    * **Java** :  
` import hdf.hdf5lib.H5[](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1_h5.html);  
  import hdf.hdf5lib.HDF5Constants[](https://support.hdfgroup.org/documentation/hdf5/latest/classhdf_1_1hdf5lib_1_1_h_d_f5_constants.html); `


##  Steps to create a file
To create an HDF5 file you must: 
  * Specify property lists (or use the defaults). 
  * Create the file. 
  * Close the file (and property lists if needed).


Example: 
The following Python and C examples create a file, file.h5, and then close it. The resulting HDF5 file will only contain a root group: ![](https://support.hdfgroup.org/documentation/hdf5/latest/crtf-pic.png)  
---  
Calling h5py.File with ‘w’ for the file access flag will create a new HDF5 file and overwrite an existing file with the same name. “file” is the file handle returned from opening the file. When finished with the file, it must be closed. When not specifying property lists, the default property lists are used:
_Python_ import h5py file = h5py.File (‘file.h5’, ‘w’) file.close ()  
---  
The H5Fcreate function creates an HDF5 file. [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) is the file access flag to create a new file and overwrite an existing file with the same name, and [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) is the value specified to use a default property list.
_C_ #include “hdf5.h” int main() { [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id; [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) status; file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4) ("file.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));  status = [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124) (file_id); } [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a) #define H5F_ACC_TRUNC **Definition** H5Fpublic.h:30 [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) int64_t hid_t **Definition** H5Ipublic.h:60 [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) #define H5P_DEFAULT **Definition** H5Ppublic.h:220 [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) int herr_t **Definition** H5public.h:252 [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124) herr_t H5Fclose(hid_t file_id) Terminates access to an HDF5 file. [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4) hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id) Creates an HDF5 file.  
---  
##  Steps to create a dataset
As described previously, an HDF5 dataset consists of the raw data, as well as the metadata that describes the data (datatype, spatial information, and properties). To create a dataset you must: 
  * Define the dataset characteristics (datatype, dataspace, properties). 
  * Decide which group to attach the dataset to. 
  * Create the dataset. 
  * Close the dataset handle from step 3.


Example: 
The code excerpts below show the calls that need to be made to create a 4 x 6 integer dataset dset in a file dset.h5. The dataset will be located in the root group: ![](https://support.hdfgroup.org/documentation/hdf5/latest/crtdset.png)  
---  
With Python, the creation of the dataspace is included as a parameter in the dataset creation method. Just one call will create a 4 x 6 integer dataset dset. A pre-defined Big Endian 32-bit integer datatype is specified. The create_dataset method creates the dataset in the root group (the file object). The dataset is close by the Python interface.
_Python_ dataset = file.create_dataset("dset",(4, 6), h5py.h5t.STD_I32BE)  
---  
To create the same dataset in C, you must specify the dataspace with the [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae "Creates a new simple dataspace and opens it for access.") function, create the dataset by calling [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf), and then close the dataspace and dataset with calls to [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609 "Closes the specified dataset.") and [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1 "Releases and terminates access to a dataspace."). [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) is specified to use a default property list. Note that the file identifier (file_id) is passed in as the first parameter to [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf), which creates the dataset in the root group.
_C_ // Create the dataspace for the dataset. dims[0] = 4; dims[1] = 6; dataspace_id = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(2, dims, NULL); // Create the dataset. dataset_id = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (file_id, "/dset", [H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8), dataspace_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)); // Close the dataset and dataspace status = [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dataset_id); status = [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(dataspace_id); [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) #define H5Dcreate **Definition** H5version.h:1124 [H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609) herr_t H5Dclose(hid_t dset_id) Closes the specified dataset. [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1) herr_t H5Sclose(hid_t space_id) Releases and terminates access to a dataspace. [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae) hid_t H5Screate_simple(int rank, const hsize_t dims[], const hsize_t maxdims[]) Creates a new simple dataspace and opens it for access. [H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8) #define H5T_STD_I32BE **Definition** H5Tpublic.h:466  
---  
##  Writing to or reading from a dataset
Once you have created or opened a dataset you can write to it:
_Python_ data = np.zeros((4,6)) for i in range(4): for j in range(6): data[i][j]= i*6+j+1 dataset[...] = data <-- Write data to dataset data_read = dataset[...] <-- Read data from dataset  
---  
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) is passed in for the memory and file dataspace parameters to indicate that the entire dataspace of the dataset is specified. These two parameters can be modified to allow subsetting of a dataset. The native predefined datatype, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), is used for reading and writing so that HDF5 will do any necessary integer conversions:
_C_ status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) (dataset_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dset_data); status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c) (dataset_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dset_data); [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484) #define H5S_ALL **Definition** H5Spublic.h:33 [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c) herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void *buf) Reads raw data from a dataset into a provided buffer. [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf) Writes raw data from a buffer to a dataset. [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) #define H5T_NATIVE_INT **Definition** H5Tpublic.h:988  
---  
##  Steps to create a group
An HDF5 group is a structure containing zero or more HDF5 objects. Before you can create a group you must obtain the location identifier of where the group is to be created. Following are the steps that are required: 
  * Decide where to put the group – in the “root group” (or file identifier) or in another group. Open the group if it is not already open. 
  * Define properties or use the default. 
  * Create the group. 
  * Close the group.

Creates attributes that are attached to the dataset dset ![](https://support.hdfgroup.org/documentation/hdf5/latest/crtgrp.png)  
---  
The code below opens the dataset dset.h5 with read/write permission and creates a group MyGroup in the root group. Properties are not specified so the defaults are used:
_Python_ import h5py file = h5py.File('dset.h5', 'r+')  group = file.create_group ('MyGroup') file.close()  
---  
To create the group MyGroup in the root group, you must call [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8), passing in the file identifier returned from opening or creating the file. The default property lists are specified with [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f). The group is then closed:
_C_ group_id = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) (file_id, "MyGroup", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)); status = [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221) (group_id); [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) #define H5Gcreate **Definition** H5version.h:1240 [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221) herr_t H5Gclose(hid_t group_id) Closes the specified group.  
---  
##  Steps to create and write to an attribute
To create an attribute you must open the object that you wish to attach the attribute to. Then you can create, access, and close the attribute as needed: 
  * Open the object that you wish to add an attribute to. 
  * Create the attribute 
  * Write to the attribute 
  * Close the attribute and the object it is attached to.

Creates attributes that are attached to the dataset dset ![](https://support.hdfgroup.org/documentation/hdf5/latest/crtatt.png)  
---  
The dataspace, datatype, and data are specified in the call to create an attribute in Python:
_Python_ dataset.attrs["Units"] = “Meters per second” <-- Create string attr_data = np.zeros((2,)) attr_data[0] = 100 attr_data[1] = 200 dataset.attrs.create("Speed", attr_data, (2,), “i”) <-- Create Integer  
---  
To create an integer attribute in C, you must create the dataspace, create the attribute, write to it and then close it in separate steps:
_C_ [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) attribute_id, dataspace_id; // identifiers [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dims; int attr_data[2]; [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) status; ... // Initialize the attribute data. attr_data[0] = 100; attr_data[1] = 200; // Create the data space for the attribute. dims = 2; dataspace_id = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, &dims, NULL); // Create a dataset attribute. attribute_id = [H5Acreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308) (dataset_id, "Units", [H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8), dataspace_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)); // Write the attribute data. status = [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c)(attribute_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), attr_data); // Close the attribute. status = [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda)(attribute_id); // Close the dataspace. status = [H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(dataspace_id); [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) uint64_t hsize_t **Definition** H5public.h:304 [H5Acreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308) hid_t H5Acreate2(hid_t loc_id, const char *attr_name, hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id) Creates an attribute attached to a specified object. [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c) herr_t H5Awrite(hid_t attr_id, hid_t type_id, const void *buf) Writes data to an attribute. [H5Aclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaef4394b661e2c930879e9868e122bdda) herr_t H5Aclose(hid_t attr_id) Closes the specified attribute.  
---  
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


