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
  * [ Properties and Property Lists in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#title1 "H2")
  * [ Property List Classes, Property Lists, and Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#title2 "H2")
  * [ Programming Model for Properties and Property Lists](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#title3 "H2")
  * [ Generic Properties Interface and User-defined Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#title4 "H2")
  * [ Property List Function Summaries](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#title5 "H2")
  * [ Additional Property List Resources](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#title6 "H2")
  * [ Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Properties and Property Lists in HDF5
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  Properties and Property Lists in HDF5
HDF5 property lists are the main vehicle to configure the behavior of HDF5 API functions.
Typically, property lists are created by instantiating one of the built-in or user-defined property list classes. After adding suitable properties, property lists are used when opening or creating HDF5 items, or when reading or writing data. Property lists can be modified by adding or changing properties. Property lists are deleted by closing the associated handles.
##  Introduction
HDF5 properties and property lists make it possible to shape or modify an HDF5 file, group, dataset, attribute, committed datatype, or even an I/O stream, in a number of ways. For example, you can do any of the following: 
  * Customize the storage layout of a file to suit a project or task. 
  * Create a chunked dataset. 
  * Apply compression or filters to raw data. 
  * Use either ASCII or UTF-8 character encodings. 
  * Create missing groups on the fly. 
  * Switch between serial and parallel I/O. 
  * Create consistency within a single file or across an international project.


Some properties enable an HDF5 application to take advantage of the capabilities of a specific computing environment while others make a file more compact; some speed the reading or writing of data while others enable more record-keeping at a per-object level. HDF5 offers nearly one hundred specific properties that can be used in literally thousands of combinations to maximize the usability of HDF5-stored data.
At the most basic level, a property list is a collection of properties, represented by name/value pairs that can be passed to various HDF5 functions, usually modifying default settings. A property list inherits a set of properties and values from a property list class. But that statement hardly provides a complete picture; in the rest of this section and in the next section, [Property List Classes, Property Lists, and Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#subsec_plist_class) , we will discuss these things in much more detail. After reading that material, the reader should have a reasonably complete understanding of how properties and property lists can be used in HDF5 applications.
![](https://support.hdfgroup.org/documentation/hdf5/latest/PropListEcosystem.gif) The HDF5 property environment  
---  
The remaining sections in this chapter discuss the following topics: 
  * What are properties, property lists, and property list classes? 
  * Property list programming model 
  * Generic property functions 
  * Summary listings of property list functions 
  * Additional resources


The discussions and function listings in this chapter focus on general property operations, object and link properties, and related functions.
File, group, dataset, datatype, and attribute properties are discussed in the chapters devoted to those features, where that information will be most convenient to users. For example, [HDF5 Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#sec_dataset) discusses dataset creation property lists and functions, dataset access property lists and functions, and dataset transfer property lists and functions. This chapter does not duplicate those discussions.
Generic property operations are an advanced feature and are beyond the scope of this guide.
This chapter assumes an understanding of the following chapters of this [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * [The HDF5 Data Model and File Structure](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_m__u_g.html#sec_data_model)
  * [The HDF5 Library and Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5__u_g.html#sec_program)


##  Property List Classes, Property Lists, and Properties
HDF5 property lists and the property list interface [Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) provide a mechanism for storing characteristics of objects in an HDF5 file and economically passing them around in an HDF5 application. In this capacity, property lists significantly reduce the burden of additional function parameters throughout the HDF5 API. Another advantage of property lists is that features can often be added to HDF5 by adding only property list functions to the API; this is particularly true when all other requirements of the feature can be accomplished internally to the library.
For instance, a file creation operation needs to know several things about a file, such as the size of the userblock or the sizes of various file data structures. Bundling this information as a property list simplifies the interface by reducing the number of parameters to the function [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4).
As illustrated in the figure above ("The HDF5 property environment"), the HDF5 property environment is a three-level hierarchy: 
  * Property list classes 
  * Property lists 
  * Properties


The following subsections discuss property list classes, property lists, and properties in more detail.
###  Property List Classes
A property list class defines the roles that property lists of that class can play. Each class includes all properties that are valid for that class with each property set to its default value. HDF5 offers a property lists class for each of the following situations.
Property list classes in HDF5 Property List Class |  | For further discussion   
---|---|---  
File creation (FCPL)  |  [H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977) | See various sections of [The HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#sec_file)  
File access (FAPL)  |  [H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e) | Used only as [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f).   
File mount (FMPL)  |  [H5P_FILE_MOUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a3f57eb3c4081b40ff8b036f438e68e5b) | For more information, see [File Mount Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#FileMountProps)  
Object creation (OCPL)  |  [H5P_OBJECT_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a42df2a1c964d2b985abc6e422abf6463) | See [Object Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html)  
Object copy (OCPYPL)  |  [H5P_OBJECT_COPY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a7cedfd989522e8d7697a414d1d707e43) |   
Group creation (GCPL)  |  [H5P_GROUP_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a8330a95b257d45d6347a2daa96f261e9) | See [Programming Model for Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#subsec_group_program)  
Group access (GAPL)  |  [H5P_GROUP_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aca0fe0d98945364fe1320bf3af056b9d) |   
Link creation (LCPL)  |  [H5P_LINK_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad2c1d10104f0262c826350ccbf7c49f2) | See examples in [Programming Model for Properties and Property Lists](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#subsec_plist_program) and [Link Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html)  
Link access (LAPL)  |  [H5P_LINK_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a130a641715c9a0f3597792ce630fbe6f) |   
Dataset creation (DCPL)  |  [H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102) | See [Programming Model for Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#subsec_dataset_program)  
Dataset access (DAPL)  |  [H5P_DATASET_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afd849c0834c8ce6580b7c2537dbd9b5d) |   
Dataset transfer (DXPL)  |  [H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d) |   
Datatype creation (TCPL)  |  [H5P_DATATYPE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6d9318d499a66b4a934fe1839b29566e) | See various sections of [HDF5 Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#sec_datatype)  
String creation (STRCPL)  |  [H5P_STRING_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad5c40cc58ce5ddb42dff95eb684bd6cf) | See [Programming Model for Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#subsec_dataset_program) and [Programming Model for Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#subsec_datatype_program)  
Attribute creation (ACPL)  |  [H5P_ATTRIBUTE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa0102211c679e031e2e9831b66c48a12) | See [Working with Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#subsec_attribute_work).   
Map creation (MCPL)  |  [H5P_MAP_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a50729bf4bfe1588f1ef3919e18e7d546) | See [Map Property Lists](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#subsec_map_plist). Only available with VOL connectors that support maps.   
Map access (MAPL)  |  [H5P_MAP_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a79320cf1e087a106edf5389499366ab3) | See [Map Property Lists](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m__u_g.html#subsec_map_plist). Only available with VOL connectors that support maps.   
Note: In the table above, the abbreviations to the right of each property list class name in this table are widely used in both HDF5 programmer documentation and HDF5 source code. For example, [File Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html) (FCPL) is the file creation property list, [Object Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html) (OCPL) is the object creation property list, [Object Copy Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html) (OCPYPL) is object copy property list, [String Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_t_r_c_p_l.html) (STRCPL) is the string creation property list, and [Map Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_c_p_l.html) (MCPL) and [VOL Data Mapping Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___m_a_p_l.html) (MAPL) are the map creation and access property lists. These abbreviations may appear in either uppercase or lowercase.
The “HDF5 property list class inheritance hierarchy” figure, immediately following, illustrates the inheritance hierarchy of HDF5's property list classes. Properties are defined at the root of the HDF5 property environment ([Property List Class Root](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html) in the figure below). Property list classes then inherit properties from that root, either directly or indirectly through a parent class. In every case, a property list class inherits only the properties relevant to its role. For example, the [Object Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html) (OCPL) inherits all properties that are relevant to the creation of any object while the [Group Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html) (GCPL) inherits only those properties that are relevant to group creation.
![](https://support.hdfgroup.org/documentation/hdf5/latest/PropListClassInheritance.gif) HDF5 property list class inheritance hierarchy  
---  
Note: In the figure above, property list classes displayed in black are directly accessible through the programming interface; the root of the property environment and the [String Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___s_t_r_c_p_l.html) and [Object Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html) property list classes, in gray above, are not user-accessible. The red empty set symbol indicates that the [File Mount Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_m_p_l.html) (FMPL) is an empty class; that is, it has no set table properties. For more information, see [File Mount Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#FileMountProps). Abbreviations used in this figure are defined in the preceding table, [Property list classes in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#table_plist).
###  Property Lists
A property list is a collection of related properties that are used together in specific circumstances. A new property list created from a property list class inherits the properties of the property list class and each property's default value. A fresh dataset creation property list, for example, includes all of the HDF5 properties relevant to the creation of a new dataset.
Property lists are implemented as containers holding a collection of name/value pairs. Each pair specifies a property name and a value for the property. A property list usually contains information for one to many properties.
HDF5's default property values are designed to be reasonable for general use cases. Therefore, an application can often use a property list without modification. On the other hand, adjusting property list settings is a routine action and there are many reasons for an application to do so.
A new property list may either be derived from a property list class or copied from an existing property list. When a property list is created from a property list class, it contains all the properties that are relevant to the class, with each property set to its default value. A new property list created by copying an existing property list will contain the same properties and property values as the original property list. In either case, the property values can be changed as needed through the HDF5 API.
Property lists can be freely reused to create consistency. For example, a single set of file, group, and dataset creation property lists might be created at the beginning of a project and used to create hundreds, thousands, even millions, of consistent files, file structures, and datasets over the project's life. When such consistency is important to a project, this is an economical means of providing it.
###  Properties
A property is the basic element of the property list hierarchy. HDF5 offers nearly one hundred properties controlling things ranging from file access rights, to the storage layout of a dataset, through optimizing the use of a parallel computing environment.
Further examples include the following: 
Purpose | Examples | Property List   
---|---|---  
Specify the driver to be used to open a file  | A POSIX driver or an MPI IO driver  |  [File Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html)  
Specify filters to be applied to a dataset  | Gzip compression or checksum evaluation  |  [Dataset Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html)  
Specify whether to record key times associated with an object  | Creation time and/or last-modified time  |  [Object Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html)  
Specify the access mode for a file opened via an external link  | Read-only or read-write  |  [Link Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html)  
Each property is initialized with a default value. For each property, there are one or more dedicated H5Pset_*calls that can be used to change that value.
#### Creation, access, and transfer properties:
Properties fall into one of several major categories: creation properties, access properties, and transfer properties.
Creation properties control permanent object characteristics. These characteristics must be established when an object is created, cannot change through the life of the object (they are immutable), and the property setting usually has a permanent presence in the file.
Examples of creation properties include: Whether a dataset is stored in a compact, contiguous, or chunked layout   
  
The default for this dataset creation property ([H5Pset_layout](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9)) is that a dataset is stored in a contiguous block. This works well for datasets with a known size limit that will fit easily in system memory.   
  
A chunked layout is important if a dataset is to be compressed, to enable extending the dataset's size, or to enable caching during I/O.   
  
A compact layout is suitable only for very small datasets because the raw data is stored in the object header.   
---  
Creation of intermediate groups when adding an object to an HDF5 file  
  
This link creation property, [H5Pset_create_intermediate_group](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c), enables an application to add an object in a file without having to know that the group or group hierarchy containing that object already exists. With this property set, HDF5 automatically creates missing groups. If this property is not set, an application must verify that each group in the path exists, and create those that do not, before creating the new object; if any group is missing, the create operation will fail.   
Whether an HDF5 file is a single file or a set of tightly related files that form a virtual HDF5 file  
  
Certain file creation properties enable the application to select one of several file layouts. Examples of the available layouts include a standard POSIX-compliant layout ([H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db)), a family of files ([H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d)), and a split file layout that separates raw data and metadata into separate files ([H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175)). These and other file layout options are discussed in [Alternate File Storage Layouts and Low-level File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsec_file_alternate_drivers).   
To enable error detection when creating a dataset  
  
In settings where data integrity is vulnerable, it may be desirable to set checksumming when datasets are created ([H5Pset_fletcher32](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga8bc81abfbd0393b0a46e121f817a3f81)). A subsequent application will then have a means to verify data integrity when reading the dataset.   
Access properties control transient object characteristics. These characteristics may change with the circumstances under which an object is accessed.
Examples of access properties include: The driver used to open a file  
  
For example, a file might be created with the MPI I/O driver ([H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1)) during high-speed data acquisition in a parallel computing environment. The same file might later be analyzed in a serial computing environment with I/O access handled through the serial POSIX driver ([H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db)).   
---  
Optimization settings in specialized environments  
  
Optimizations differ across computing environments and according to the needs of the task being performed, so are transient by nature.   
Transfer properties apply only to datasets and control transient aspects of data I/O. These characteristics may change with the circumstances under which data is accessed.
Examples of dataset transfer properties include: To enable error detection when reading a dataset  
  
If checksumming has been set on a dataset (with [H5Pset_fletcher32](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga8bc81abfbd0393b0a46e121f817a3f81), in the dataset creation property list), an application reading that dataset can choose whether to check for data integrity ([H5Pset_edc_check](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga0d95dfa506784acc9aed850c99713609)).   
---  
Various properties to optimize chunked data I/O on parallel computing systems  
  
HDF5 provides several properties for tuning I/O of chunked datasets in a parallel computing environment ([H5Pset_dxpl_mpio_chunk_opt](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga677259e85909add92495621fcad64e3a), [H5Pset_dxpl_mpio_chunk_opt_num](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga498f3dba540bf54d1e6ab3d9f92231f1), [H5Pset_dxpl_mpio_chunk_opt_ratio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#gafdcd34a6dc7a576f8a697318ad3d7b96), and [H5Pget_mpio_actual_chunk_opt_mode](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga37ec8c3b3f1880ed6e1b300bc4ee9ed5)).  
  
Optimal settings differ due to the characteristics of a computing environment and due to an application's data access patterns; even when working with the same file, these settings might change for every application and every platform.   
##  Programming Model for Properties and Property Lists
The programming model for HDF5 property lists is actually quite simple: 
  * Create a property list. 
  * Modify the property list, if required. 
  * Use the property list. 
  * Close the property list.


There are nuances, of course, but that is the basic process.
In some cases, you will not have to define property lists at all. If the default property settings are sufficient for your application, you can tell HDF5 to use the default property list.
The following sections first discuss the use of default property lists, then each step of the programming model, and finally a few less frequently used property list operations.
###  Using Default Property Lists
Default property lists can simplify many routine HDF5 tasks because you do not always have to create every property list you use.
An application that would be well-served by HDF5's default property settings can use the default property lists simply by substituting the value [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) for a property list identifier. HDF5 will then apply the default property list for the appropriate property list class.
For example, the function [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df) calls for a link creation property list, a dataset creation property list, and a dataset access property list. If the default properties are suitable for a dataset, this call can be made as 
dset_id = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)( loc_id, name, dtype_id, space_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) );
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)
hid_t H5Dcreate2(hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)
Creates a new dataset and links it into the file.
HDF5 will then apply the default link creation, dataset creation, and dataset access property lists correctly.
Of course, you would not want to do this without considering where it is appropriate, as there may be unforeseen consequences. Consider, for example, the use of chunked datasets. Optimal chunking is quite dependent on the makeup of the dataset and the most common access patterns, both of which must be taken into account in setting up the size and shape of chunks.
###  Basic Steps of the Programming Model
The steps of the property list programming model are described in the sub-sections below.
#### Create a Property List
A new property list can be created either as an instance of a property list class or by copying an existing property list. Consider the following examples. A new dataset creation property list is first created "from scratch" with [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa). A second dataset creation property list is then created by copying the first one with [H5Pcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b).
dcplA_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)
#define H5P_DATASET_CREATE
**Definition** H5Ppublic.h:60
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
The new dataset creation property list is created as an instance of the property list class [H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102).
The new dataset creation property list's identifier is returned in dcplA_id and the property list is initialized with default dataset creation property values.
A list of valid classes appears in the table [Property list classes in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_p__u_g.html#table_plist).
dcplB_id = [H5Pcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b) (dcplA_id);
[H5Pcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gad2663ccbcbf76b96cde4c104588ae21b)
hid_t H5Pcopy(hid_t plist_id)
Copies an existing property list to create a new property list.
A new dataset creation property list, dcplB_id, is created as a copy of dcplA_id and is initialized with dataset creation property values currently in dcplA_id.
At this point, dcplA_id and dcplB_id are identical; they will both contain any modified property values that were changed in dcplA_id before dcplB_id was created. They may, however, diverge as additional property values are reset in each.
While we are creating property lists, let's create a link creation property list; we will need this property list when the new dataset is linked into the file below: 
lcplAB_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_LINK_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad2c1d10104f0262c826350ccbf7c49f2));
[H5P_LINK_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad2c1d10104f0262c826350ccbf7c49f2)
#define H5P_LINK_CREATE
**Definition** H5Ppublic.h:116
#### Change Property Values
This section describes how to set property values.
Later in this section, the dataset creation property lists dcplA_id and dcplB_id created in the section above will be used respectively to create chunked and contiguous datasets. To set this up, we must set the layout property in each property list. The following example sets dcplA_id for chunked datasets and dcplB_id for contiguous datasets: 
error = [H5Pset_layout](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9) (dcplA_id, [H5D_CHUNKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06eadc846667d1f23d573964d22549e5f262));
error = [H5Pset_layout](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9) (dcplB_id, [H5D_CONTIGUOUS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06ea6161acec1a11680d488b5bb8694c79f1));
[H5D_CONTIGUOUS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06ea6161acec1a11680d488b5bb8694c79f1)
@ H5D_CONTIGUOUS
**Definition** H5Dpublic.h:61
[H5D_CHUNKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06eadc846667d1f23d573964d22549e5f262)
@ H5D_CHUNKED
**Definition** H5Dpublic.h:62
[H5Pset_layout](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9)
herr_t H5Pset_layout(hid_t plist_id, H5D_layout_t layout)
Sets the type of storage used to store the raw data for a dataset.
Since dcplA_id specifies a chunked layout, we must also set the number of dimensions and the size of the chunks. The example below specifies that datasets created with dcplA_id will be 3-dimensional and that the chunk size will be 100 in each dimension: 
error = [H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b) (dcplA_id, 3, [100,100,100]);
[H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)
herr_t H5Pset_chunk(hid_t plist_id, int ndims, const hsize_t dim[])
Sets the size of the chunks used to store a chunked layout dataset.
These datasets will be created with UTF-8 encoded names. To accomplish that, the following example sets the character encoding property in the link creation property list to create link names with UTF-8 encoding: 
error = [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc) (lcplAB_id, [H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658));
[H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)
@ H5T_CSET_UTF8
**Definition** H5Tpublic.h:97
[H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc)
herr_t H5Pset_char_encoding(hid_t plist_id, H5T_cset_t encoding)
Sets the character encoding used to encode link and attribute names.
dcplA_id can now be used to create chunked datasets and dcplB_id to create contiguous datasets. And with the use of lcplAB_id, they will be created with UTF-8 encoded names.
#### Use the Property List
Once the required property lists have been created, they can be used to control various HDF5 processes. For illustration, consider dataset creation.
Assume that the datatype dtypeAB and the dataspaces dspaceA and dspaceB have been defined and that the location identifier locAB_id specifies the group AB in the current HDF5 file. We have already created the required link creation and dataset creation property lists. For the sake of illustration, we assume that the default dataset access property list meets our application requirements. The following calls would create the datasets dsetA and dsetB in the group AB. The raw data in dsetA will be contiguous while dsetB raw data will be chunked; both datasets will have UTF-8 encoded link names:
dsetA_id = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)( locAB_id, dsetA, dtypeAB, dspaceA_id,
lcplAB_id, dcplA_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) );
dsetB_id = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)( locAB_id, dsetB, dtypeAB, dspaceB_id,
lcplAB_id, dcplB_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f) );
#### Close the Property List
Generally, creating or opening anything in an HDF5 file results in an HDF5 identifier. These identifiers are of HDF5 type [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) and include things like file identifiers, often expressed as file_id; dataset identifiers, dset_id; and property list identifiers, plist_id. To reduce the risk of memory leaks, all of these identifiers must be closed once they are no longer needed.
Property list identifiers are no exception to this rule, and [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb) is used for this purpose. The calls immediately following would close the property lists created and used in the examples above.
error = [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb) (dcplA_id);
error = [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb) (dcplB_id);
error = [H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb) (lcplAB_id);
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)
herr_t H5Pclose(hid_t plist_id)
Terminates access to a property list.
###  Additional Property List Operations
A few property list operations fall outside of the programming model described above. This section describes those operations.
#### Query the Class of an Existing Property List
Occasionally an application will have a property list but not know the corresponding property list class. A call such as in the following example will retrieve the unknown class of a known property list: 
PList_Class = [H5Pget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5) (dcplA_id);
[H5Pget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga9b230c1e85790f9f45c4ca2e79dd62c5)
hid_t H5Pget_class(hid_t plist_id)
Returns the property list class identifier for a property list.
Upon this function's return, PList_Class will contain the value [H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102) indicating that dcplA_id is a dataset creation property list.
#### Determine Current Creation Property List Settings in an Existing Object
After a file has been created, another application may work on the file without knowing how the creation properties for the file were set up. Retrieving these property values is often unnecessary; HDF5 can read the data and knows how to deal with any properties it encounters.
But sometimes an application must do something that requires knowing the creation property settings. HDF5 makes the acquisition of this information fairly straight-forward; for each property setting call, H5Pset_*, there is a corresponding H5Pget_*call to retrieve the property's current setting.
Consider the following examples which illustrate the determination of dataset layout and chunking settings:
The application must first identify the creation property list with the appropriate get creation property list call. There is one such call for each kind of object.
[H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163) will return a property list identifier for the creation property list that was used to create the dataset. Call it DCPL1_id.
[H5Pset_layout](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga75d80991a8f467e0d454c53a383ae7f9) sets a dataset's layout to be compact, contiguous, or chunked.
[H5Pget_layout](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga655530b0f40990507fedeef6b3068db3) called with DCPL1_id will return the dataset's layout, either [H5D_COMPACT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06ea922bab7d90bea9d3a0bb9082e0ca334d), [H5D_CONTIGUOUS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06ea6161acec1a11680d488b5bb8694c79f1), or [H5D_CHUNKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#a57e163d4c263b585ca2d904996f5e06eadc846667d1f23d573964d22549e5f262).
[H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b) sets the rank of a dataset, that is the number of dimensions it will have, and the maximum size of each dimension.
[H5Pget_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga4ef814034f601f48ab1ed6db79b4354c), also called with DCPL1_id, will return the rank of the dataset and the maximum size of each dimension.
If a creation property value has not been explicitly set, these H5Pget_calls will return the property's default value.
#### Determine Access Property Settings
Access property settings are quite different from creation properties. Since access property settings are not retained in an HDF5 file or object, there is normally no knowledge of the settings that were used in the past. On the other hand, since access properties do not affect characteristics of the file or object, this is not normally an issue. For more information, see "Access and Creation Property Exceptions."
One circumstance under which an application might need to determine access property settings might be when a file or object is already open but the application does not know the property list settings. In that case, the application can use the appropriate get access property list call to retrieve a property list identifier. For example, if the dataset dsetA from the earlier examples is still open, the following call would return an identifier for the dataset access property list in use: 
dsetA_dacpl_id = [H5Dget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga252c0ddac7a7817bd757190e7398353b)( dsetA_id );
[H5Dget_access_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga252c0ddac7a7817bd757190e7398353b)
hid_t H5Dget_access_plist(hid_t dset_id)
Returns the dataset access property list associated with a dataset.
The application could then use the returned property list identifier to analyze the property settings
##  Generic Properties Interface and User-defined Properties
HDF5's generic property interface provides tools for managing the entire property hierarchy and for the creation and management of user-defined property lists and properties. This interface also makes it possible for an application or a driver to create, modify, and manage custom properties, property lists, and property list classes. A comprehensive list of functions for this interface appears under "Generic Property Operations (Advanced)" in the "H5P: Property List Interface" section of the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
Further discussion of HDF5's generic property interface and user-defined properties and property lists is beyond the scope of this document.
##  Property List Function Summaries
General property functions, generic property functions and macros, property functions that are used with multiple types of objects, and object and link property functions are listed below.
Property list functions that apply to a specific type of object are listed in the chapter that discusses that object. For example, the [HDF5 Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#sec_dataset) chapter has two property list function listings: one for dataset creation property list functions and one for dataset access property list functions. As has been stated, this chapter is not intended to describe every property list function.
[Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) reference manual
##  Additional Property List Resources
Property lists are ubiquitous in an HDF5 environment and are therefore discussed in many places in HDF5 documentation. The following sections and listings in the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html) are of particular interest: 
  * In the [The HDF5 Data Model and File Structure](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_m__u_g.html#sec_data_model) chapter, see [Property List](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_m__u_g.html#subsubsec_data_model_abstract_plist). 
  * In the [The HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#sec_file) chapter, see the following sections and listings: 
    * [File Creation and File Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsec_file_creation_access)
    * [File Property Lists](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsec_file_property_lists)
    * [Example with the File Creation Property List](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsubsec_file_examples_props)
    * [Example with the File Access Property List](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#subsubsec_file_examples_access)
    * [Dataset creation property list functions (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#dcpl_table_tag)
    * [File access property list functions (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#fapl_table_tag)
    * [File driver property list functions (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#fd_pl_table_tag)
  * In the [HDF5 Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#sec_attribute) chapter, see "Attribute creation property list functions (H5P)". 
  * In the [HDF5 Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#sec_group) chapter, see "Group creation property list functions (H5P)". 
  * Property lists are discussed throughout [HDF5 Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#sec_dataset).


All property list functions are described in the [Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) section of the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html). The function index at the top of the page provides a categorized listing grouped by property list class. Those classes are listed below: 
  * [File Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html)
  * [File Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html)
  * [Group Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html)
  * [Dataset Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html)
  * [Dataset Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_a_p_l.html)
  * [Dataset Transfer Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html)
  * [Link Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html)
  * [Link Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html)
  * [Object Creation Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html)
  * [Object Copy Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_y_p_l.html)


Additional categories not related to the class structure are as follows: 
  * General property list operations 
  * Generic property list functions


The general property functions can be used with any property list; the generic property functions constitute an advanced feature.
The in-memory file image feature of HDF5 uses property lists in a manner that differs substantially from their use elsewhere in HDF5. Those who plan to use in-memory file images must study [HDF5 File Image](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_i_m__u_g.html).
##  Notes
#### File Mount Properties
While the file mount property list class [H5P_FILE_MOUNT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a3f57eb3c4081b40ff8b036f438e68e5b) is a valid HDF5 property list class, no file mount properties are defined by the HDF5 Library. References to a file mount property list should always be expressed as [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), meaning the default file mount property list.
#### Access and Creation Property Exceptions
There are a small number of exceptions to the rule that creation properties are always retained in a file or object and access properties are never retained.
The following properties are file access properties but they are not transient; they have permanent and different effects on a file. They could be validly classified as file creation properties as they must be set at creation time to properly create the file. But they are access properties because they must also be set when a file is reopened to properly access the file. 
Property | Related function   
---|---  
Family file driver  |  [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d)  
Split file driver  |  [H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175)  
Core file driver  |  [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f)  
The following is a link creation property, but it is not relevant after an object has been created and is not retained in the file or object. 
Property | Related function   
---|---  
Create missing intermediate groups  |  [H5Pset_create_intermediate_group](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_c_p_l.html#ga66c4c5d3f34e5cf65d00e47a5387383c)  
Previous Chapter [HDF5 Error Handling](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_e__u_g.html#sec_error) - Next Chapter [The HDF5 Virtual Object Layer (VOL)](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_v_l__u_g.html#sec_vol)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


