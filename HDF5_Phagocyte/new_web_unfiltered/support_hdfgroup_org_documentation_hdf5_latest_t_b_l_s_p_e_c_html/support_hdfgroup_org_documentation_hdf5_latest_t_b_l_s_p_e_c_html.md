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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_t_b_l_s_p_e_c.html#title0 "H1")
  * [ Table Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_b_l_s_p_e_c.html#title1 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Table Specification Version 1.0
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Specifications](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html)
* * *
The HDF5 specification defines the standard objects and storage for the standard HDF5 objects. (For information about the HDF5 library, model and specification, see the HDF documentation.) This document is an additional specification to define a standard profile for how to store tables in HDF5. Table data in HDF5 is stored as HDF5 datasets with standard attributes to define the properties of the tables.
#  Introduction
A generic table is a sequence of records, each record has a name and a type. Table data is stored as an HDF5 one dimensional compound dataset. A table is defined as a collection of records whose values are stored in fixed-length fields. All records have the same structure and all values in each field have the same data type.
The dataset for a table is distinguished from other datasets by giving it an attribute "CLASS=TABLE". Optional attributes allow the storage of a title for the Table and for each column, and a fill value for each column.
#  Table Attributes
The attributes for the Table are strings. They are written with the [H5LTset_attribute_string](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gae60e8867e35c255dc5b5f3de134df80c "Creates and writes a string attribute.") Lite API function. "Required" attributes must always be used. "Optional" attributes must be used when required. 
**Table 1. Attributes of an Image Dataset** **Attribute Name** |  **Required**  
**Optional** |  **Type** |  **String Size** |  **Value** |  **Description**  
---|---|---|---|---|---  
CLASS  | Required  | String  | 5  | "TABLE"  | This attribute is type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d), with size 5. For all Tables, the value of this attribute is **TABLE**. This attribute identifies this data set as intended to be interpreted as Table that conforms to the specifications on this page.   
VERSION  | Required  | String  | 3  | "0.2"  | This attribute is of type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d), with size corresponding to the length of the version string. This attribute identifies the version number of this specification to which it conforms. The current version number is "0.2".   
TITLE  | Optional  | String  |  |  | The **TITLE** is an optional String that is to be used as the informative title of the whole table. The **TITLE** is set with the parameter table_title of the function [H5TBmake_table](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4 "Creates and writes a table.").   
FIELD_(n)_NAME  | Required  | String  |  |  | The **FIELD_(n)_NAME** is an optional String that is to be used as the informative title of column n of the table. For each of the fields the word **FIELD_** is concatenated with the zero based field (n) index together with the name of the field.   
FIELD_(n)_FILL  | Optional  | String  |  |  | The **FIELD_(n)_FILL** is an optional String that is the fill value for column n of the table. For each of the fields the word **FIELD_** is concatenated with the zero based field (n) index together with the fill value, if present. This value is written only when a fill value is defined for the table.   
The following section of code shows the calls necessary to the creation of a table. 
// Create a new HDF5 file using default properties.
file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("my_table.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Call the make table function
[H5TBmake_table](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4)("Table Title", file_id, "Table1", NFIELDS, NRECORDS, dst_size, field_names, dst_offset, field_type, chunk_size, fill_data, compress, p_data);
// Close the file.
status = [H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file_id);
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
[H5TBmake_table](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4)
H5HL_DLL herr_t H5TBmake_table(const char *table_title, hid_t loc_id, const char *dset_name, hsize_t nfields, hsize_t nrecords, size_t type_size, const char *field_names[], const size_t *field_offset, const hid_t *field_types, hsize_t chunk_size, void *fill_data, int compress, const void *buf)
Creates and writes a table.
For more information see the [HDF5 Table APIs (H5TB)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html) reference manual page and the [HDF5 High Level Table](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_b__u_g.html), which includes examples.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Specifications](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


