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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title2 "H2")
  * [◆ H5TBadd_records_from()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title3 "H2")
  * [◆ H5TBAget_fill()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title4 "H2")
  * [◆ H5TBAget_title()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title5 "H2")
  * [◆ H5TBappend_records()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title6 "H2")
  * [◆ H5TBcombine_tables()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title7 "H2")
  * [◆ H5TBdelete_field()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title8 "H2")
  * [◆ H5TBdelete_record()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title9 "H2")
  * [◆ H5TBget_field_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title10 "H2")
  * [◆ H5TBget_table_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title11 "H2")
  * [◆ H5TBinsert_field()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title12 "H2")
  * [◆ H5TBinsert_record()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title13 "H2")
  * [◆ H5TBmake_table()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title14 "H2")
  * [◆ H5TBread_fields_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title15 "H2")
  * [◆ H5TBread_fields_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title16 "H2")
  * [◆ H5TBread_records()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title17 "H2")
  * [◆ H5TBread_table()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title18 "H2")
  * [◆ H5TBwrite_fields_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title19 "H2")
  * [◆ H5TBwrite_fields_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title20 "H2")
  * [◆ H5TBwrite_records()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#title21 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#func-members)
HDF5 Table APIs (H5TB)
## Detailed Description
_Creating and manipulating HDF5 datasets intended to be interpreted as tables (H5TB)_
The HDF5 Table API defines a standard storage for HDF5 datasets that are intended to be interpreted as tables. A table is defined as a collection of records whose values are stored in fixed-length fields. All records have the same structure, and all values in each field have the same data type. 

Note
     **Programming hints:**      To use any of these functions or subroutines, you must first include the relevant include file (C) or module (Fortran) in your application.       The following line includes the HDF5 Table package, H5TB, in C applications: 
#include "hdf5_hl.h"      To include the H5TB module in Fortran applications specify: 
use [h5tb](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5tb.html)
[h5tb](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5tb.html)
This module contains Fortran interfaces for H5TB.
**Definition** H5TBff_gen.F90:29
Fortran applications must also include [H5open](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga27fa33dc262dda95c5aa8df533837480) before any HDF5 calls to initialize global variables and [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574) after all HDF5 calls to close the Fortran interface.
  * Creation
    * [H5TBmake_table](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4)
  * Storage
    * [H5TBappend_records](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga4a8c5e6eca7119562365e796cbddb869) (No Fortran)
    * [H5TBwrite_records](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga55943628b2dfa001c679822bff0423b1) (No Fortran)
    * [H5TBwrite_fields_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga1284baf9489d596c76fa0458c7b3def6)
    * [H5TBwrite_fields_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga0b956c26a0e5257dbde15c3e69ac31ee)
  * Modification
    * [H5TBdelete_record](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga2bc8caf57fcff335ada3bc35558afa36) (No Fortran)
    * [H5TBinsert_record](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga372a7ccb50846b8b93acfb3c31d38c75) (No Fortran)
    * [H5TBadd_records_from](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga8cb3bfc58f48abc5e65254b69564ddef) (No Fortran)
    * [H5TBcombine_tables](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5aaca5013124852c566f3b6d0c89fb1b) (No Fortran)
    * [H5TBinsert_field](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga26257e0f5cbdeb74b472b328f42c2aa3)
    * [H5TBdelete_field](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga04b55d602ef44e5128389ec284d68b1d)

| 
  * Retrieval
    * [H5TBread_table](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae2074f18c8fd2c4cea26639386e066b7)
    * [H5TBread_records](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga84846ed25ee8cc57b89a842d566bc187) (No Fortran)
    * [H5TBread_fields_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga6d4baea1ae71f3a4863bbbb40609811d)
    * [H5TBread_fields_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga49aae7fc2576211d9da65ed02007a793)
  * Query
    * [H5TBget_table_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5c3e0a9f56f3d32d0ccd1ce5ea40ee73)
    * [H5TBget_field_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae9aa3c0aa43f468b56eb82cb04dff0b2)
  * Query Table Attributes
    * [H5TBAget_fill](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gabf7a9c110d125794c43e6a3678888a5d)
    * [H5TBAget_title](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga3c970bbc5ed74b6d69462ab0893c0d80)

  
---|---  
##  Functions  
---  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBadd_records_from](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga8cb3bfc58f48abc5e65254b69564ddef) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name1, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start1, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, const char *dset_name2, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start2)  
| Add records from first table to second table.   
  
H5HL_DLL [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) |  [H5TBAget_fill](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gabf7a9c110d125794c43e6a3678888a5d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, unsigned char *dst_buf)  
| Reads the table attribute fill values.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBAget_title](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga3c970bbc5ed74b6d69462ab0893c0d80) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, char *table_title)  
| Reads a table's title.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBappend_records](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga4a8c5e6eca7119562365e796cbddb869) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t type_size, const size_t *field_offset, const size_t *dst_sizes, const void *buf)  
| Adds records to the end of the table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBcombine_tables](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5aaca5013124852c566f3b6d0c89fb1b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id1, const char *dset_name1, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id2, const char *dset_name2, const char *dset_name3)  
| Combines records from two tables into a third.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBdelete_field](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga04b55d602ef44e5128389ec284d68b1d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, const char *field_name)  
| Deletes a field from a table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBdelete_record](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga2bc8caf57fcff335ada3bc35558afa36) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords)  
| Delete records.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBget_field_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae9aa3c0aa43f468b56eb82cb04dff0b2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, char *field_names[], size_t *field_sizes, size_t *field_offsets, size_t *type_size)  
| Gets information about a table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBget_table_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5c3e0a9f56f3d32d0ccd1ce5ea40ee73) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *nfields, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *nrecords)  
| Gets the table dimensions.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBinsert_field](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga26257e0f5cbdeb74b472b328f42c2aa3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, const char *field_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) field_type, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) position, const void *fill_data, const void *buf)  
| Insert a new field into a table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBinsert_record](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga372a7ccb50846b8b93acfb3c31d38c75) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t dst_size, const size_t *dst_offset, const size_t *dst_sizes, void *buf)  
| Insert records.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBmake_table](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4) (const char *table_title, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nfields, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t type_size, const char *field_names[], const size_t *field_offset, const [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) *field_types, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) chunk_size, void *fill_data, int compress, const void *buf)  
| Creates and writes a table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBread_fields_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga49aae7fc2576211d9da65ed02007a793) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nfields, const int *field_index, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t type_size, const size_t *field_offset, const size_t *dst_sizes, void *buf)  
| Reads one or several fields. The fields are identified by index.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBread_fields_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga6d4baea1ae71f3a4863bbbb40609811d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, const char *field_names, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t type_size, const size_t *field_offset, const size_t *dst_sizes, void *buf)  
| Reads one or several fields. The fields are identified by name.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBread_records](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga84846ed25ee8cc57b89a842d566bc187) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t type_size, const size_t *dst_offset, const size_t *dst_sizes, void *buf)  
| Reads records.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBread_table](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae2074f18c8fd2c4cea26639386e066b7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, size_t dst_size, const size_t *dst_offset, const size_t *dst_sizes, void *dst_buf)  
| Reads a table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBwrite_fields_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga0b956c26a0e5257dbde15c3e69ac31ee) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nfields, const int *field_index, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t type_size, const size_t *field_offset, const size_t *dst_sizes, const void *buf)  
| Overwrites fields.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBwrite_fields_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga1284baf9489d596c76fa0458c7b3def6) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, const char *field_names, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t type_size, const size_t *field_offset, const size_t *dst_sizes, const void *buf)  
| Overwrites fields.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5TBwrite_records](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga55943628b2dfa001c679822bff0423b1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) nrecords, size_t type_size, const size_t *field_offset, const size_t *dst_sizes, const void *buf)  
| Overwrites records.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga8cb3bfc58f48abc5e65254b69564ddef)H5TBadd_records_from()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBadd_records_from  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name1_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start1_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | const char * |  _dset_name2_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start2_ )  
Add records from first table to second table. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name1 | The name of the dataset to read the records   
[in] | start1 | The position to read the records from the first table   
[in] | nrecords | The number of records to read from the first table   
[in] | dset_name2 | The name of the dataset to write the records   
[in] | start2 | The position to write the records on the second table 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBadd_records_from()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga8cb3bfc58f48abc5e65254b69564ddef "Add records from first table to second table.") adds records from a dataset named `dset_name1` to a dataset named `dset_name2`. Both tables are attached to the object specified by the identifier loc_id. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gabf7a9c110d125794c43e6a3678888a5d)H5TBAget_fill()
H5HL_DLL [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) H5TBAget_fill  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
|  | unsigned char * |  _dst_buf_ )  
Reads the table attribute fill values. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | Name of table   
[in] | dset_id | Table identifier   
[out] | dst_buf | Buffer of fill values for table fields 

Returns
     A return value of 1 indicates that a fill value is present.       A return value of 0 indicates a fill value is not present.       A return value <0 indicates an error.  
H5TBget_fill() reads the table attribute fill values into the buffer `dst_buf` for the table specified by `dset_id` and `dset_name` located in `loc_id`. 

Example
    
unsigned char tmp_fill_buf[40];
...
file_id = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(FILENAME, [H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dataset_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file_id, TABLE_NAME, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
datatype_id = [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)(dataset_id);
status = [H5TBget_table_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5c3e0a9f56f3d32d0ccd1ce5ea40ee73)(file_id, TABLE_NAME, &nfields, &nrecords);
hasfill = [H5TBAget_fill](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gabf7a9c110d125794c43e6a3678888a5d)(file_id, TABLE_NAME, dataset_id, tmp_fill_buf);
for (i = 0; i < nfields; i++) {
member_type_id = [H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb)(datatype_id, (unsigned)i);
native_mem_type_id = [H5Tget_native_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga05b99133058637e8daa5d745381ddd3d)(member_type_id, [H5T_DIR_ASCEND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a891787104495ea80c677ab7042bfc33da0c49a07f752898162207bb2767b20cc6));
member_offset = [H5Tget_member_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af)(datatype_id, (unsigned)i);
printf("member_offset: %i\n", member_offset);
memb_class = [H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898)(member_type_id);
switch (memb_class) {
case [H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037):
/* convert unsigned char array to integer */
break;
case [H5T_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2e92f1a42a19de186a139ab8ff0745a9):
/* convert unsigned char array to double or float */
if ([H5Tequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa92250f289b557b63cba974defa20b0f)(native_mem_type_id, [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09))) {
}
else if ([H5Tequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa92250f289b557b63cba974defa20b0f)(native_mem_type_id, [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c))) {
f.i = tmp_fill_buf[member_offset] | (tmp_fill_buf[member_offset + 1] << 8) |
(tmp_fill_buf[member_offset + 2] << 16) | (tmp_fill_buf[member_offset + 3] << 24);
printf("Field %i Fill Value: %lf\n", i, f.f);
}
break;
case [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa):
/* convert unsigned char array to string */
strsize = [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(member_type_id);
printf("Field %i Fill Value: ", i);
for (j = 0; j < strsize; j++)
printf("%c", tmp_fill_buf[member_offset + j]);
printf("\n");
break;
}
[H5F_ACC_RDWR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a402673dec5c537b27a49a9a8bd6140b4)
#define H5F_ACC_RDWR
**Definition** H5Fpublic.h:29
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa)
@ H5T_STRING
**Definition** H5Tpublic.h:35
[H5T_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2e92f1a42a19de186a139ab8ff0745a9)
@ H5T_FLOAT
**Definition** H5Tpublic.h:33
[H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037)
@ H5T_INTEGER
**Definition** H5Tpublic.h:32
[H5T_DIR_ASCEND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a891787104495ea80c677ab7042bfc33da0c49a07f752898162207bb2767b20cc6)
@ H5T_DIR_ASCEND
**Definition** H5Tpublic.h:160
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
[H5Tget_member_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af)
size_t H5Tget_member_offset(hid_t type_id, unsigned membno)
Retrieves the offset of a field of a compound datatype.
[H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb)
hid_t H5Tget_member_type(hid_t type_id, unsigned membno)
Returns the datatype of the specified member.
[H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)
hid_t H5Dget_type(hid_t dset_id)
Returns an identifier for a copy of the datatype for a dataset.
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
[H5TBget_table_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5c3e0a9f56f3d32d0ccd1ce5ea40ee73)
H5HL_DLL herr_t H5TBget_table_info(hid_t loc_id, const char *dset_name, hsize_t *nfields, hsize_t *nrecords)
Gets the table dimensions.
[H5TBAget_fill](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gabf7a9c110d125794c43e6a3678888a5d)
H5HL_DLL htri_t H5TBAget_fill(hid_t loc_id, const char *dset_name, hid_t dset_id, unsigned char *dst_buf)
Reads the table attribute fill values.
[H5Tget_native_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga05b99133058637e8daa5d745381ddd3d)
hid_t H5Tget_native_type(hid_t type_id, H5T_direction_t direction)
Returns the native datatype identifier of a specified datatype.
[H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)
size_t H5Tget_size(hid_t type_id)
Returns the size of a datatype.
[H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898)
H5T_class_t H5Tget_class(hid_t type_id)
Returns a datatype class.
[H5Tequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa92250f289b557b63cba974defa20b0f)
htri_t H5Tequal(hid_t type1_id, hid_t type2_id)
Determines whether two datatype identifiers refer to the same datatype.
[H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09)
#define H5T_NATIVE_DOUBLE
**Definition** H5Tpublic.h:1036
[H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c)
#define H5T_NATIVE_FLOAT
**Definition** H5Tpublic.h:1030
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga3c970bbc5ed74b6d69462ab0893c0d80)H5TBAget_title()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBAget_title  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | char * |  _table_title_ )  
Reads a table's title. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[out] | table_title | Buffer for title name 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
H5TBget_title() returns the title of the table identified by `loc_id` in a buffer `table_title`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga4a8c5e6eca7119562365e796cbddb869)H5TBappend_records()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBappend_records  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _type_size_ ,   
|  | const size_t * |  _field_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | const void * |  _buf_ )  
Adds records to the end of the table. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to overwrite   
[in] | nrecords | The number of records to append   
[in] | type_size | The size of the structure type, as calculated by `sizeof()`.   
[in] | field_offset | An array containing the offsets of the fields. These offsets can be calculated with the [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro   
[in] | dst_sizes | An array containing the sizes of the fields   
[in] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBappend_records()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga4a8c5e6eca7119562365e796cbddb869 "Adds records to the end of the table.") adds records to the end of the table named `dset_name` attached to the object specified by the identifier `loc_id`. The dataset is extended to hold the new records. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5aaca5013124852c566f3b6d0c89fb1b)H5TBcombine_tables()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBcombine_tables  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id1_ ,   
---|---|---|---  
|  | const char * |  _dset_name1_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id2_ ,   
|  | const char * |  _dset_name2_ ,   
|  | const char * |  _dset_name3_ )  
Combines records from two tables into a third. 
* * * 

Parameters
     [in] | loc_id1 | Identifier of the file or group in which the first table is located   
---|---|---  
[in] | dset_name1 | The name of the first table to combine   
[in] | loc_id2 | Identifier of the file or group in which the second table is located   
[in] | dset_name2 | The name of the second table to combine   
[in] | dset_name3 | The name of the new table 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBcombine_tables()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5aaca5013124852c566f3b6d0c89fb1b "Combines records from two tables into a third.") combines records from two datasets named `dset_name1` and `dset_name2`, to a new table named `dset_name3`. These tables can be located on different files, identified by `loc_id1` and `loc_id2` (identifiers obtained with [H5Fcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.")). They can also be located on the same file. In this case one uses the same identifier for both parameters `loc_id1` and `loc_id2`. If two files are used, the third table is written in the first file. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga04b55d602ef44e5128389ec284d68b1d)H5TBdelete_field()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBdelete_field  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | const char * |  _field_name_ )  
Deletes a field from a table. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the table   
[in] | field_name | The name of the field to delete 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBdelete_field()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga04b55d602ef44e5128389ec284d68b1d "Deletes a field from a table.") deletes a field named `field_name` from the table `dset_name`. Note: this function requires the table to be re-created and rewritten in its entirety, and this can result in some unused space in the file, and can also take a great deal of time if the table is large. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga2bc8caf57fcff335ada3bc35558afa36)H5TBdelete_record()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBdelete_record  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ )  
Delete records. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset   
[in] | start | The start record to delete from   
[in] | nrecords | The number of records to delete 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBdelete_record()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga2bc8caf57fcff335ada3bc35558afa36 "Delete records.") deletes nrecords number of records starting from `start` from the middle of the table `dset_name` ("pulling up" all the records after it). 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae9aa3c0aa43f468b56eb82cb04dff0b2)H5TBget_field_info()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBget_field_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | char * |  _field_names_[],   
|  | size_t * |  _field_sizes_ ,   
|  | size_t * |  _field_offsets_ ,   
|  | size_t * |  _type_size_ )  
Gets information about a table. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to read   
[out] | field_names | An array containing the names of the fields   
[out] | field_sizes | An array containing the size of the fields   
[out] | field_offsets | An array containing the offsets of the fields   
[out] | type_size | The size of the HDF5 datatype associated with the table. (More specifically, the size in bytes of the HDF5 compound datatype used to define a row, or record, in the table) 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBget_field_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae9aa3c0aa43f468b56eb82cb04dff0b2 "Gets information about a table.") gets information about a dataset named `dset_name` attached to the object specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5c3e0a9f56f3d32d0ccd1ce5ea40ee73)H5TBget_table_info()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBget_table_info  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _nfields_ ,   
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _nrecords_ )  
Gets the table dimensions. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to read   
[out] | nfields | The number of fields   
[out] | nrecords | The number of records 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBget_table_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5c3e0a9f56f3d32d0ccd1ce5ea40ee73 "Gets the table dimensions.") retrieves the table dimensions from a dataset named `dset_name` attached to the object specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga26257e0f5cbdeb74b472b328f42c2aa3)H5TBinsert_field()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBinsert_field  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | const char * |  _field_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _field_type_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _position_ ,   
|  | const void * |  _fill_data_ ,   
|  | const void * |  _buf_ )  
Insert a new field into a table. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the table   
[in] | field_name | The name of the field to insert   
[in] | field_type | The data type of the field   
[in] | position | The zero based index position where to insert the field   
[in] | fill_data | Fill value data for the field. This parameter can be NULL   
[in] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBinsert_field()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga26257e0f5cbdeb74b472b328f42c2aa3 "Insert a new field into a table.") inserts a new field named `field_name` into the table `dset_name`. Note: this function requires the table to be re-created and rewritten in its entirety, and this can result in some unused space in the file, and can also take a great deal of time if the table is large. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga372a7ccb50846b8b93acfb3c31d38c75)H5TBinsert_record()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBinsert_record  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _dst_size_ ,   
|  | const size_t * |  _dst_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | void * |  _buf_ )  
Insert records. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset   
[in] | start | The position to insert   
[in] | nrecords | The number of records to insert   
[in] | dst_size | The size in bytes of the structure associated with the table   
[in] | dst_offset | An array containing the offsets of the fields   
[in] | dst_sizes | An array containing the size in bytes of the fields   
[in] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBinsert_record()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga372a7ccb50846b8b93acfb3c31d38c75 "Insert records.") inserts records into the middle of the table ("pushing down" all the records after it) 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4)H5TBmake_table()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBmake_table  | ( | const char * |  _table_title_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nfields_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _type_size_ ,   
|  | const char * |  _field_names_[],   
|  | const size_t * |  _field_offset_ ,   
|  | const [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) * |  _field_types_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _chunk_size_ ,   
|  | void * |  _fill_data_ ,   
|  | int |  _compress_ ,   
|  | const void * |  _buf_ )  
Creates and writes a table. 
* * * 

Parameters
     [in] | table_title | The title of the table   
---|---|---  
[in] | loc_id | Location identifier. The identifier may be that of a file or group.   
[in] | dset_name | The name of the dataset to create   
[in] | nfields | The number of fields   
[in] | nrecords | The number of records   
[in] | type_size | The size in bytes of the structure associated with the table; This value is obtained with `sizeof()`.   
[in] | field_names | An array containing the names of the fields   
[in] | field_offset | An array containing the offsets of the fields   
[in] | field_types | An array containing the type of the fields   
[in] | chunk_size | The chunk size   
[in] | fill_data | Fill values data   
[in] | compress | Flag that turns compression on or off   
[in] | buf | Buffer with data to be written to the table 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBmake_table()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4 "Creates and writes a table.") creates and writes a dataset named `dset_name` attached to the object specified by the identifier loc_id. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga49aae7fc2576211d9da65ed02007a793)H5TBread_fields_index()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBread_fields_index  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nfields_ ,   
|  | const int * |  _field_index_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _type_size_ ,   
|  | const size_t * |  _field_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | void * |  _buf_ )  
Reads one or several fields. The fields are identified by index. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to read   
[in] | nfields | The number of fields to read (This parameter is also the size of the `field_index` array.) fields to read   
[in] | field_index | The indexes of the fields to read   
[in] | start | The start record to read from   
[in] | nrecords | The number of records to read   
[in] | type_size | The size in bytes of the structure associated with the table (This value is obtained with `sizeof()`)   
[in] | field_offset | An array containing the offsets of the fields   
[in] | dst_sizes | An array containing the size in bytes of the fields   
[out] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBread_fields_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga49aae7fc2576211d9da65ed02007a793 "Reads one or several fields. The fields are identified by index.") reads the fields identified by `field_index` from a dataset named `dset_name` attached to the object specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga6d4baea1ae71f3a4863bbbb40609811d)H5TBread_fields_name()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBread_fields_name  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | const char * |  _field_names_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _type_size_ ,   
|  | const size_t * |  _field_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | void * |  _buf_ )  
Reads one or several fields. The fields are identified by name. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to read   
[in] | field_names | An array containing the names of the fields to read   
[in] | start | The start record to read from   
[in] | nrecords | The number of records to read   
[in] | type_size | The size in bytes of the structure associated with the table (This value is obtained with `sizeof()`.)   
[in] | field_offset | An array containing the offsets of the fields   
[in] | dst_sizes | An array containing the size in bytes of the fields   
[out] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBread_fields_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga6d4baea1ae71f3a4863bbbb40609811d "Reads one or several fields. The fields are identified by name.") reads the fields identified by `field_names` from a dataset named `dset_name` attached to the object specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga84846ed25ee8cc57b89a842d566bc187)H5TBread_records()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBread_records  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _type_size_ ,   
|  | const size_t * |  _dst_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | void * |  _buf_ )  
Reads records. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to read   
[in] | start | The start record to read from   
[in] | nrecords | The number of records to read   
[in] | type_size | The size of the structure type, as calculated by `sizeof()`  
[in] | dst_offset | An array containing the offsets of the fields. These offsets can be calculated with the [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro   
[in] | dst_sizes | An array containing the size in bytes of the fields   
[out] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBread_records()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga84846ed25ee8cc57b89a842d566bc187 "Reads records.") reads some records identified from a dataset named `dset_name` attached to the object specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae2074f18c8fd2c4cea26639386e066b7)H5TBread_table()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBread_table  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | size_t |  _dst_size_ ,   
|  | const size_t * |  _dst_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | void * |  _dst_buf_ )  
Reads a table. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to read   
[in] | dst_size | The size of the structure type, as calculated by `sizeof()`  
[in] | dst_offset | An array containing the offsets of the fields. These offsets can be calculated with the [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro   
[in] | dst_sizes | An array containing the sizes of the fields. These sizes can be calculated with the sizeof() macro.   
[in] | dst_buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBread_table()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae2074f18c8fd2c4cea26639386e066b7 "Reads a table.") reads a table named `dset_name` attached to the object specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga0b956c26a0e5257dbde15c3e69ac31ee)H5TBwrite_fields_index()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBwrite_fields_index  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nfields_ ,   
|  | const int * |  _field_index_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _type_size_ ,   
|  | const size_t * |  _field_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | const void * |  _buf_ )  
Overwrites fields. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to overwrite   
[in] | nfields | The number of fields to overwrite. This parameter is also the size of the `field_index` array.   
[in] | field_index | The indexes of the fields to write   
[in] | start | The zero based index record to start writing   
[in] | nrecords | The number of records to write   
[in] | type_size | The size of the structure type, as calculated by `sizeof()`.   
[in] | field_offset | An array containing the offsets of the fields. These offsets can be calculated with the [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro   
[in] | dst_sizes | An array containing the sizes of the fields   
[in] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBwrite_fields_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga0b956c26a0e5257dbde15c3e69ac31ee "Overwrites fields.") overwrites one or several fields specified by `field_index` with a buffer `buf` from a dataset named `dset_name` attached to the object specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga1284baf9489d596c76fa0458c7b3def6)H5TBwrite_fields_name()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBwrite_fields_name  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | const char * |  _field_names_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _type_size_ ,   
|  | const size_t * |  _field_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | const void * |  _buf_ )  
Overwrites fields. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to overwrite   
[in] | field_names | The names of the fields to write   
[in] | start | The zero index record to start writing   
[in] | nrecords | The number of records to write   
[in] | type_size | The size of the structure type, as calculated by `sizeof()`.   
[in] | field_offset | An array containing the offsets of the fields. These offsets can be calculated with the [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro   
[in] | dst_sizes | An array containing the sizes of the fields   
[in] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBwrite_fields_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga1284baf9489d596c76fa0458c7b3def6 "Overwrites fields.") overwrites one or several fields specified by `field_names` with data in `buf` from a dataset named `dset_name` attached to the object specified by the identifier `loc_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga55943628b2dfa001c679822bff0423b1)H5TBwrite_records()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5TBwrite_records  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _nrecords_ ,   
|  | size_t |  _type_size_ ,   
|  | const size_t * |  _field_offset_ ,   
|  | const size_t * |  _dst_sizes_ ,   
|  | const void * |  _buf_ )  
Overwrites records. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to overwrite   
[in] | start | The zero index record to start writing   
[in] | nrecords | The number of records to write   
[in] | type_size | The size of the structure type, as calculated by `sizeof()`.   
[in] | field_offset | An array containing the offsets of the fields. These offsets can be calculated with the [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro   
[in] | dst_sizes | An array containing the sizes of the fields   
[in] | buf | Buffer with data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
[H5TBwrite_records()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga55943628b2dfa001c679822bff0423b1 "Overwrites records.") overwrites records starting at the zero index position start of the table named `dset_name` attached to the object specified by the identifier `loc_id`. 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


