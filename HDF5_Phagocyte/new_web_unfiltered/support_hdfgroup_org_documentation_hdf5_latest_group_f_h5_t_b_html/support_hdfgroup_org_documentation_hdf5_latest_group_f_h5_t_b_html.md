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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title0 "H2")
  * [ Modules](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title1 "H2")
  * [ Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title2 "H2")
  * [Function/Subroutine Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title3 "H2")
  * [◆ h5tbdelete_field_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title4 "H2")
  * [◆ h5tbget_field_info_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title5 "H2")
  * [◆ h5tbget_table_info_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title6 "H2")
  * [◆ h5tbinsert_field_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title7 "H2")
  * [◆ h5tbmake_table_f() [1/2]](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title8 "H2")
  * [◆ h5tbmake_table_f() [2/2]](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title9 "H2")
  * [◆ h5tbread_field_index_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title10 "H2")
  * [◆ h5tbread_field_name_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title11 "H2")
  * [◆ h5tbread_table_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title12 "H2")
  * [◆ h5tbwrite_field_index_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title13 "H2")
  * [◆ h5tbwrite_field_name_f()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#title14 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Modules](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#namespaces) | [Functions/Subroutines](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#func-members)
Fortran High Level Table (H5TB) Interface
## Detailed Description 

See also
     [HDF5 Table APIs (H5TB)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html), C-HL API      [HDF5 High Level Table](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_b__u_g.html), User Guide 
##  Modules  
---  
module  | [h5tb](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5tb.html)  
| This module contains Fortran interfaces for H5TB.   
  
##  Functions/Subroutines  
---  
subroutine  |  [h5tbdelete_field_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga17337fa67e0e5428f9f77eb3a9afd6d6) (loc_id, dset_name, field_name, errcode)  
| Deletes a field from a table.   
  
subroutine  |  [h5tbget_field_info_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga4f66143082feb48482071385d9e69cd2) (loc_id, dset_name, nfields, field_names, field_sizes, field_offsets, type_size, errcode, maxlen_out)  
| Gets information about a table.   
  
subroutine  |  [h5tbget_table_info_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga94684db0c23355a61b01fedb63004d15) (loc_id, dset_name, nfields, nrecords, errcode)  
| Gets the table dimensions.   
  
subroutine  |  [h5tbinsert_field_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#gae7ad0b0fcd8716afba4edc28b1a9a943) (loc_id, dset_name, field_name, field_type, position, buf, errcode)  
| Insert a new field into a table.   
  
subroutine  |  [h5tbmake_table_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#gab4c09a81b47b9faa41267b4a85c22638) (table_title, loc_id, dset_name, nfields, nrecords, type_size, field_names, field_offset, field_types, chunk_size, compress, errcode)  
| Creates (DOES NOT WRITE) a dataset named `dset_name` attached to the object specified by the identifier `loc_id`.   
  
subroutine  |  [h5tbmake_table_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#gabde6041677cbb2b55e724eb8c0039d0d) (table_title, loc_id, dset_name, nfields, nrecords, type_size, field_names, field_offset, field_types, chunk_size, fill_data, compress, data, errcode)  
| Creates and writes a dataset named `dset_name` attached to the object specified by the identifier `loc_id`.   
  
subroutine  |  [h5tbread_field_index_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga6912acf184ebfae7da75306611891e4f) (loc_id, dset_name, field_index, start, nrecords, type_size, buf, errcode)  
| Reads field. The fields are identified by index.   
  
subroutine  |  [h5tbread_field_name_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#gab6251e60902dd6d52621d42fc727fef4) (loc_id, dset_name, field_name, start, nrecords, type_size, buf, errcode)  
| Reads one or several fields. The fields are identified by name.   
  
subroutine  |  [h5tbread_table_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga7aac089dbc0bbbc2df1ee694b5800ce6) (loc_id, dset_name, nfields, dst_size, dst_offset, dst_sizes, dst_buf, errcode)  
| Reads a table.   
  
subroutine  |  [h5tbwrite_field_index_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga9fc3119bc7a56d7e5d242637b794ff1e) (loc_id, dset_name, field_index, start, nrecords, type_size, buf, errcode)  
| Overwrites a field.   
  
subroutine  |  [h5tbwrite_field_name_f](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga25b403d2677d61d33a646f81e54df073) (loc_id, dset_name, field_name, start, nrecords, type_size, buf, errcode)  
| Overwrites field.   
  
## Function/Subroutine Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga17337fa67e0e5428f9f77eb3a9afd6d6)h5tbdelete_field_f()
subroutine h5tbdelete_field_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | character(len=*), intent(in) |  _field_name_ ,   
|  | integer |  _errcode_ )  
Deletes a field from a table.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the table.   
field_name | The name of the field to delete.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5TBdelete_field()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga04b55d602ef44e5128389ec284d68b1d)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga4f66143082feb48482071385d9e69cd2)h5tbget_field_info_f()
subroutine h5tbget_field_info_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nfields_ ,   
|  | character(len=*), dimension(nfields), intent(inout) |  _field_names_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), dimension(nfields), intent(inout) |  _field_sizes_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), dimension(nfields), intent(inout) |  _field_offsets_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(inout) |  _type_size_ ,   
|  | integer |  _errcode_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), optional |  _maxlen_out_ )  
Gets information about a table.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to read.   
nfields | The number of fields.   
field_names | An array containing the names of the fields.   
field_sizes | An array containing the size of the fields.   
field_offsets | An array containing the offsets of the fields.   
type_size | The size of the HDF5 datatype associated with the table (i.e., the size in bytes of the HDF5 compound datatype used to define a row, or record, in the table).   
errcode | Returns 0 if successful and -1 if it fails.   
maxlen_out | Maximum character length of the field names.  
See C API: [H5TBget_field_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae9aa3c0aa43f468b56eb82cb04dff0b2)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga94684db0c23355a61b01fedb63004d15)h5tbget_table_info_f()
subroutine h5tbget_table_info_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(inout) |  _nfields_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(inout) |  _nrecords_ ,   
|  | integer |  _errcode_ )  
Gets the table dimensions.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to read.   
nfields | The number of fields.   
nrecords | The number of records.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5TBget_table_info()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga5c3e0a9f56f3d32d0ccd1ce5ea40ee73)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#gae7ad0b0fcd8716afba4edc28b1a9a943)h5tbinsert_field_f()
subroutine h5tbinsert_field_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | character(len=*), intent(in) |  _field_name_ ,   
|  | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _field_type_ ,   
|  | integer, intent(in) |  _position_ ,   
|  | type(type), dimension(*), intent(in) |  _buf_ ,   
|  | integer |  _errcode_ )  
Insert a new field into a table.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the table.   
field_name | The name of the field to insert.   
field_type | The data type of the field.   
position | The zero based index position where to insert the field.   
buf | Buffer with data.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5TBinsert_field()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga26257e0f5cbdeb74b472b328f42c2aa3)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#gab4c09a81b47b9faa41267b4a85c22638)h5tbmake_table_f() [1/2]
subroutine h5tbmake_table_f  | ( | character(len=*), intent(in) |  _table_title_ ,   
---|---|---|---  
|  | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nfields_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nrecords_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(in) |  _type_size_ ,   
|  | character(len=*), dimension(1:nfields), intent(in) |  _field_names_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), dimension(1:nfields), intent(in) |  _field_offset_ ,   
|  | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), dimension(1:nfields), intent(in) |  _field_types_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _chunk_size_ ,   
|  | integer, intent(in) |  _compress_ ,   
|  | integer |  _errcode_ )  
Creates (DOES NOT WRITE) a dataset named `dset_name` attached to the object specified by the identifier `loc_id`.  

Attention
    Obsolete API, use the Fortran 2003 version instead. 

Parameters
     table_title | The title of the table.   
---|---  
loc_id | Location identifier. The identifier may be that of a file or group.   
dset_name | The name of the dataset to create.   
nfields | The number of fields.   
nrecords | The number of records.   
type_size | The size in bytes of the structure associated with the table. Obtained with sizeof or storage_size.   
field_names | An array containing the names of the fields.   
field_offset | An array containing the offsets of the fields.   
field_types | An array containing the type of the fields.   
chunk_size | The chunk size.   
compress | Flag that turns compression on or off.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5TBmake_table()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#gabde6041677cbb2b55e724eb8c0039d0d)h5tbmake_table_f() [2/2]
subroutine h5tbmake_table_f  | ( | character(len=*), intent(in) |  _table_title_ ,   
---|---|---|---  
|  | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nfields_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nrecords_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(in) |  _type_size_ ,   
|  | character(len=*), dimension(1:nfields), intent(in) |  _field_names_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), dimension(1:nfields), intent(in) |  _field_offset_ ,   
|  | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), dimension(1:nfields), intent(in) |  _field_types_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _chunk_size_ ,   
|  | type(c_ptr), intent(in) |  _fill_data_ ,   
|  | integer, intent(in) |  _compress_ ,   
|  | type(c_ptr), intent(in) |  _data_ ,   
|  | integer |  _errcode_ )  
Creates and writes a dataset named `dset_name` attached to the object specified by the identifier `loc_id`.  

Attention
    The preferred API, Fortran 2003 version. 

Parameters
     table_title | The title of the table   
---|---  
loc_id | Location identifier. The identifier may be that of a file or group.   
dset_name | The name of the dataset to create   
nfields | The number of fields   
nrecords | The number of records   
type_size | The size in bytes of the structure associated with the table; This value is obtained with sizeof().   
field_names | An array containing the names of the fields   
field_offset | An array containing the offsets of the fields   
field_types | An array containing the type of the fields   
chunk_size | The chunk size   
fill_data | Fill values data   
compress | Flag that turns compression on or off   
data | Buffer with data to be written to the table   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5TBmake_table()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga083cfcddd7624068dc028f68437009e4)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga6912acf184ebfae7da75306611891e4f)h5tbread_field_index_f()
subroutine h5tbread_field_index_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer, intent(in) |  _field_index_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _start_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nrecords_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(in) |  _type_size_ ,   
|  | type(type), dimension(*), intent(inout) |  _buf_ ,   
|  | integer |  _errcode_ )  
Reads field. The fields are identified by index.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to read.   
field_index | The indexes of the fields to read.   
start | The start record to read from.   
nrecords | The number of records to read.   
type_size | The size in bytes of the structure associated with the table. Obtained with sizeof or storage_size.   
buf | Buffer with data.   
errcode | Returns 0 if successful and -1 if it fails.  
See similar C API: [H5TBread_fields_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga49aae7fc2576211d9da65ed02007a793)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#gab6251e60902dd6d52621d42fc727fef4)h5tbread_field_name_f()
subroutine h5tbread_field_name_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | character(len=*), intent(in) |  _field_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _start_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nrecords_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(in) |  _type_size_ ,   
|  | type(type), dimension(*), intent(inout) |  _buf_ ,   
|  | integer |  _errcode_ )  
Reads one or several fields. The fields are identified by name.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to read.   
field_name | An array containing the names of the fields to read.   
start | The start record to read from.   
nrecords | The number of records to read.   
type_size | The size in bytes of the structure associated with the table. Obtained with sizeof or storage_size.   
buf | Buffer with data   
errcode | Returns 0 if successful and -1 if it fails.  
See similar C API: [H5TBread_fields_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga6d4baea1ae71f3a4863bbbb40609811d)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga7aac089dbc0bbbc2df1ee694b5800ce6)h5tbread_table_f()
subroutine h5tbread_table_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nfields_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(in) |  _dst_size_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), dimension(1:nfields), intent(in) |  _dst_offset_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), dimension(1:nfields), intent(in) |  _dst_sizes_ ,   
|  | type(c_ptr) |  _dst_buf_ ,   
|  | integer |  _errcode_ )  
Reads a table.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to read   
nfields | Number of fields, i.e., size of dst_offset and dst_sizes arrays.   
dst_size | The size of the structure type, as calculated by sizeof or storage_size   
dst_offset | An array containing the offsets of the fields. These offsets can be calculated with H5OFFSETOF.   
dst_sizes | An array containing the sizes of the fields. These sizes can be calculated with sizeof or storage_size.   
dst_buf | Pointer to buffer with data.   
errcode | Returns 0 if successful and -1 if it fails.  
See C API: [H5TBread_table()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#gae2074f18c8fd2c4cea26639386e066b7)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga9fc3119bc7a56d7e5d242637b794ff1e)h5tbwrite_field_index_f()
subroutine h5tbwrite_field_index_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | integer, intent(in) |  _field_index_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _start_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nrecords_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(in) |  _type_size_ ,   
|  | integer, dimension(*), intent(in) |  _buf_ ,   
|  | integer |  _errcode_ )  
Overwrites a field.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to overwrite.   
field_index | The indexe of the fields to write.   
start | The zero based index record to start writing.   
nrecords | The number of records to write.   
type_size | The size of the structure type, as calculated by sizeof or storage_size.   
buf | Buffer with data.   
errcode | Returns 0 if successful and -1 if it fails.  
See similar C API: [H5TBwrite_fields_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga0b956c26a0e5257dbde15c3e69ac31ee)
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5_t_b.html#ga25b403d2677d61d33a646f81e54df073)h5tbwrite_field_name_f()
subroutine h5tbwrite_field_name_f  | ( | integer([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)), intent(in) |  _loc_id_ ,   
---|---|---|---  
|  | character(len=*), intent(in) |  _dset_name_ ,   
|  | character(len=*), intent(in) |  _field_name_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _start_ ,   
|  | integer([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)), intent(in) |  _nrecords_ ,   
|  | integer([size_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespaceh5fortran__types.html#a3be2e7678b801f965eed1fffb85947bd)), intent(in) |  _type_size_ ,   
|  | type(type), dimension(*), intent(in) |  _buf_ ,   
|  | integer |  _errcode_ )  
Overwrites field.  

Parameters
     loc_id | Location identifier. The identifier may be that of a file or group.   
---|---  
dset_name | The name of the dataset to overwrite   
field_name | The names of the fields to write   
start | The zero index record to start writing   
nrecords | The number of records to write   
type_size | The size of the structure type, as calculated by sizeof or storage_size.   
buf | Buffer with data.   
errcode | Returns 0 if successful and -1 if it fails.  
See similar C API: [H5TBwrite_fields_name()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t_b.html#ga1284baf9489d596c76fa0458c7b3def6)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


