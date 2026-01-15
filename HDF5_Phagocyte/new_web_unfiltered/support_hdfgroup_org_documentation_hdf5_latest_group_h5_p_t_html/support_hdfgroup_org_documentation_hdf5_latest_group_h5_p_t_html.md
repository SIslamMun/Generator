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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title2 "H2")
  * [◆ H5PTappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title3 "H2")
  * [◆ H5PTclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title4 "H2")
  * [◆ H5PTcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title5 "H2")
  * [◆ H5PTcreate_fl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title6 "H2")
  * [◆ H5PTcreate_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title7 "H2")
  * [◆ H5PTfree_vlen_buff()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title8 "H2")
  * [◆ H5PTget_dataset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title9 "H2")
  * [◆ H5PTget_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title10 "H2")
  * [◆ H5PTget_next()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title11 "H2")
  * [◆ H5PTget_num_packets()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title12 "H2")
  * [◆ H5PTget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title13 "H2")
  * [◆ H5PTis_valid()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title14 "H2")
  * [◆ H5PTis_varlen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title15 "H2")
  * [◆ H5PTopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title16 "H2")
  * [◆ H5PTread_packets()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title17 "H2")
  * [◆ H5PTset_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#title18 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#func-members)
HDF5 Packet Table APIs (H5PT)
## Detailed Description
_Creating and manipulating HDF5 datasets to support append- and read-only operations on table data (H5PT)_
The HDF5 Packet Table API is designed to allow records to be appended to and read from a table. Packet Table datasets are chunked, allowing them to grow as needed.
The Packet Table API, with the H5PT prefix, is not to be confused with the H5TB Table API (H5TB prefix). The H5TB APIs are stateless (H5TB Tables do not need to be opened or closed) but H5PT Packet Tables require less performance overhead. Also, H5TB Tables support insertions and deletions, while H5PT Packet Tables support only append operations. H5TB functions should not be called on tables created with the H5PT API, or vice versa.
Packet Tables are datasets in an HDF5 file, so while their contents should not be changed outside of the H5PT API calls, the datatypes of Packet Tables can be queried using [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb). Packet Tables can also be given attributes using the normal HDF5 APIs. 

Note
     **Programming hints:**      The following line includes the HDF5 Packet Table package, H5PT, in C applications: 
#include "hdf5_hl.h"
Without this include, an application will not have access to these functions.
  * [H5PTappend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaf889c2198653bd9bba6d82ba5020cd79)   
Appends packets to the end of a packet table.
  * [H5PTclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94)   
Closes an open packet table.
  * [H5PTcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gadebf2e89c0d382bb6a97fbf3c35b3261)   
Creates a packet table to store fixed-length or variable-length packets.
  * [H5PTcreate_fl](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gabed39877667be44b8b6bf3e6a4e2e3ab)   
Creates a packet table to store fixed-length packets.
  * [H5PTcreate_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga430cc0ee1f3d03f84c7169cfdf38e966)   
Resets a packet table's index to the first packet.
  * [H5PTfree_vlen_buff](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gab8a810a8ecca21eb24fe01ba142faf66)   
Releases memory allocated in the process of reading variable-length packets.
  * [H5PTget_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaba97ace057d14ac0381456347bba45c0)   
Returns the backend dataset of this packet table.
  * [H5PTget_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga11c003c9f4cf3ad3ed340a20963548a8)   
Gets the current record index for a packet table
  * [H5PTget_next](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga23ce6f7e0a08f14b6e516db3dab71385)   
Reads packets from a packet table starting at the current index.
  * [H5PTget_num_packets](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga2ebf603409c2f4959d47261a4cf379d2)   
Returns the number of packets in a packet table.
  * [H5PTget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga86cf3105b2aab245bedaa55ecd131d8d)   
Returns the backend datatype of this packet table.
  * [H5PTis_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gafec157fe9025d41b99b7f65c951bb159)   
Determines whether an identifier points to a packet table.
  * [H5PTis_varlen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga4d4d6aa197f39e07c1094363355cce50)   
Determines whether a packet table contains variable-length or fixed-length packets.
  * [H5PTopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga5053a36ebe72f78820138495a0e359fb)   
Opens an existing packet table.
  * [H5PTread_packets](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga9c230c5e08e26dde4d62057e568ebb52)   
Reads a number of packets from a packet table.
  * [H5PTset_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaa3f21cb7b44b0e8d9dea03ee2e344b8a)   
Sets a packet table's index. 


##  Functions  
---  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTappend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaf889c2198653bd9bba6d82ba5020cd79) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id, size_t nrecords, const void *data)  
| Appends packets to the end of a packet table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id)  
| Closes an open packet table.   
  
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5PTcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gadebf2e89c0d382bb6a97fbf3c35b3261) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) chunk_size, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id)  
| Creates a packet table to store fixed-length or variable-length packets.   
  
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5PTcreate_fl](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gabed39877667be44b8b6bf3e6a4e2e3ab) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) chunk_size, int compression)  
| Creates a packet table to store fixed-length packets.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTcreate_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga430cc0ee1f3d03f84c7169cfdf38e966) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id)  
| Resets a packet table's index to the first packet.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTfree_vlen_buff](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gab8a810a8ecca21eb24fe01ba142faf66) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id, size_t bufflen, void *buff)  
| Releases memory allocated in the process of reading variable-length packets.   
  
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5PTget_dataset](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaba97ace057d14ac0381456347bba45c0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id)  
| Returns the backend dataset of this packet table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTget_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga11c003c9f4cf3ad3ed340a20963548a8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *pt_index)  
| Gets the current record index for a packet table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTget_next](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga23ce6f7e0a08f14b6e516db3dab71385) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id, size_t nrecords, void *data)  
| Reads packets from a packet table starting at the current index.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTget_num_packets](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga2ebf603409c2f4959d47261a4cf379d2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *nrecords)  
| Returns the number of packets in a packet table.   
  
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5PTget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga86cf3105b2aab245bedaa55ecd131d8d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id)  
| Returns the backend datatype of this packet table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTis_valid](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gafec157fe9025d41b99b7f65c951bb159) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id)  
| Determines whether an identifier points to a packet table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTis_varlen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga4d4d6aa197f39e07c1094363355cce50) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id)  
| Determines whether a packet table contains variable-length or fixed-length packets.   
  
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  [H5PTopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga5053a36ebe72f78820138495a0e359fb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *dset_name)  
| Opens an existing packet table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTread_packets](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga9c230c5e08e26dde4d62057e568ebb52) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) start, size_t nrecords, void *data)  
| Reads a number of packets from a packet table.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5PTset_index](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaa3f21cb7b44b0e8d9dea03ee2e344b8a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) table_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) pt_index)  
| Sets a packet table's index.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaf889c2198653bd9bba6d82ba5020cd79)H5PTappend()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTappend  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _table_id_ ,   
---|---|---|---  
|  | size_t |  _nrecords_ ,   
|  | const void * |  _data_ )  
Appends packets to the end of a packet table. 
* * * 

Parameters
     [in] | table_id | Identifier of packet table to which packets should be appended   
---|---|---  
[in] | nrecords | Number of packets to be appended   
[in] | data | Buffer holding data to write 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The [H5PTappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaf889c2198653bd9bba6d82ba5020cd79 "Appends packets to the end of a packet table.") writes `nrecords` packets to the end of a packet table specified by `table_id`. `data` is a buffer containing the data to be written. For a packet table holding fixed-length packets, this data should be in the packet table's datatype. For a variable-length packet table, the data should be in the form of [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) structs. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94)H5PTclose()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTclose  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _table_id_ | ) |   
---|---|---|---|---|---  
Closes an open packet table. 
* * * 

Parameters
     [in] | table_id | Identifier of packet table to be closed  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The [H5PTclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94 "Closes an open packet table.") ends access to a packet table specified by `table_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gadebf2e89c0d382bb6a97fbf3c35b3261)H5PTcreate()
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5PTcreate  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dtype_id_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _chunk_size_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _plist_id_ )  
Creates a packet table to store fixed-length or variable-length packets. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the packet table to create   
[in] | dtype_id | The datatype of the packet   
[in] | chunk_size | The size in number of table entries per chunk   
[in] | plist_id | Identifier of the property list. Can be used to specify the compression of the packet table. 

Returns
    Returns an identifier for the new packet table or [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) on error.  
The [H5PTcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gadebf2e89c0d382bb6a97fbf3c35b3261 "Creates a packet table to store fixed-length or variable-length packets.") creates and opens a packet table named `dset_name` attached to the object specified by the identifier `loc_id`. The created packet table should be closed with [H5PTclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94 "Closes an open packet table."), eventually.
The datatype, `dtype_id`, may specify any datatype, including variable-length data. If `dtype_id` specifies a compound datatype, one or more fields in that compound type may be variable-length.
`chunk_size` is the size in number of table entries per chunk. Packet table datasets use HDF5 chunked storage to allow them to grow. This value allows the user to set the size of a chunk. The chunk size affects performance. 

Since
    1.10.0 and 1.8.17 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gabed39877667be44b8b6bf3e6a4e2e3ab)H5PTcreate_fl()
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5PTcreate_fl  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dtype_id_ ,   
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _chunk_size_ ,   
|  | int |  _compression_ )  
Creates a packet table to store fixed-length packets. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the dataset to create   
[in] | dtype_id | The datatype of a packet.   
[in] | chunk_size | The size in number of table entries per chunk.   
[in] | compression | The compression level; a value of 0 through 9. 

Returns
    Returns an identifier for the packet table or [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) on error. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000004)**
    This function was deprecated in favor of the function [H5PTcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gadebf2e89c0d382bb6a97fbf3c35b3261 "Creates a packet table to store fixed-length or variable-length packets.").  
The [H5PTcreate_fl()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gabed39877667be44b8b6bf3e6a4e2e3ab "Creates a packet table to store fixed-length packets.") creates and opens a packet table named `dset_name` attached to the object specified by the identifier `loc_id`. It should be closed with [H5PTclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94 "Closes an open packet table.").
The datatype, `dtype_id`, may specify any datatype, including variable-length data. If `dtype_id` specifies a compound datatype, one or more fields in that compound type may be variable-length.
`chunk_size` is the size in number of table entries per chunk. Packet table datasets use HDF5 chunked storage to allow them to grow. This value allows the user to set the size of a chunk. The chunk size affects performance.
`compression` is the compression level, a value of 0 through 9. Level 0 is faster but offers the least compression; level 9 is slower but offers maximum compression. A setting of -1 indicates that no compression is desired. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga430cc0ee1f3d03f84c7169cfdf38e966)H5PTcreate_index()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTcreate_index  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _table_id_ | ) |   
---|---|---|---|---|---  
Resets a packet table's index to the first packet. 
* * * 

Parameters
     [in] | table_id | Identifier of packet table whose index should be initialized.  
---|---|--- 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Each packet table keeps an index of the "current" packet so that `get_next` can iterate through the packets in order. [H5PTcreate_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga430cc0ee1f3d03f84c7169cfdf38e966 "Resets a packet table's index to the first packet.") initializes a packet table's index, and should be called before using `get_next`. The index must be initialized every time a packet table is created or opened; this information is lost when the packet table is closed. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gab8a810a8ecca21eb24fe01ba142faf66)H5PTfree_vlen_buff()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTfree_vlen_buff  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _table_id_ ,   
---|---|---|---  
|  | size_t |  _bufflen_ ,   
|  | void * |  _buff_ )  
Releases memory allocated in the process of reading variable-length packets. 
* * * 

Parameters
     [in] | table_id | Packet table whose memory should be freed.   
---|---|---  
[in] | bufflen | Size of `buff`  
[in] | buff | Buffer that was used to read in variable-length packets 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
When variable-length packets are read, memory is automatically allocated to hold them, and must be freed. [H5PTfree_vlen_buff()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gab8a810a8ecca21eb24fe01ba142faf66 "Releases memory allocated in the process of reading variable-length packets.") frees this memory, and should be called whenever packets are read from a variable-length packet table. 

Version
    1.10.0 and 1.8.17 Function re-introduced. Function had been removed in 1.8.3. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaba97ace057d14ac0381456347bba45c0)H5PTget_dataset()
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5PTget_dataset  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _table_id_ | ) |   
---|---|---|---|---|---  
Returns the backend dataset of this packet table. 
* * * 

Parameters
     [in] | table_id | Identifier of the packet table  
---|---|--- 

Returns
    Returns a dataset identifier or H5I_INVALID_HID on error.  
The [H5PTget_dataset()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaba97ace057d14ac0381456347bba45c0 "Returns the backend dataset of this packet table.") returns the identifier of the dataset storing the packet table `table_id`. This dataset identifier will be closed by [H5PTclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94 "Closes an open packet table."). 

Since
    1.10.0 and 1.8.17 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga11c003c9f4cf3ad3ed340a20963548a8)H5PTget_index()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTget_index  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _table_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _pt_index_ )  
Gets the current record index for a packet table. 
* * * 

Parameters
     [in] | table_id | Table identifier   
---|---|---  
[out] | pt_index | Current record index 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The [H5PTget_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga11c003c9f4cf3ad3ed340a20963548a8 "Gets the current record index for a packet table.") returns the current record index `pt_index` for the table identified by `table_id`. 

Since
    1.8.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga23ce6f7e0a08f14b6e516db3dab71385)H5PTget_next()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTget_next  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _table_id_ ,   
---|---|---|---  
|  | size_t |  _nrecords_ ,   
|  | void * |  _data_ )  
Reads packets from a packet table starting at the current index. 
* * * 

Parameters
     [in] | table_id | Identifier of packet table to read from   
---|---|---  
[in] | nrecords | Number of packets to be read   
[out] | data | Buffer into which to read data 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The [H5PTget_next()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga23ce6f7e0a08f14b6e516db3dab71385 "Reads packets from a packet table starting at the current index.") reads `nrecords` packets starting with the "current index" from a packet table specified by `table_id`. The packet table's index is set and reset with [H5PTset_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaa3f21cb7b44b0e8d9dea03ee2e344b8a "Sets a packet table's index.") and [H5PTcreate_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga430cc0ee1f3d03f84c7169cfdf38e966 "Resets a packet table's index to the first packet."). `data` is a buffer into which the data should be read.
For a packet table holding variable-length records, the data returned in the buffer will be in form of a [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) struct containing the length of the data and a pointer to it in memory. The memory used by this data must be freed using [H5PTfree_vlen_buff()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gab8a810a8ecca21eb24fe01ba142faf66 "Releases memory allocated in the process of reading variable-length packets."). 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga2ebf603409c2f4959d47261a4cf379d2)H5PTget_num_packets()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTget_num_packets  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _table_id_ ,   
---|---|---|---  
|  |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _nrecords_ )  
Returns the number of packets in a packet table. 
* * * 

Parameters
     [in] | table_id | Identifier of packet table to query   
---|---|---  
[out] | nrecords | Number of packets in packet table 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The [H5PTget_num_packets()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga2ebf603409c2f4959d47261a4cf379d2 "Returns the number of packets in a packet table.") returns by reference the number of packets in a packet table specified by `table_id`. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga86cf3105b2aab245bedaa55ecd131d8d)H5PTget_type()
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5PTget_type  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _table_id_ | ) |   
---|---|---|---|---|---  
Returns the backend datatype of this packet table. 
* * * 

Parameters
     [in] | table_id | Identifier of the packet table  
---|---|--- 

Returns
    Returns a datatype identifier or H5I_INVALID_HID on error.  
The [H5PTget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga86cf3105b2aab245bedaa55ecd131d8d "Returns the backend datatype of this packet table.") returns the identifier of the datatype used by the packet table `table_id`. This datatype identifier will be closed by [H5PTclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94 "Closes an open packet table."). 

Since
    1.10.0 and 1.8.17 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gafec157fe9025d41b99b7f65c951bb159)H5PTis_valid()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTis_valid  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _table_id_ | ) |   
---|---|---|---|---|---  
Determines whether an identifier points to a packet table. 
* * * 

Parameters
     [in] | table_id | Identifier to query  
---|---|--- 

Returns
    Returns a non-negative value if `table_id` is a valid packet table, otherwise returns a negative value.  
The [H5PTis_valid()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gafec157fe9025d41b99b7f65c951bb159 "Determines whether an identifier points to a packet table.") returns a non-negative value if `table_id` corresponds to an open packet table, and returns a negative value otherwise. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga4d4d6aa197f39e07c1094363355cce50)H5PTis_varlen()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTis_varlen  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) | _table_id_ | ) |   
---|---|---|---|---|---  
Determines whether a packet table contains variable-length or fixed-length packets. 
* * * 

Parameters
     [in] | table_id | Packet table to query  
---|---|--- 

Returns
    Returns 1 for a variable-length packet table, 0 for fixed-length, or a negative value on error.  
The [H5PTis_varlen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga4d4d6aa197f39e07c1094363355cce50 "Determines whether a packet table contains variable-length or fixed-length packets.") returns 1 (TRUE) if `table_id` is a packet table containing variable-length records. It returns 0 (FALSE) if `table_id` is a packet table containing fixed-length records. If `table_id` is not a packet table, a negative value is returned. 

Version
    1.10.0 and 1.8.17 Function re-introduced. Function had been removed in 1.8.3. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga5053a36ebe72f78820138495a0e359fb)H5PTopen()
H5HL_DLL [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) H5PTopen  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _dset_name_ )  
Opens an existing packet table. 
* * * 

Parameters
     [in] | loc_id | Location identifier. The identifier may be that of a file or group.   
---|---|---  
[in] | dset_name | The name of the packet table to open 

Returns
    Returns an identifier for the packet table or [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba) on error.  
[H5PTopen()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga5053a36ebe72f78820138495a0e359fb "Opens an existing packet table.") opens an existing packet table in the file or group specified by `loc_id`. `dset_name` is the name of the packet table and is used to identify it in the file. This function is used to open both fixed-length packet tables and variable-length packet tables. The packet table should later be closed with [H5PTclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga972403e11e64729d8435c363b138ae94 "Closes an open packet table."). 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga9c230c5e08e26dde4d62057e568ebb52)H5PTread_packets()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTread_packets  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _table_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _start_ ,   
|  | size_t |  _nrecords_ ,   
|  | void * |  _data_ )  
Reads a number of packets from a packet table. 
* * * 

Parameters
     [in] | table_id | Identifier of packet table to read from   
---|---|---  
[in] | start | Packet to start reading from   
[in] | nrecords | Number of packets to be read   
[out] | data | Buffer into which to read data. 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The [H5PTread_packets()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#ga9c230c5e08e26dde4d62057e568ebb52 "Reads a number of packets from a packet table.") reads `nrecords` packets starting at packet number `start` from a packet table specified by `table_id`. `data` is a buffer into which the data should be read.
For a packet table holding variable-length records, the data returned in the buffer will be in form of [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) structs, each containing the length of the data and a pointer to it in memory. The memory used by this data must be freed using [H5PTfree_vlen_buff()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gab8a810a8ecca21eb24fe01ba142faf66 "Releases memory allocated in the process of reading variable-length packets."). 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaa3f21cb7b44b0e8d9dea03ee2e344b8a)H5PTset_index()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5PTset_index  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _table_id_ ,   
---|---|---|---  
|  | [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) |  _pt_index_ )  
Sets a packet table's index. 
* * * 

Parameters
     [in] | table_id | Identifier of packet table whose index is to be set   
---|---|---  
[in] | pt_index | The packet to which the index should point 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
Each packet table keeps an index of the "current" packet so that `get_next` can iterate through the packets in order. [H5PTset_index()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p_t.html#gaa3f21cb7b44b0e8d9dea03ee2e344b8a "Sets a packet table's index.") sets this index to point to a user-specified packet (the packets are zero-indexed). 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


