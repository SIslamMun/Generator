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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t_d_i_s_c.html#title0 "H1")
  * [ File Format Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t_d_i_s_c.html#title1 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 File Format Discussion
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * * 

Document's Audience: 
    
Current [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html) library designers and knowledgeable external developers. 

Background Reading: 
     [HDF5 File Format Specification Version 4.0](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html)   
This describes the current HDF5 file format. 
#  Introduction
**What is this document about?**  
This document attempts to explain the HDF5 file format specification with a few examples and describes some potential improvements to the format specification.
#  File Format Examples
This section has several small programs and describes the format of a file created with each of them.
Example program one - _Create an empty file_ : 
#include "hdf5.h"
#include <assert.h>
int main()
{
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fid; /* File ID */
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) ret; /* Generic return value */
/* Create the file */
fid=[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("example1.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
assert(fid&gt;=0);
/* Close the file */
ret=[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(fid);
assert(ret&gt;=0);
return(0);
}
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
**Super Block** byte  | byte  | byte  | byte   
---|---|---|---  
\211  | 'H'  | 'D'  | 'F'   
\r  | \n  | \032  | \n   
0  | 0  | 0  | 0   
0  | 8  | 8  | 0   
4  | 16   
0x00000003   
0   
  
0xffffffffffffffff  
  
  
?   
  
0xffffffffffffffff  
  
  
| 0   
---  
  
928  
  
  
H5G_CACHED_STAB (1)   
0   
|    
384  
  
  
---  
  
96  
  
  
%h5debug example1.h5
Reading signature at address 0 (rel)
File Super Block...
File name: example1.h5
File access flags 0x00000000
File open reference count: 1
Address of super block: 0 (abs)
Size of user block: 0 bytes
Super block version number: 0
Free list version number: 0
Root group symbol table entry version number: 0
Shared header version number: 0
Size of file offsets ([haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) type): 8 bytes
Size of file lengths ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) type): 8 bytes
Symbol table leaf node 1/2 rank: 4
Symbol table internal node 1/2 rank: 16
File consistency flags: 0x00000003
Base address: 0 (abs)
Free list address: UNDEF (rel)
Address of driver information block: UNDEF (rel)
Root group symbol table entry:
Name offset into private heap: 0
Object header address: 928
Dirty: Yes
Cache info type: Symbol Table
Cached information:
B-tree address: 384
Heap address: 96
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f)
uint64_t haddr_t
**Definition** H5public.h:355
**Root Group Object Header** byte  | byte  | byte  | byte   
---|---|---|---  
1  | 0  | 2   
1   
32   
0x0011  | 16   
0x01  | 0   
|    
384  
  
  
---  
  
96  
  
  
0  | 0   
0x00  | 0   
%h5debug example1.h5 928
New address: 928
Reading signature at address 928 (rel)
Object Header...
Dirty: 0
Version: 1
Header size (in bytes): 16
Number of links: 1
Number of messages (allocated): 2 (32)
Number of chunks (allocated): 1 (8)
Chunk 0...
Dirty: 0
Address: 944
Size in bytes: 32
Message 0...
Message ID (sequence number): 0x0011 stab(0)
Shared message: No
Constant: Yes
Raw size in obj header: 16 bytes
Chunk number: 0
Message Information:
B-tree address: 384
Name heap address: 96
Message 1...
Message ID (sequence number): 0x0000 null(0)
Shared message: No
Constant: No
Raw size in obj header: 0 bytes
Chunk number: 0
Message Information:
<no info="" for="" this="" message="">
**Root Group Local Heap** byte  | byte  | byte  | byte   
---|---|---|---  
'H'  | 'E'  | 'A'  | 'P'   
0   
256   
8   
128   
%h5debug example1.h5 96
New address: 96
Reading signature at address 96 (rel)
Local Heap...
Dirty: 0
Header size (in bytes): 32
Address of heap data: 128
Data bytes allocated on disk: 256
Data bytes allocated in core: 256
Free Blocks (offset, size):
Block #0: 8, 248
Percent of heap used: 3.12%
Data follows (`__' indicates free region)...
0: 00 00 00 00 00 00 00 00 __ __ __ __ __ __ __ __ ........
16: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
32: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
48: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
64: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
80: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
96: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
112: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
128: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
144: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
160: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
176: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
192: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
208: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
224: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
240: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
**Root Group B-tree** byte  | byte  | byte  | byte   
---|---|---|---  
'T'  | 'R'  | 'E'  | 'E'   
0  | 0  | 0   
  
0xffffffffffffffff  
  
  
  
0xffffffffffffffff  
  
  
%h5debug example1.h5 384 96
New address: 384
Reading signature at address 384 (rel)
Tree type ID: H5B_SNODE_ID
Size of node: 544
Size of raw (disk) key: 8
Dirty flag: False
Number of initial dirty children: 0
Level: 0
Address of left sibling: UNDEF
Address of right sibling: UNDEF
Number of children (max): 0 (32)
Example program two - _Create a file with a single dataset in it_ : 
#include "hdf5.h"
#include <assert.h>
int main()
{
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fid; /* File ID */
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) sid; /* Dataspace ID */
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) did; /* Dataset ID */
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) ret; /* Generic return value */
/* Create the file */
fid=[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("example2.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
assert(fid&gt;=0);
/* Create a scalar dataspace for the dataset */
sid=[H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)([H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8));
assert(sid&gt;=0);
/* Create a trivial dataset */
did=[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(fid, "Dataset", [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), sid, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
assert(did&gt;=0);
/* Close the dataset */
ret=[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(did);
assert(ret&gt;=0);
/* Close the dataspace */
ret=[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(sid);
assert(ret&gt;=0);
/* Close the file */
ret=[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(fid);
assert(ret&gt;=0);
return(0);
}
[H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8)
@ H5S_SCALAR
**Definition** H5Spublic.h:90
[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
#define H5Dcreate
**Definition** H5version.h:1124
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)
herr_t H5Dclose(hid_t dset_id)
Closes the specified dataset.
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)
herr_t H5Sclose(hid_t space_id)
Releases and terminates access to a dataspace.
[H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)
hid_t H5Screate(H5S_class_t type)
Creates a new dataspace of a specified type.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
**Super Block** byte  | byte  | byte  | byte   
---|---|---|---  
\211  | 'H'  | 'D'  | 'F'   
\r  | \n  | \032  | \n   
0  | 0  | 0  | 0   
0  | 8  | 8  | 0   
4  | 16   
0x00000003   
0   
  
0xffffffffffffffff  
  
  
?   
  
0xffffffffffffffff  
  
  
| 0   
---  
  
928  
  
  
H5G_CACHED_STAB (1)   
0   
|    
384  
  
  
---  
  
96  
  
  
%h5debug example2.h5
Reading signature at address 0 (rel)
File Super Block...
File name: example2.h5
File access flags 0x00000000
File open reference count: 1
Address of super block: 0 (abs)
Size of user block: 0 bytes
Super block version number: 0
Free list version number: 0
Root group symbol table entry version number: 0
Shared header version number: 0
Size of file offsets ([haddr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a8c86e866f40d7167cf9a1934c72b856f) type): 8 bytes
Size of file lengths ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) type): 8 bytes
Symbol table leaf node 1/2 rank: 4
Symbol table internal node 1/2 rank: 16
File consistency flags: 0x00000003
Base address: 0 (abs)
Free list address: UNDEF (rel)
Address of driver information block: UNDEF (rel)
Root group symbol table entry:
Name offset into private heap: 0
Object header address: 928
Dirty: Yes
Cache info type: Symbol Table
Cached entry information:
B-tree address: 384
Heap address: 96
**Root Group Object Header** byte  | byte  | byte  | byte   
---|---|---|---  
1  | 0  | 2   
1   
32   
0x0011  | 16   
0x01  | 0   
|    
384  
  
  
---  
  
96  
  
  
0  | 0   
0x00  | 0   
%h5debug example2.h5 928
New address: 928
Reading signature at address 928 (rel)
Object Header...
Dirty: 0
Version: 1
Header size (in bytes): 16
Number of links: 1
Number of messages (allocated): 2 (32)
Number of chunks (allocated): 1 (8)
Chunk 0...
Dirty: 0
Address: 944
Size in bytes: 32
Message 0...
Message ID: 0x0011 stab(0)
Shared message: No
Constant: Yes
Raw size in obj header: 16 bytes
Chunk number: 0
Message Information:
B-tree address: 384
Name heap address: 96
Message 1...
Message ID: 0x0000 null(0)
Shared message: No
Constant: No
Raw size in obj header: 0 bytes
Chunk number: 0
Message Information:
<no info="" for="" this="" message="">
**Root Group Local Heap** byte  | byte  | byte  | byte   
---|---|---|---  
'H'  | 'E'  | 'A'  | 'P'   
0   
256   
16   
128   
%h5debug example2.h5 96
New address: 96
Reading signature at address 96 (rel)
Local Heap...
Dirty: 0
Header size (in bytes): 32
Address of heap data: 128
Data bytes allocated on disk: 256
Data bytes allocated in core: 256
Free Blocks (offset, size):
Block #0: 16, 240
Percent of heap used: 6.25%
Data follows (`__' indicates free region)...
0: 00 00 00 00 00 00 00 00 44 61 74 61 73 65 74 00 ........Dataset.
16: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
32: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
48: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
64: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
80: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
96: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
112: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
128: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
144: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
160: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
176: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
192: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
208: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
224: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
240: __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
**Root Group B-tree** byte  | byte  | byte  | byte   
---|---|---|---  
'T'  | 'R'  | 'E'  | 'E'   
0  | 0  | 1   
  
0xffffffffffffffff  
  
  
  
0xffffffffffffffff  
  
  
  
0  
  
  
  
1248  
  
  
  
8  
  
  
%h5debug example2.h5 384 96
New address: 384
Reading signature at address 384 (rel)
Tree type ID: H5B_SNODE_ID
Size of node: 544
Size of raw (disk) key: 8
Dirty flag: False
Number of initial dirty children: 0
Level: 0
Address of left sibling: UNDEF
Address of right sibling: UNDEF
Number of children (max): 1 (32)
Child 0...
Address: 1248
Left Key:
Heap offset: 0
Name :
Right Key:
Heap offset: 8
Name : Dataset
**Root Group B-tree Symbol Table Node** byte  | byte  | byte  | byte   
---|---|---|---  
'S'  | 'N'  | 'O'  | 'D'   
1  | 0  | 1   
| 8   
---  
  
976  
  
  
0   
0   
  
  
0  
  
  
  
%h5debug example2.h5 1248 96
New address: 1248
Reading signature at address 1248 (rel)
Symbol Table Node...
Dirty: No
Size of Node (in bytes): 328
Number of Symbols: 1 of 8
Symbol 0:
Name: `Dataset'
Name offset into private heap: 8
Object header address: 976
Dirty: No
Cache info type: Nothing Cached
**'/Dataset' Object Header** byte  | byte  | byte  | byte   
---|---|---|---  
_Version:_ 1  |  _Reserved:_ 0  |  _Number of Header Messages:_ 6   
_Object Reference Count:_ 1   
_Total Object Header Size:_ 256   
_Fill Value Header Message_ |  |  |  |   
---|---|---|---  
_Message Type:_ 0x0005 |  _Message Data Size:_ 8   
_Flags:_ 0x01 |  _Reserved:_ 0   
_Version:_ 1 |  _Space Allocation Time:_ 2 (Late) |  _Fill Value Writing Time:_ 0 (At allocation) |  _Fill Value Defined:_ 0 (Undefined)   
_Fill Value Datatype Size:_ 0 (Use dataset's datatype for fill-value datatype)   
_Datatype Header Message_ |  |  |  |   
---|---|---|---  
_Message Type:_ 0x0003 |  _Message Data Size:_ 16   
_Flags:_ 0x01 |  _Reserved:_ 0   
|  _Version:_ 0x1  |  _Class:_ 0x0 (Fixed-Point)   
---|---  
_Fixed-Point Bit-Field:_ 0x08 (Little-endian, No padding, Signed)   
_Size:_ 4   
_Bit Offset:_ 0 |  _Bit Precision:_ 32   
_Message Alignment Filler:_ -   
_Dataspace Header Message_ |  |  |  |   
---|---|---|---  
_Message Type:_ 0x0001 |  _Message Data Size:_ 8   
_Flags:_ 0x00 |  _Reserved:_ 0   
_Version:_ 1 |  _Rank:_ 0 (Scalar) |  _Flags:_ 0x00 (No maximum dimensions, no permutation information) |  _Reserved:_ 0   
_Reserved:_ 0   
_Layout Header Message_ |  |  |  |   
---|---|---|---  
_Message Type:_ 0x0008 |  _Message Data Size:_ 24   
_Flags:_ 0x00 |  _Reserved:_ 0   
_Version:_ 1 |  _Rank:_ 1 (Dataspace rank+1) |  _Class:_ 1 (Contiguous) |  _Reserved:_ 0   
_Reserved:_ 0   
  
_Address:_ 0xffffffffffffffff (Undefined)  
  
  
_Dimension 0 Size:_ 4 (Datatype size)   
_Message Alignment Filler:_ -   
_Modification Date & Time Header Message_ |  |  |  |   
---|---|---|---  
_Message Type:_ 0x0012 |  _Message Data Size:_ 8   
_Flags:_ 0x00 |  _Reserved:_ 0   
_Version:_ 1 |  _Reserved:_ 0   
_Seconds Since Epoch:_ 1052401700 (2003-05-08 08:48:20 CDT)   
_Null Header Message_ |  |  |  |   
---|---|---|---  
_Message Type:_ 0x0000  |  _Message Data Size:_ 144   
_Flags:_ 0x00  |  _Reserved:_ 0   
%h5debug example2.h5 976
New address: 976
Reading signature at address 976 (rel)
Object Header...
Dirty: 0
Version: 1
Header size (in bytes): 16
Number of links: 1
Number of messages (allocated): 6 (32)
Number of chunks (allocated): 1 (8)
Chunk 0...
Dirty: 0
Address: 992
Size in bytes: 256
Message 0...
Message ID (sequence number): 0x0005 `fill_new' (0)
Shared: No
Constant: Yes
Raw size in obj header: 8 bytes
Chunk number: 0
Message Information:
Version: 1
Space Allocation Time: Late
Fill Time: On Allocation
Fill Value Defined: Undefined
Size: 0
Data type: <dataset type="">
Message 1...
Message ID (sequence number): 0x0003 data_type(0)
Shared message: No
Constant: Yes
Raw size in obj header: 16 bytes
Chunk number: 0
Message Information:
Type class: integer
Size: 4 bytes
Byte order: little endian
Precision: 32 bits
Offset: 0 bits
Low pad type: zero
High pad type: zero
Sign scheme: 2's comp
Message 2...
Message ID (sequence number): 0x0001 simple_dspace(0)
Shared message: No
Constant: No
Raw size in obj header: 8 bytes
Chunk number: 0
Message Information:
Rank: 0
Message 3...
Message ID (sequence number): 0x0008 layout(0)
Shared message: No
Constant: No
Raw size in obj header: 24 bytes
Chunk number: 0
Message Information:
Data address: UNDEF
Number of dimensions: 1
Size: {4}
Message 4...
Message ID (sequence number): 0x0012 mtime_new(0)
Shared message: No
Constant: No
Raw size in obj header: 8 bytes
Chunk number: 0
Message Information:
Time: 2003-03-05 14:52:00 CST
Message 5...
Message ID (sequence number): 0x0000 null(0)
Shared message: No
Constant: No
Raw size in obj header: 144 bytes
Chunk number: 0
Message Information:
<no info="" for="" this="" message="">
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


