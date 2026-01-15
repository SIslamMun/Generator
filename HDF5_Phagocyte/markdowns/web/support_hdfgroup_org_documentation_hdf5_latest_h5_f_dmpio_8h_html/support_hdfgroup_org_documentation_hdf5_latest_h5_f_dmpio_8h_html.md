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
  * [ Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#title1 "H2")
  * [ Variables](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#title2 "H2")
  * [Macro Definition Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#title3 "H2")
  * [◆ H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#title4 "H2")
  * [Variable Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#title5 "H2")
  * [◆ H5FD_mpi_opt_types_g](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#title6 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Macros](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#define-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#func-members) | [Variables](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#var-members)
H5FDmpio.h File Reference
`#include "H5FDpublic.h[](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dpublic_8h.html)"`  

##  Macros  
---  
#define  |  [H5FD_MPIO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf) ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_MPIO_id_g)  
##  Functions  
---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga3f1295b2ee6dcb59288ac885c5562aa8) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [H5FD_mpio_xfer_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16) *xfer_mode)  
| Returns the data transfer mode.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pget_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga01e8ad4d6c343c84e46a423bcee5521a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, MPI_Comm *comm, MPI_Info *info)  
| Returns MPI IO communicator information.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [H5FD_mpio_xfer_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16) xfer_mode)  
| Sets data transfer mode.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_dxpl_mpio_chunk_opt](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga677259e85909add92495621fcad64e3a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [H5FD_mpio_chunk_opt_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#aad0ca234af678f0f0976d810ee2bf246) opt_mode)  
| Sets a flag specifying linked-chunk I/O or multi-chunk I/O.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_dxpl_mpio_chunk_opt_num](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga498f3dba540bf54d1e6ab3d9f92231f1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, unsigned num_chunk_per_proc)  
| Sets a numeric threshold for linked-chunk I/O.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_dxpl_mpio_chunk_opt_ratio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#gafdcd34a6dc7a576f8a697318ad3d7b96) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, unsigned percent_num_proc_per_chunk)  
| Sets a ratio threshold for collective I/O.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_dxpl_mpio_collective_opt](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#gadd80f197d0e03841c5e7f0f4f02d4103) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [H5FD_mpio_collective_opt_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#afaf7d5667632176e8daca47549e29fb8) opt_mode)  
| Sets low-level data transfer mode.   
  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id, MPI_Comm comm, MPI_Info info)  
| Stores MPI IO communicator information to the file access property list.   
  
##  Variables  
---  
bool  | [H5FD_mpi_opt_types_g](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#ae3d5d932c685a5db933f5ce632876861)  
## Macro Definition Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#a7a231bc1d78744088a4e1d297284cabf)H5FD_MPIO
#define H5FD_MPIO ([H5OPEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#abb549ba9fa5c4f3107d8ff19602705da) H5FD_MPIO_id_g)  
---  
ID for the mpio VFD 
## Variable Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html#ae3d5d932c685a5db933f5ce632876861)H5FD_mpi_opt_types_g
| bool H5FD_mpi_opt_types_g  
---  
extern  
  * [src](https://support.hdfgroup.org/documentation/hdf5/latest/dir_68267d1309a1af8e8297ef4c3efbcdba.html)
  * [H5FDmpio.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpio_8h.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


