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
  * [ Overview of Parallel HDF5 (PHDF5) Design](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html#title0 "H1")
  * [ Parallel Programming with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html#title1 "H2")
  * [ Creating and Accessing a File with PHDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html#title2 "H2")
  * [ Creating and Accessing a Dataset with PHDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html#title3 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
A Brief Introduction to Parallel HDF5
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
* * *
If you are new to HDF5 please see the [Learning the Basics](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_basics.html) topic first.
#  Overview of Parallel HDF5 (PHDF5) Design
There were several requirements that we had for Parallel HDF5 (PHDF5). These were: 
  * Parallel HDF5 files had to be compatible with serial HDF5 files and sharable between different serial and parallel platforms. 
  * Parallel HDF5 had to be designed to have a single file image to all processes, rather than having one file per process. Having one file per process can cause expensive post processing, and the files are not usable by different processes. 
  * A standard parallel I/O interface had to be portable to different platforms.


With these requirements of HDF5 our initial target was to support MPI programming, but not for shared memory programming. We had done some experimentation with thread-safe support for Pthreads and for OpenMP, and decided to use these.
Implementation requirements were to: 
  * Not use Threads, since they were not commonly supported in 1998 when we were looking at this. 
  * Not have a reserved process, as this might interfere with parallel algorithms. 
  * Not spawn any processes, as this is not even commonly supported now.


The following shows the Parallel HDF5 implementation layers:
![](https://support.hdfgroup.org/documentation/hdf5/latest/pimplayer.gif)  
---  
##  Parallel Programming with HDF5
This tutorial assumes that you are somewhat familiar with parallel programming with MPI (Message Passing Interface).
If you are not familiar with parallel programming, here is a tutorial that may be of interest: [Tutorial on HDF5 I/O tuning at NERSC (PDF)](https://support.hdfgroup.org/releases/hdf5/documentation/hdf5_topics/2016_NERSC_Introduction_to_Scientific_IO.pdf). (NOTE: As of 2024, the specific systems described in this tutorial are outdated.)
Some of the terms that you must understand in this tutorial are: 
  * **MPI Communicator** Allows a group of processes to communicate with each other.
Following are the MPI routines for initializing MPI and the communicator and finalizing a session with MPI: 
C  | Fortran  | Description   
---|---|---  
MPI_Init  | MPI_INIT  | Initialize MPI (MPI_COMM_WORLD usually)   
MPI_Comm_size  | MPI_COMM_SIZE  | Define how many processes are contained in the communicator   
MPI_Comm_rank  | MPI_COMM_RANK  | Define the process ID number within the communicator (from 0 to n-1)   
MPI_Finalize  | MPI_FINALIZE  | Exiting MPI   
  * **Collective** MPI defines this to mean all processes of the communicator must participate in the right order. 


Parallel HDF5 opens a parallel file with a communicator. It returns a file handle to be used for future access to the file.
All processes are required to participate in the collective Parallel HDF5 API. Different files can be opened using different communicators.
Examples of what you can do with the Parallel HDF5 collective API: 
  * File Operation: Create, open and close a file 
  * Object Creation: Create, open, and close a dataset 
  * Object Structure: Extend a dataset (increase dimension sizes) 
  * Dataset Operations: Write to or read from a dataset (Array data transfer can be collective or independent.)


Once a file is opened by the processes of a communicator: 
  * All parts of the file are accessible by all processes. 
  * All objects in the file are accessible by all processes. 
  * Multiple processes write to the same dataset. 
  * Each process writes to an individual dataset.



See also
    [Collective Calling Requirements in Parallel HDF5 Applications](https://support.hdfgroup.org/documentation/hdf5/latest/collective_calls.html)
Please refer to the Supported Configuration Features Summary in the release notes for the current release of HDF5 for an up-to-date list of the platforms that we support Parallel HDF5 on.
##  Creating and Accessing a File with PHDF5
The programming model for creating and accessing a file is as follows: 
  1. Set up an access template object to control the file access mechanism. 
  2. Open the file. 
  3. Close the file. 


Each process of the MPI communicator creates an access template and sets it up with MPI parallel access information. This is done with the [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class.") call to obtain the file access property list and the [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.") call to set up parallel I/O access.
Following is example code for creating an access template in HDF5: _C_
23 MPI_Comm comm = MPI_COMM_WORLD;
24 MPI_Info info = MPI_INFO_NULL;
25
26 /*
27 * Initialize MPI
28 */
29 MPI_Init(&argc, &argv);
30 MPI_Comm_size(comm, &mpi_size);
31 MPI_Comm_rank(comm, &mpi_rank);
32
33 /*
34 * Set up file access property list with parallel I/O access
35 */
36 plist_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e)); 37 [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1)(plist_id, comm, info);
[H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e)
#define H5P_FILE_ACCESS
**Definition** H5Ppublic.h:56
[H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1)
herr_t H5Pset_fapl_mpio(hid_t fapl_id, MPI_Comm comm, MPI_Info info)
Stores MPI IO communicator information to the file access property list.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
_Fortran_
23 comm = MPI_COMM_WORLD
24 info = MPI_INFO_NULL
25
26 CALL MPI_INIT(mpierror)
27 CALL MPI_COMM_SIZE(comm, mpi_size, mpierror)
28 CALL MPI_COMM_RANK(comm, mpi_rank, mpierror)
29 !
30 ! Initialize FORTRAN interface 
31 !
32 CALL h5open_f(error)
33
34 !
35 ! Setup file access property list with parallel I/O access.
36 !
37 CALL h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error) 38 CALL h5pset_fapl_mpio_f(plist_id, comm, info, error)
The following example programs create an HDF5 file using Parallel HDF5:
[C: ph5_file_create.c](https://github.com/HDFGroup/hdf5/blob/develop/HDF5Examples/C/H5PAR/ph5_file_create.c)
[F90: ph5_f90_file_create.F90](https://github.com/HDFGroup/hdf5/blob/develop/HDF5Examples/FORTRAN/H5PAR/ph5_f90_file_create.F90)
##  Creating and Accessing a Dataset with PHDF5
The programming model for creating and accessing a dataset is as follows: 
  1. Create or open a Parallel HDF5 file with a collective call to: [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
  2. Obtain a copy of the file transfer property list and set it to use collective or independent I/O. 
     * Do this by first passing a data transfer property list class type to: [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class.")
     * Then set the data transfer mode to either use independent I/O access or to use collective I/O, with a call to: [H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5 "Sets data transfer mode.")
Following are the parameters required by this call: _C_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, [H5FD_mpio_xfer_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16) xfer_mode )
dxpl_id IN: Data transfer property list identifier
xfer_mode IN: Transfer mode:
[H5FD_MPIO_INDEPENDENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a6c67820a8798cd75a6f0ebbb44e9a2af) - use independent I/O access
(default)
[H5FD_MPIO_COLLECTIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a75d4dc80546ad3c16d2d7647ab267fab) - use collective I/O access
[H5FD_mpio_xfer_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16)
H5FD_mpio_xfer_t
**Definition** H5FDmpi.h:36
[H5FD_MPIO_INDEPENDENT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a6c67820a8798cd75a6f0ebbb44e9a2af)
@ H5FD_MPIO_INDEPENDENT
**Definition** H5FDmpi.h:37
[H5FD_MPIO_COLLECTIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a75d4dc80546ad3c16d2d7647ab267fab)
@ H5FD_MPIO_COLLECTIVE
**Definition** H5FDmpi.h:38
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5)
herr_t H5Pset_dxpl_mpio(hid_t dxpl_id, H5FD_mpio_xfer_t xfer_mode)
Sets data transfer mode.
_Fortran_
h5pset_dxpl_mpi_f (prp_id, data_xfer_mode, hdferr)
prp_id IN: Property List Identifier (INTEGER (HID_T))
data_xfer_mode IN: Data transfer mode (INTEGER)
H5FD_MPIO_INDEPENDENT_F (0)
H5FD_MPIO_COLLECTIVE_F (1)
hdferr IN: Error code (INTEGER)
     * Access the dataset with the defined transfer property list. All processes that have opened a dataset may do collective I/O. Each process may do an independent and arbitrary number of data I/O access calls, using: [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.")
If a dataset is unlimited, you can extend it with a collective call to: [H5Dextend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gac4c0ff57977b1f39c1055296e39cbe91 "Extends a dataset.")


The following code demonstrates a collective write using Parallel HDF5: _C_
95 /*
96 * Create property list for collective dataset write.
97 */
98 plist_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa) ([H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d)); 99 [H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5) (plist_id, [H5FD_MPIO_COLLECTIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a75d4dc80546ad3c16d2d7647ab267fab));
100
101 status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) (dset_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), memspace, filespace,
102 plist_id, data);
[H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d)
#define H5P_DATASET_XFER
**Definition** H5Ppublic.h:68
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
_Fortran_
108 ! Create property list for collective dataset write
109 !
110 CALL h5pcreate_f (H5P_DATASET_XFER_F, plist_id, error) 111 CALL h5pset_dxpl_mpio_f (plist_id, H5FD_MPIO_COLLECTIVE_F, error)
112
113 !
114 ! Write the dataset collectively.
115 !
116 CALL h5dwrite_f (dset_id, H5T_NATIVE_INTEGER, data, dimsfi, error, &
117 file_space_id = filespace, mem_space_id = memspace, xfer_prp = plist_id)
The following example programs create an HDF5 dataset using Parallel HDF5:
[C: ph5_dataset.c](https://github.com/HDFGroup/hdf5/blob/develop/HDF5Examples/C/H5PAR/ph5_dataset.c)
[F90: ph5_f90_dataset.F90](https://github.com/HDFGroup/hdf5/blob/develop/HDF5Examples/FORTRAN/H5PAR/ph5_f90_dataset.F90)
###  Hyperslabs
The programming model for writing and reading hyperslabs is: /li Each process defines the memory and file hyperslabs. /li Each process executes a partial write/read call which is either collective or independent.
The memory and file hyperslabs in the first step are defined with the [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.").
The start (or offset), count, stride, and block parameters define the portion of the dataset to write to. By changing the values of these parameters you can write hyperslabs with Parallel HDF5 by contiguous hyperslab, by regularly spaced data in a column/row, by patterns, and by chunks:
  * [Writing by Contiguous Hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_cont_hyperslab.html)

  
---  
  * [Writing by Regularly Spaced Data](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_regular_spaced.html)

  
  * [Writing by Pattern](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_pattern.html)

  
  * [Writing by Chunk](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_chunk.html)

  
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


