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
  * [ The HDF5 Virtual File Driver Interface](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title1 "H2")
  * [ Purpose and Benefits](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title2 "H2")
  * [ Built-in File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title3 "H2")
  * [ Selecting a File Driver](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title4 "H2")
  * [ Custom File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title5 "H2")
  * [ Parallel File Drivers](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title6 "H2")
  * [ Performance Considerations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title7 "H2")
  * [ Querying File Driver Information](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title8 "H2")
  * [ Summary](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Virtual File Drivers
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  The HDF5 Virtual File Driver Interface
##  Introduction
The HDF5 Virtual File Driver (VFD) interface provides an abstraction layer for file I/O operations, enabling HDF5 to work with different file storage mechanisms. The VFD layer intercepts all low-level file access operations and forwards them to a specific driver implementation, allowing HDF5 files to be stored in various ways beyond simple POSIX files. 

See also
     [File Drivers (H5FD)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f_d.html) Reference Manual
##  Purpose and Benefits
The Virtual File Driver interface serves several important purposes:
  * **Storage Flexibility** : Enables HDF5 to work with different storage backends including local files, parallel file systems, memory, and cloud storage.


  * **Performance Optimization** : Allows selection of file drivers optimized for specific computing environments and I/O patterns.


  * **Parallel I/O** : Provides support for MPI-based parallel I/O through specialized drivers.


  * **Custom Storage** : Enables development of custom file drivers for specialized storage requirements.


##  Built-in File Drivers
HDF5 includes several standard Virtual File Drivers:
  * **SEC2 Driver** : The default POSIX I/O driver using standard system calls like read() and write(). Suitable for most serial applications on local file systems. Set with [H5Pset_fapl_sec2](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga737e9a4df38fafbe672b4b989851b3db "Modifies the file access property list to use the H5FD_SEC2 driver.").


  * **STDIO Driver** : Uses buffered I/O from the C standard library (fread/fwrite). May provide better performance for some applications. Set with [H5Pset_fapl_stdio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga77f0643117835e7f7992d573761b5052 "Sets the standard I/O driver.").


  * **Core Driver** : Stores the HDF5 file entirely in memory, with optional backing store to disk. Provides fastest I/O for temporary files or small datasets. Set with [H5Pset_fapl_core](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac9f8c916dbe7b5e4ef23e9a7b15f370f "Modifies the file access property list to use the H5FD_CORE driver.").


  * **Family Driver** : Splits a logical HDF5 file across multiple physical files of equal size. Useful for circumventing file system limitations. Set with [H5Pset_fapl_family](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga045ab2235f949a47bbb83e1b4c7e6e4d "Sets the file access property list to use the family driver.").


  * **Multi Driver** : Stores different types of HDF5 data in separate files (metadata, raw data, etc.). Can optimize I/O by placing different data types on different storage devices. Set with [H5Pset_fapl_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga5d9096f3413a7b66b28afcbf15d05396 "Sets up use of the multi-file driver.").


  * **Split Driver** : A simplified version of the Multi driver that separates metadata and raw data into two files. Set with [H5Pset_fapl_split](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gadf6f57e56463fce641cf07a586848175 "Emulates the old split file driver.").


  * **Log Driver** : Wraps another driver and logs all file access operations. Useful for debugging and I/O profiling. Set with [H5Pset_fapl_log](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac62545cff97df64a3a4a7377e161ce8d "Sets up the logging virtual file driver \(H5FD_LOG\) for use.").


  * **MPI-IO Driver** : Enables parallel I/O using MPI-IO for HPC applications. Required for parallel HDF5 operations. Set with [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.").


  * **Subfiling Driver** : A parallel I/O driver that improves parallel I/O performance on parallel file systems by splitting the logical HDF5 file into multiple subfiles distributed across I/O concentrator nodes. Reduces contention and improves scalability for large-scale parallel applications. Set with [H5Pset_fapl_subfiling](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga65dbddbba216fdd0bdf99b4feaa74db2 "Modifies the specified File Access Property List to use the H5FD_SUBFILING driver.").


  * **Direct Driver** : Uses direct I/O (O_DIRECT) to bypass OS caching. Can improve performance for large sequential I/O. Set with [H5Pset_fapl_direct](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4525295096d292c48459b44574ba858c "Sets up use of the direct I/O driver.").


  * **Onion Driver** : Provides revision control for HDF5 files by storing file modifications as separate revisions. Enables tracking changes over time and accessing previous versions. Set with [H5Pset_fapl_onion](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gade8995b5e2d1ea53861501d98129d762 "set the onion info for the file access property list").


  * **Splitter Driver** : Writes file operations simultaneously to two different channels using different VFDs. Useful for creating redundant copies or logging I/O to separate locations. Set with [H5Pset_fapl_splitter](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacba378b6c9258d33da609ff7241d64c5 "Sets the file access property list to use the splitter driver.").


  * **Mirror Driver** : Mirrors all file operations to a remote server in real-time over a network connection. Enables remote backup and replication scenarios. Set with [H5Pset_fapl_mirror](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4fa5d96d399dcecba277bb485e3eb4cf "Modifies the file access property list to use the H5FD_MIRROR driver.").


  * **ROS3 Driver** : Read-only driver for accessing HDF5 files in S3-compatible object storage. Set with [H5Pset_fapl_ros3](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.").


  * **HDFS Driver** : Read-only driver for accessing HDF5 files in Hadoop Distributed File System. Set with [H5Pset_fapl_hdfs](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacc14b36c543d837e3257c216939d8931 "Modifies the file access property list to use the H5FD_HDFS driver.").


##  Selecting a File Driver
File drivers are selected through the file access property list when opening or creating a file. The basic pattern is:
  * Create a file access property list with [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa "Creates a new property list as an instance of a property list class.")
  * Set the desired file driver using the appropriate H5Pset_fapl_* function 
  * Pass the property list to [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") or [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc "Opens an existing HDF5 file.")


##  Custom File Drivers
Applications can implement custom file drivers by:
  * Defining a [H5FD_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__class__t.html) structure with function pointers for all required operations 
  * Implementing the driver callbacks (open, close, read, write, etc.) 
  * Registering the driver with [H5FDregister](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a576ba6b6ebc34ae422bf40700a4c372d)
  * Setting the driver in a file access property list with [H5Pset_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga8bcce60e23e9d2a019212c63b146502e "Sets a file driver.")


Custom drivers enable specialized I/O strategies such as: 
  * Integration with custom storage systems 
  * Transparent encryption or compression at the I/O layer 
  * Specialized caching strategies 
  * Network-based storage protocols


##  Parallel File Drivers
For parallel HDF5 applications, the MPI-IO file driver is required (see [A Brief Introduction to Parallel HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html) for details on parallel HDF5 programming). This driver coordinates file access across multiple MPI processes, enabling collective I/O operations and preventing conflicts. Parallel applications must:
  * Build HDF5 with parallel support enabled 
  * Use the MPI-IO file driver via [H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1 "Stores MPI IO communicator information to the file access property list.")
  * Provide MPI communicator and info objects 
  * Coordinate file access across processes


The Subfiling driver provides additional performance benefits for large-scale parallel applications on parallel file systems. It works by:
  * Distributing the HDF5 file across multiple subfiles 
  * Designating I/O concentrator processes (typically one per node) 
  * Striping data across subfiles to reduce contention 
  * Enabling better parallel I/O scaling on Lustre, GPFS, and similar file systems


The Subfiling driver is particularly beneficial when running at scale on parallel file systems where a single shared file can become a bottleneck.
##  Performance Considerations
Choosing the right file driver can significantly impact I/O performance:
  * **Local Files** : SEC2 or STDIO drivers typically provide good performance 
  * **Temporary Data** : Core driver provides fastest access by avoiding disk I/O 
  * **Large Files** : Family driver can work around file size limitations 
  * **Parallel Applications** : MPI-IO driver required for coordinated parallel access 
  * **Large-Scale Parallel** : Subfiling driver can dramatically improve performance on shared parallel file systems by reducing metadata contention and enabling better striping 
  * **Network Storage** : Consider drivers optimized for network protocols 
  * **Redundancy** : Splitter or Mirror drivers enable real-time backup and replication 
  * **Versioning** : Onion driver enables tracking file revisions for provenance and rollback


##  Querying File Driver Information
Applications can query the current file driver:
  * [H5Pget_driver](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga43a733fe9723dd15f5ad7abda909a1b8 "Returns low-lever driver identifier.") retrieves the driver identifier from a file access property list 
  * [H5FDis_driver_registered_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#af9c0b9593676b130d1ff51caad2b7413) checks if a specific driver is available by name 
  * [H5FDis_driver_registered_by_value](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_ddevelop_8h.html#a188a3d06e17aeb8551d56ed5da3f3c6b) checks if a specific driver is available by value 
  * Driver-specific property list functions retrieve driver parameters


##  Summary
The HDF5 Virtual File Driver interface provides: 
  * Abstraction of file I/O operations for flexibility and portability 
  * Multiple built-in drivers for common storage scenarios 
  * Support for parallel I/O via MPI-IO 
  * Extensibility through custom driver implementation 
  * Performance optimization opportunities through driver selection


Proper selection and configuration of file drivers is essential for optimal HDF5 performance in different computing environments.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


