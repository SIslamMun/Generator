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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_i_o_f_l_o_w.html#title0 "H1")
  * [ Notes From Accompanying Figures](https://support.hdfgroup.org/documentation/hdf5/latest/_i_o_f_l_o_w.html#title1 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Raw I/O Flow Notes
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
  * Created by Quincey Koziol, August 20, 2003 
  * Document's Audience: Current [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html) library designers and knowledgeable external developers.


#  Introduction
What is this document about?  
This document attempts to supplement the flow charts describing the flow of control for raw data I/O in the library. The following figures provide the main information: 
![High-Level View of Writing Raw Data](https://support.hdfgroup.org/documentation/hdf5/latest/IOFlow.gif)  
---  
![Perform Serial or Parallel I/O](https://support.hdfgroup.org/documentation/hdf5/latest/IOFlow2.gif)  
![Gather/Convert/Scatter](https://support.hdfgroup.org/documentation/hdf5/latest/IOFlow3.gif)  
#  Notes From Accompanying Figures
This section provides notes to augment the information in the accompanying figures.
  1. **Validate Parameters** - Resolve any H5S_ALL parameters for dataspace selections to actual dataspaces, allocate conversion buffers, etc. 
  2. **Space Allocated in File?** - Space may not have been allocated in the file to store the dataset data, if "late allocation" was chosen for the allocation time when the dataset was created. 
  3. **Allocate & Fill Space** - These operations allocate both contiguous and chunked dataset's space in the file. The chunked dataset space allocation iterates through all the chunks in the file and allocates both the B-tree information and the raw data in the file. Because of the way filters work, fill-values are written out for chunked datasets as they are allocated, instead of as a separate step. In parallel I/O, the chunked dataset allocation can potentially be time-consuming, since all the raw data in the dataset is allocated from one process. 
  4. **Datatype Conversion Needed?** - This currently is the deciding factor between doing "direct I/O" (in serial or parallel) and needing to perform gather/convert/scatter operations. I believe that MPI is capable of performing a limited range of type conversions and if so, we should add support to detect when they can be used. This will allow more I/O operations to be performed collectively. 
  5. **Collective I/O Requested/Allowed?** - A user has to both request that collective I/O occur and also their I/O operation must meet the requirements that the library sets for supporting collective parallel I/O: 
     * The dataspace must be scalar or simple (which is a no-op really, since we don't support "complex" dataspaces in the library currently). 
     * The selection must be regular. "all" selections and hyperslab selections that were made with only one call to [H5Sselect_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.") (i.e. not a hyperslab selection that has been aggregated over multiple selection calls) are regular. Supporting point and irregular hyperslab selections are on the "to do" list. 
     * The dataset must be stored contiguously on disk (as shown in the figure also). Supporting chunked dataset storage is also on the "to do" list. 
  6. **Build "chunk map"** - This step still has some scalability issues as it creates a data structure that is proportional to the number of chunks which will be written to, which could potentially be very large. Building the "chunk map" information incrementally is on the "to do" list also. 
  7. **Perform Chunked I/O** - As the figure shows, there is no support for collective parallel I/O on chunked datasets currently. As noted earlier, this is on the "to do" list. 
  8. **Perform "Direct" Serial I/O** - "Direct" serial I/O writes data from the application's buffer, without any intervening buffer or memory copies. For maximum efficiency and performance, the elements in the selections should be adjoining. 
  9. **Perform Collective Parallel I/O** - This step also writes data directly from an application buffer, but additionally uses collective MPI I/O operations to combine the data from each process in the parallel application in an efficient manner. 


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


