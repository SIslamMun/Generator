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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title0 "H1")
  * [ Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title1 "H1")
  * [ Multi-dataset I/O support](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title2 "H1")
  * [ Incremental file space allocation support](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title3 "H1")
  * [ Performance Considerations](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title4 "H1")
  * [ Begin with a good chunking strategy](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title5 "H2")
  * [ Avoid chunk sharing](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title6 "H2")
  * [ Collective metadata operations](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title7 "H2")
  * [ Align chunks in the file](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title8 "H2")
  * [ File free space managers](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title9 "H2")
  * [ Low-level collective vs. independent I/O](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title10 "H2")
  * [ Runtime HDF5 Library version](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Parallel Compression
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Introduction
When an HDF5 dataset is created, the application can specify optional data filters to be applied to the dataset (as long as the dataset uses a chunked data layout). These filters may perform compression, shuffling, checksumming/error detection and more on the dataset data. The filters are added to a filter pipeline for the dataset and are automatically applied to the data during dataset writes and reads.
Prior to the HDF5 1.10.2 release, a parallel HDF5 application could read datasets with filters applied to them, but could not write to those datasets in parallel. The datasets would have to first be written in a serial HDF5 application or from a single MPI rank in a parallel HDF5 application. This restriction was in place because:
  * Updating the data in filtered datasets requires management of file metadata, such as the dataset's chunk index and file space for data chunks, which must be done collectively in order for MPI ranks to have a consistent view of the file. At the time, HDF5 lacked the collective coordination of this metadata management.
  * When multiple MPI ranks are writing independently to the same chunk in a dataset (even if their selected portions of the chunk don't overlap), the whole chunk has to be read, unfiltered, modified, re-filtered and then written back to disk. This read-modify-write style of operation would cause conflicts among the MPI ranks and lead to an inconsistent view of the file.


Introduced in the HDF5 1.10.2 release, the parallel compression feature allows an HDF5 application to write in parallel to datasets with filters applied to them, as long as collective I/O is used. The feature introduces new internal infrastructure that coordinates the collective management of the file metadata between MPI ranks during dataset writes. It also accounts for multiple MPI ranks writing to a chunk by assigning ownership to one of the MPI ranks, at which point the other MPI ranks send their modifications to the owning MPI rank.
The parallel compression feature is always enabled when HDF5 is built with parallel enabled, but the feature may be disabled if the necessary MPI-3 routines are not available. Therefore, HDF5 conditionally defines the macro `H5_HAVE_PARALLEL_FILTERED_WRITES` which an application can check for to see if the feature is available.
#  Examples
Using the parallel compression feature is very similar to using compression in serial HDF5, except that dataset writes **must** be collective:
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d));
[H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5)(dxpl_id, [H5FD_MPIO_COLLECTIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a75d4dc80546ad3c16d2d7647ab267fab));
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(..., dxpl_id, ...);
[H5FD_MPIO_COLLECTIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a75d4dc80546ad3c16d2d7647ab267fab)
@ H5FD_MPIO_COLLECTIVE
**Definition** H5FDmpi.h:38
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d)
#define H5P_DATASET_XFER
**Definition** H5Ppublic.h:68
[H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5)
herr_t H5Pset_dxpl_mpio(hid_t dxpl_id, H5FD_mpio_xfer_t xfer_mode)
Sets data transfer mode.
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
The following are two simple examples of using the parallel compression feature:
[ph5_filtered_writes.c](https://github.com/HDFGroup/hdf5/blob/develop/HDF5Examples/C/H5PAR/ph5_filtered_writes.c)
[ph5_filtered_writes_no_sel.c](https://github.com/HDFGroup/hdf5/blob/develop/HDF5Examples/C/H5PAR/ph5_filtered_writes_no_sel.c)
The former contains simple examples of using the parallel compression feature to write to compressed datasets, while the latter contains an example of how to write to compressed datasets when one or MPI ranks don't have any data to write to a dataset. Remember that the feature requires these writes to use collective I/O, so the MPI ranks which have nothing to contribute must still participate in the collective write call.
#  Multi-dataset I/O support
The parallel compression feature is supported when using the multi-dataset I/O API routines ([H5Dwrite_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaf6213bf3a876c1741810037ff2bb85d8 "Writes raw data from a set buffers to a set of datasets.")/[H5Dread_multi](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8eb1c838aff79a17de385d0707709915 "Reads raw data from a set of datasets into the provided buffers.")), but the following should be kept in mind:
  * Parallel writes to filtered datasets **must** still be collective, even when using the multi-dataset I/O API routines
  * When the multi-dataset I/O API routines are passed a mixture of filtered and unfiltered datasets, the library currently has to perform I/O on them separately in two phases. Since there is some slight complexity involved in this, it may be best (depending on the number of datasets, number of selected chunks, number of filtered vs. unfiltered datasets, etc.) to make two individual multi-dataset I/O calls, one for the filtered datasets and one for the unfiltered datasets. When performing writes to the datasets, this would also allow independent write access to the unfiltered datasets if desired, while still performing collective writes to the filtered datasets.


#  Incremental file space allocation support
HDF5's file space allocation time function. [H5Pset_alloc_time](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga85faefca58387bba409b65c470d7d851 "Sets the timing for storage space allocation."), is a dataset creation property that can have significant effects on application performance, especially if the application uses parallel HDF5. In a serial HDF5 application, the default file space allocation time for chunked datasets is **incremental**. This means that allocation of space in the HDF5 file for data chunks is deferred until data is first written to those chunks. In parallel HDF5, the file space allocation time was previously always forced to **early** , which allocates space in the file for all of a dataset's data chunks at creation time (or during the first open of a dataset if it was created serially). This would ensure that all the necessary file space was allocated so MPI ranks could perform independent I/O operations on a dataset without needing further coordination of file metadata as described previously.
While this strategy has worked in the past, it has some noticeable drawbacks. For one, the larger the chunked dataset being created, the more noticeable overhead there will be during dataset creation as all of the data chunks are being allocated in the HDF5 file. Further, these data chunks will, by default, be filled, using [H5Pset_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga4335bb45b35386daa837b4ff1b9cd4a4 "Sets the fill value for a dataset."), with HDF5's default fill data value, leading to extraordinary dataset creation overhead and resulting in pre-filling large portions of a dataset that the application might have been planning to overwrite anyway. Even worse, there will be more initial overhead from compressing that fill data before writing it out, only to have it read back in, unfiltered and modified the first time a chunk is written to. In the past, it was typically suggested that parallel HDF5 applications should use [H5Pset_fill_time](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga6bd822266b31f86551a9a1d79601b6a2 "Sets the time when fill values are written to a dataset.") with a value of [H5D_FILL_TIME_NEVER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_dpublic_8h.html#aa39293626c4e68dd28b06c0dc84bde4aaa87fbf4f3ebf96f2f3effe7bf46c1528) in order to disable writing of the fill value to dataset chunks, but this isn't ideal if the application actually wishes to make use of fill values.
With [improvements made](https://www.hdfgroup.org/2022/03/04/parallel-compression-improvements-in-hdf5-1-13-1/) to the parallel compression feature for the HDF5 1.14.0 release, **incremental** file space allocation is now the default for datasets created in parallel _only if they have filters applied to them_. **Early** file space allocation is still supported for these datasets if desired and is still forced for datasets created in parallel that do _not_ have filters applied to them. This change should significantly reduce the overhead of creating filtered datasets in parallel HDF5 applications and should be helpful to applications that wish to use a fill value for these datasets. It should also help significantly reduce the size of the HDF5 file, as file space for the data chunks is allocated as needed rather than all at once.
#  Performance Considerations
Since getting good performance out of HDF5's parallel compression feature involves several factors, the following is a list of performance considerations (generally from most to least important) and best practices to take into account when trying to get the optimal performance out of the parallel compression feature.
##  Begin with a good chunking strategy
Starting with a good [Chunking in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/hdf5_chunking.html) strategy will generally have the largest impact on overall application performance. The different chunking parameters can be difficult to fine-tune, but it is essential to start with a well-performing chunking layout before adding compression and parallel I/O into the mix. Compression itself adds overhead and may have side effects that necessitate further adjustment of the chunking parameters and HDF5 application settings. Consider that the chosen chunk size becomes a very important factor when compression is involved, as data chunks have to be completely read and re-written to perform partial writes to the chunk.
[Improving I/O Performance When Working with HDF5 Compressed Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/improve_compressed_perf.html) is a useful reference for more information on getting good performance when using a chunked dataset layout.
##  Avoid chunk sharing
Since the parallel compression feature has to assign ownership of data chunks to a single MPI rank in order to avoid the previously described read-modify-write issue, an HDF5 application may need to take care when determining how a dataset will be divided up among the MPI ranks writing to it. Each dataset data chunk that is written to by more than 1 MPI rank will incur extra MPI overhead as one of the ranks takes ownership and the other ranks send it their data and information about where in the chunk that data belongs. While not always possible to do, an HDF5 application will get the best performance out of parallel compression if it can avoid writing in a way that causes more than 1 MPI rank to write to any given data chunk in a dataset.
##  Collective metadata operations
The parallel compression feature typically works with a significant amount of metadata related to the management of the data chunks in datasets. In initial performance results gathered from various HPC machines, it was found that the parallel compression feature did not scale well at around 8k MPI ranks and beyond. On further investigation, it became obvious that the bottleneck was due to heavy filesystem pressure from the metadata management for dataset data chunks as they changed size (as a result of data compression) and moved around in the HDF5 file.
Enabling collective metadata operations in the HDF5 application (as in the below snippet) showed significant improvement in performance and scalability and is generally always recommended unless application performance shows negative benefits by doing so.
...
hid_t fapl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
[H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1)(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
[H5Pset_all_coll_metadata_ops](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gac45976cb82e4d3f085f191e5872d0316)(fapl_id, 1);
[H5Pset_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca)(fapl_id, 1);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("file.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl_id);
...
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e)
#define H5P_FILE_ACCESS
**Definition** H5Ppublic.h:56
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Pset_coll_metadata_write](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga0163bef7ee102029286830c2c30162ca)
herr_t H5Pset_coll_metadata_write(hid_t plist_id, bool is_collective)
Sets metadata write mode to be collective or independent (default)
[H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1)
herr_t H5Pset_fapl_mpio(hid_t fapl_id, MPI_Comm comm, MPI_Info info)
Stores MPI IO communicator information to the file access property list.
[H5Pset_all_coll_metadata_ops](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_a_p_l.html#gac45976cb82e4d3f085f191e5872d0316)
herr_t H5Pset_all_coll_metadata_ops(hid_t plist_id, bool is_collective)
Sets metadata I/O mode for read operations to be collective or independent (default)
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
##  Align chunks in the file
The natural layout of an HDF5 file may cause dataset data chunks to end up at addresses in the file that do not align well with the underlying file system, possibly leading to poor performance. As an example, Lustre performance is generally good when writes are aligned with the chosen stripe size. The HDF5 application can use [H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.") to have a bit more control over where objects in the HDF5 file end up. However, do note that setting the alignment of objects generally wastes space in the file and has the potential to dramatically increase its resulting size, so caution should be used when choosing the alignment parameters.
[H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.") has two parameters that control the alignment of objects in the HDF5 file, the "threshold" value and the alignment value. The threshold value specifies that any object greater than or equal in size to that value will be aligned in the file at addresses which are multiples of the chosen alignment value. While the value 0 can be specified for the threshold to make every object in the file be aligned according to the alignment value, this isn't generally recommended, as it will likely waste an excessive amount of space in the file.
In the example below, the chosen dataset chunk size is provided for the threshold value and 1MiB is specified for the alignment value. Assuming that 1MiB is an optimal alignment value (e.g., assuming that it matches well with the Lustre stripe size), this should cause dataset data chunks to be well-aligned and generally give good write performance.
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fapl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
[H5Pset_fapl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga7519d659a83ef5717d7a5d95baf0e9b1)(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
/* Assuming Lustre stripe size is 1MiB, align data chunks
in the file to address multiples of 1MiB. */
[H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a)(fapl_id, dataset_chunk_size, 1048576);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("file.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl_id);
[H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a)
herr_t H5Pset_alignment(hid_t fapl_id, hsize_t threshold, hsize_t alignment)
Sets alignment properties of a file access property list.
##  File free space managers
As data chunks in a dataset get written to and compressed, they can change in size and be relocated in the HDF5 file. Since parallel compression usually involves many data chunks in a file, this can create significant amounts of free space in the file over its lifetime and eventually cause performance issues.
An HDF5 application can use [H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l...") with a value of [H5F_FSPACE_STRATEGY_PAGE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a9cc492c4b5c936e48716a8dab3691bccacd625bd864903e71132c9098929f5a0a) to enable the paged aggregation feature, which can accumulate metadata and raw data for dataset data chunks into well-aligned, configurably sized **pages** for better performance. However, note that using the paged aggregation feature will cause any setting from [H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list.") to be ignored. While an application should be able to get comparable performance effects by setting the size of these pages, using [H5Pset_file_space_page_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gad012d7f3c2f1e1999eb1770aae3a4963 "Sets the file space page size for a file creation property list."), to be equal to the value that would have been set for [H5Pset_alignment](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gab99d5af749aeb3896fd9e3ceb273677a "Sets alignment properties of a file access property list."), this may not necessarily be the case and should be studied.
Note that [H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e "Sets the file space handling strategy and persisting free-space values for a file creation property l...") has a **persist** parameter. This determines whether or not the file free space manager should include extra metadata in the HDF5 file about free space sections in the file. If this parameter is **false** , any free space in the HDF5 file will become unusable once the HDF5 file is closed. For parallel compression, it's generally recommended that **persist** be set to **true** , as this will keep better track of file free space for data chunks between accesses to the HDF5 file.
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977));
/* Use persistent free space manager with paged aggregation */
[H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e)(fcpl_id, [H5F_FSPACE_STRATEGY_PAGE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a9cc492c4b5c936e48716a8dab3691bccacd625bd864903e71132c9098929f5a0a), 1, 1);
/* Assuming Lustre stripe size is 1MiB, set page size to that */
[H5Pset_file_space_page_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gad012d7f3c2f1e1999eb1770aae3a4963)(fcpl_id, 1048576);
...
hid_t file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("file.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), fcpl_id, fapl_id);
[H5F_FSPACE_STRATEGY_PAGE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a9cc492c4b5c936e48716a8dab3691bccacd625bd864903e71132c9098929f5a0a)
@ H5F_FSPACE_STRATEGY_PAGE
**Definition** H5Fpublic.h:198
[H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977)
#define H5P_FILE_CREATE
**Definition** H5Ppublic.h:52
[H5Pset_file_space_page_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gad012d7f3c2f1e1999eb1770aae3a4963)
herr_t H5Pset_file_space_page_size(hid_t plist_id, hsize_t fsp_size)
Sets the file space page size for a file creation property list.
[H5Pset_file_space_strategy](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html#gae9ed9b56f290d6d24421242f1c04914e)
herr_t H5Pset_file_space_strategy(hid_t plist_id, H5F_fspace_strategy_t strategy, bool persist, hsize_t threshold)
Sets the file space handling strategy and persisting free-space values for a file creation property l...
##  Low-level collective vs. independent I/O
While the parallel compression feature requires that the HDF5 application set and maintain collective I/O at the application interface level (via [H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5 "Sets data transfer mode.")), it does not require that the actual MPI I/O that occurs at the lowest layers of HDF5 be collective; independent I/O may perform better depending on the application I/O patterns and parallel file system performance, among other factors. The application may use [H5Pset_dxpl_mpio_collective_opt](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#gadd80f197d0e03841c5e7f0f4f02d4103 "Sets low-level data transfer mode.") to control this setting and see which I/O method provides the best performance.
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d));
[H5Pset_dxpl_mpio](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga22837d8504dc1f87f175b46b348ce0e5)(dxpl_id, [H5FD_MPIO_COLLECTIVE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#a99bc5a964089fea144e7056b004bcc16a75d4dc80546ad3c16d2d7647ab267fab));
[H5Pset_dxpl_mpio_collective_opt](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#gadd80f197d0e03841c5e7f0f4f02d4103)(dxpl_id, [H5FD_MPIO_INDIVIDUAL_IO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#afaf7d5667632176e8daca47549e29fb8aaacd91139f703159fde84fb5f7778886)); /* Try independent I/O */
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(..., dxpl_id, ...);
[H5FD_MPIO_INDIVIDUAL_IO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dmpi_8h.html#afaf7d5667632176e8daca47549e29fb8aaacd91139f703159fde84fb5f7778886)
@ H5FD_MPIO_INDIVIDUAL_IO
**Definition** H5FDmpi.h:51
[H5Pset_dxpl_mpio_collective_opt](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#gadd80f197d0e03841c5e7f0f4f02d4103)
herr_t H5Pset_dxpl_mpio_collective_opt(hid_t dxpl_id, H5FD_mpio_collective_opt_t opt_mode)
Sets low-level data transfer mode.
##  Runtime HDF5 Library version
An HDF5 application can use the [H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.") routine to set the upper and lower bounds on library versions to use when creating HDF5 objects. For parallel compression specifically, setting the library version to the latest available version can allow access to better/more efficient chunk indexing types and data encoding methods. For example:
...
hid_t fapl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e));
[H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)(fapl_id, [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0), [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_id = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)("file.h5", [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), fapl_id);
...
[H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0)
@ H5F_LIBVER_LATEST
**Definition** H5Fpublic.h:187
[H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910)
herr_t H5Pset_libver_bounds(hid_t plist_id, H5F_libver_t low, H5F_libver_t high)
Controls the range of library release versions used when creating objects in a file.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


