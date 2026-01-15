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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#title0 "H1")
  * [ Direct Chunk Write Function](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#title1 "H1")
  * [ Using the Direct Chunk Write Function](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#title2 "H2")
  * [ The Design](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#title3 "H2")
  * [ Performance](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#title4 "H2")
  * [ A Word of Caution](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#title5 "H2")
  * [ A Complete Code Example](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#title6 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 High Level Optimizations
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  Introduction
The HDF5 Dataset Optimization (H5DO) interface provides high-performance functions for specialized dataset I/O operations that bypass standard HDF5 processing layers when appropriate. 

See also
     [HDF5 Optimizations APIs (H5DO)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html) Reference Manual
Since version 1.10.3 these functions are deprecated in favor of [H5Dwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") and [H5Dread_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e).
#  Direct Chunk Write Function
When a user application has a chunked dataset and is trying to write a single chunk of data with [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset."), the data goes through several steps inside the HDF5 library. The library first examines the hyperslab selection. Then it converts the data from the datatype in memory to the datatype in the file if they are different. Finally, the library processes the data in the filter pipeline. Starting with the 1.8.11 release, a new high-level C function called [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") becomes available. It writes a data chunk directly to the file bypassing the library’s hyperslab selection, data conversion, and filter pipeline processes. In other words, if an application can pre-process the data, then the application can use [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") to write the data much faster.
[H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") was developed in response to a client request. The client builds X-ray pixel detectors for use at synchrotron light sources. These detectors can produce data at the rate of tens of gigabytes per second. Before transferring the data over their network, the detectors compress the data by a factor of 10 or more. The modular architecture of the detectors can scale up its data stream in parallel and maps well to current parallel computing and storage systems. See the [Direct Chunk Write](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/DECTRIS%20Integration%20RFC%202012-11-29.pdf) for the original proposal.
##  Using the Direct Chunk Write Function
Basically, the [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") function takes a pre-processed data chunk (buf) and its size (data_size) and writes to the chunk location (offset) in the dataset ( dset_id).
The function prototype is shown below: 
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)(
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, // the dataset
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, // data transfer property list
uint32_t filter_mask, // indicates which filters are used
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)* offset, // position of the chunk
size_t data_size, // size of the actual data
const void* buf // buffer with data to be written
)
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)
H5HL_DLL herr_t H5DOwrite_chunk(hid_t dset_id, hid_t dxpl_id, uint32_t filters, const hsize_t *offset, size_t data_size, const void *buf)
Writes a raw data chunk from a buffer directly to a dataset in a file.
Below is a simple example showing how to use the function: _Example 1. Using H5DOwrite_chunk_
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) offset[2] = {4, 4};
uint32_t filter_mask = 0;
size_t nbytes = 40;
if([H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)(dset_id, dxpl, filter_mask, offset, nbytes, data_buf) < 0)
goto error;
In the example above, the dataset is 8x8 elements of int. Each chunk is 4x4. The offset of the first element of the chunk to be written is 4 and 4. In the diagram below, the shaded chunk is the data to be written. The function is writing a pre-compressed data chunk of 40 bytes (assumed) to the dataset. The zero value of the filter mask means that all filters have been applied to the pre-processed data.
![](https://support.hdfgroup.org/documentation/hdf5/latest/DOChunks_fig1.png) Figure 1. Illustration of the chunk to be written  
---  
The complete code example at the end of this topic shows how to set the value of the filter mask to indicate a filter being skipped. The corresponding bit in the filter mask is turned on when a filter is skipped. For example, if the second filter is skipped, the second bit of the filter mask should be turned on. For more information, see the [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") entry in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
##  The Design
The following diagram shows how the function [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") bypasses hyperslab selection, data conversion, and filter pipeline inside the HDF5 library.
![](https://support.hdfgroup.org/documentation/hdf5/latest/DOChunks_fig2.png) Figure 2. Diagram for H5DOwrite_chunk  
---  
##  Performance
The table below describes the results of performance benchmark tests run by HDF developers. It shows that using the new function [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") to write pre-compressed data is much faster than using the [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") function to compress and write the same data with the filter pipeline. Measurements involving [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") include compression time in the filter pipeline. Since the data is already compressed before [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") is called, use of [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") to write compressed data avoids the performance bottleneck in the HDF5 filter pipeline.
The test was run on a Linux 2.6.18 / 64-bit Intel x86_64 machine. The dataset contained 100 chunks. Only one chunk was written to the file per write call. The number of writes was 100. The time measurement was for the entire dataset with the Unix system function gettimeofday. Writing the entire dataset with one write call took almost the same amount of time as writing chunk by chunk. In order to force the system to flush the data to the file, the O_SYNC flag was used to open the file.
_Table 1. Performance result for H5DOwrite_chunk in the high-level library_
Dataset size (MB) | 95.37 | 762.94 | 2288.82   
---|---|---|---  
Size after compression (MB) | 64.14 | 512.94 | 1538.81   
Dataset dimensionality | 100x1000x250 | 100x2000x1000 | 100x2000x3000   
Chunk dimensionality | 1000x250 | 2000x1000 | 2000x3000   
Datatype | 4-byte integer | 4-byte integer | 4-byte integer   
IO speed is in MB/s and Time is in second (s). | speed1 | time2 | speed | time | speed | time   
H5Dwrite writes without compression filter | 77.27 | 1.23 | 97.02 | 7.86 | 91.77 | 24.94   
H5DOwrite_chunk writes uncompressed data | 79 | 1.21 | 95.71 | 7.97 | 89.17 | 25.67   
H5Dwrite writes with compression filter | 2.68 | 35.59 | 2.67 | 285.75 | 2.67 | 857.24   
H5DOwrite_chunk writes compressed data | 77.19 | 0.83 | 78.56 | 6.53 | 96.28 | 15.98   
Unix writes compressed data to Unix file | 76.49 | 0.84 | 95 | 5.4 | 98.59 | 15.61   
##  A Word of Caution
Since [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") writes data chunks directly in a file, developers must be careful when using it. The function bypasses hyperslab selection, the conversion of data from one datatype to another, and the filter pipeline to write the chunk. Developers should have experience with these processes before they use this function.
##  A Complete Code Example
The following is an example of using [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") to write an entire dataset by chunk. 
#include <zlib.h>
#include <math.h>
#define DEFLATE_SIZE_ADJUST(s) (ceil(((double)(s))*1.001)+12)
size_t buf_size = CHUNK_NX*CHUNK_NY*sizeof(int);
const Bytef *z_src = (const Bytef*)(direct_buf);
Bytef *z_dst; // destination buffer
uLongf z_dst_nbytes = (uLongf)DEFLATE_SIZE_ADJUST(buf_size);
uLong z_src_nbytes = (uLong)buf_size;
int aggression = 9; // Compression aggression setting
uint32_t filter_mask = 0;
size_t buf_size = CHUNK_NX*CHUNK_NY*sizeof(int);
// Create the data space
if((dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dims, maxdims)) < 0)
goto error;
// Create a new file
if((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(FILE_NAME5, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0)
goto error;
// Modify dataset creation properties, i.e. enable chunking and compression
if((cparms = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102))) < 0)
goto error;
if((status = [H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)( cparms, [RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), chunk_dims)) < 0)
goto error;
if((status = [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d)( cparms, aggression)) < 0)
goto error;
// Create a new dataset within the file using cparms creation properties
if((dset_id = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file, DATASETNAME, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f),cparms,
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0) goto error;
// Initialize data for one chunk
for(i = n = 0; i < CHUNK_NX; i++)
for(j = 0; j < CHUNK_NY; j++)
direct_buf[i][j] = n++;
// Allocate output (compressed) buffer
outbuf = malloc(z_dst_nbytes);
z_dst = (Bytef *)outbuf;
// Perform compression from the source to the destination buffer
ret = compress2(z_dst, &z_dst_nbytes, z_src, z_src_nbytes, aggression);
// Check for various zlib errors
if(Z_BUF_ERROR == ret) {
fprintf(stderr, "overflow");
goto error;
} else if(Z_MEM_ERROR == ret) {
fprintf(stderr, "deflate memory error");
goto error;
} else if(Z_OK != ret) {
fprintf(stderr, "other deflate error");
goto error;
}
// Write the compressed chunk data repeatedly to cover all the chunks in the dataset, using the direct
write function. for(i=0; i<NX/CHUNK_NX; i++) { for(j=0; j<NY/CHUNK_NY; j++) { status =
[H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)(dset_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), filter_mask, offset, z_dst_nbytes, outbuf); offset[1] += CHUNK_NY;
}
offset[0] += CHUNK_NX;
offset[1] = 0;
}
// Overwrite the first chunk with uncompressed data. Set the filter mask to indicate the compression
filter is skipped filter_mask = 0x00000001; offset[0] = offset[1] = 0; if([H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)(dset_id,
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), filter_mask, offset, buf_size, direct_buf) < 0) goto error;
// Read the entire dataset back for data verification converting ints to longs
if([H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, [H5T_NATIVE_LONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga290b9655882754ee0ec9f31b42ac04f6), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), outbuf_long) < 0)
goto error;
// Data verification here
...
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)
#define H5P_DATASET_CREATE
**Definition** H5Ppublic.h:60
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484)
#define H5S_ALL
**Definition** H5Spublic.h:33
[H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)
herr_t H5Pset_chunk(hid_t plist_id, int ndims, const hsize_t dim[])
Sets the size of the chunks used to store a chunked layout dataset.
[H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d)
herr_t H5Pset_deflate(hid_t plist_id, unsigned level)
Sets deflate (GNU gzip) compression method and compression level.
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)
herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void *buf)
Reads raw data from a dataset into a provided buffer.
[H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)
hid_t H5Dcreate2(hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)
Creates a new dataset and links it into the file.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
[H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
hid_t H5Screate_simple(int rank, const hsize_t dims[], const hsize_t maxdims[])
Creates a new simple dataspace and opens it for access.
[H5T_NATIVE_LONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga290b9655882754ee0ec9f31b42ac04f6)
#define H5T_NATIVE_LONG
**Definition** H5Tpublic.h:1000
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
[RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5)
#define RANK
**Definition** h5import.h:343
Next Chapter [HDF5 Standard for Dimension Scales](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#sec_dim_scales_stand)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


