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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#title0 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#title1 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#title2 "H2")
  * [◆ H5DOappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#title3 "H2")
  * [◆ H5DOread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#title4 "H2")
  * [◆ H5DOwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Functions](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#func-members)
HDF5 Optimizations APIs (H5DO)
## Detailed Description
_Bypassing default HDF5 behavior in order to optimize for specific use cases (H5DO)_
HDF5 functions described is this section are implemented in the HDF5 High-level library as optimized functions. These functions generally require careful setup and testing as they enable an application to bypass portions of the HDF5 library's I/O pipeline for performance purposes.
These functions are distributed in the standard HDF5 distribution and are available any time the HDF5 High-level library is available.
  * [H5DOappend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga1fcbd6fc898ccb603e85e75f9507a76c)   
Appends data to a dataset along a specified dimension.
  * [H5DOread_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76)   
Reads a raw data chunk directly from a dataset in a file into a buffer (DEPRECATED)
  * [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)   
Writes a raw data chunk from a buffer directly to a dataset in a file (DEPRECATED) 


##  Functions  
---  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DOappend](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga1fcbd6fc898ccb603e85e75f9507a76c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, unsigned axis, size_t extension, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) memtype, const void *buf)  
| Appends data to a dataset along a specified dimension.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DOread_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, uint32_t *filters, void *buf)  
| Reads a raw data chunk directly from a dataset in a file into a buffer.   
  
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset_id, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dxpl_id, uint32_t filters, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *offset, size_t data_size, const void *buf)  
| Writes a raw data chunk from a buffer directly to a dataset in a file.   
  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga1fcbd6fc898ccb603e85e75f9507a76c)H5DOappend()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DOappend  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | unsigned |  _axis_ ,   
|  | size_t |  _extension_ ,   
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _memtype_ ,   
|  | const void * |  _buf_ )  
Appends data to a dataset along a specified dimension. 
* * * 

Parameters
     [in] | dset_id | Dataset identifier   
---|---|---  
[in] | dxpl_id | Dataset transfer property list identifier   
[in] | axis | Dataset Dimension (0-based) for the append   
[in] | extension | Number of elements to append for the axis-th dimension   
[in] | memtype | The memory datatype identifier   
[in] | buf | Buffer with data for the append 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value.  
The [H5DOappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga1fcbd6fc898ccb603e85e75f9507a76c "Appends data to a dataset along a specified dimension.") routine extends a dataset by `extension` number of elements along a dimension specified by a dimension `axis` and writes `buf` of elements to the dataset. Dimension `axis` is 0-based. Elements’ type is described by `memtype`.
This routine combines calling [H5Dset_extent()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad31e1e0129f4520c531ce524de2a056f "Changes the sizes of a dataset's dimensions."), [H5Sselect_hyperslab()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region."), and [H5Dwrite()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") into a single routine that simplifies application development for the common case of appending elements to an existing dataset.
For a multi-dimensional dataset, appending to one dimension will write a contiguous hyperslab over the other dimensions. For example, if a 3-D dataset has dimension sizes (3, 5, 8), extending the 0th dimension (currently of size 3) by 3 will append 3*5*8 = 120 elements (which must be pointed to by the `buffer` parameter) to the dataset, making its final dimension sizes (6, 5, 8).
If a dataset has more than one unlimited dimension, any of those dimensions may be appended to, although only along one dimension per call to [H5DOappend()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga1fcbd6fc898ccb603e85e75f9507a76c "Appends data to a dataset along a specified dimension."). 

Since
    1.10.0 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76)H5DOread_chunk()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DOread_chunk  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ ,   
|  | uint32_t * |  _filters_ ,   
|  | void * |  _buf_ )  
Reads a raw data chunk directly from a dataset in a file into a buffer. 
* * * 

Parameters
     [in] | dset_id | Identifier for the dataset to be read   
---|---|---  
[in] | dxpl_id | Transfer property list identifier for this I/O operation   
[in] | offset | Logical position of the chunk's first element in the dataspace   
[in,out] | filters | Mask for identifying the filters used with the chunk   
[in] | buf | Buffer containing the chunk read from the dataset 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000003)**
      
This function was deprecated in favor of the function [H5Dread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e) as of HDF5-1.10.3. In HDF5 1.10.3, the functionality of [H5DOread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76 "Reads a raw data chunk directly from a dataset in a file into a buffer.") was moved to [H5Dread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e). 
For compatibility, this API call has been left as a stub which simply calls [H5Dread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e). New code should use [H5Dread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a8ff3b0cf52e30e2e12d617aa2329486e).
The [H5DOread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76 "Reads a raw data chunk directly from a dataset in a file into a buffer.") reads a raw data chunk as specified by its logical `offset` in a chunked dataset `dset_id` from the dataset in the file into the application memory buffer `buf`. The data in `buf` is read directly from the file bypassing the library's internal data transfer pipeline, including filters.
`dxpl_id` is a data transfer property list identifier.
The mask `filters` indicates which filters are used with the chunk when written. A zero value indicates that all enabled filters are applied on the chunk. A filter is skipped if the bit corresponding to the filter's position in the pipeline (`0 ≤ position < 32`) is turned on.
`offset` is an array specifying the logical position of the first element of the chunk in the dataset's dataspace. The length of the offset array must equal the number of dimensions, or rank, of the dataspace. The values in `offset` must not exceed the dimension limits and must specify a point that falls on a dataset chunk boundary.
`buf` is the memory buffer containing the chunk read from the dataset in the file. 

Example
    The following code illustrates the use of [H5DOread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76 "Reads a raw data chunk directly from a dataset in a file into a buffer.") to read a chunk from a dataset: 
#include <zlib.h>
#include <math.h>
#define DEFLATE_SIZE_ADJUST(s) (ceil(((double)(s)) * 1.001) + 12)
:
:
size_t buf_size = CHUNK_NX*CHUNK_NY*sizeof(int);
const Bytef *z_src = (const Bytef *)(direct_buf);
Bytef *z_dst; /* Destination buffer */
uLongf z_dst_nbytes = (uLongf)DEFLATE_SIZE_ADJUST(buf_size);
uLong z_src_nbytes = (uLong)buf_size;
int aggression = 9; /* Compression aggression setting */
uint32_t filter_mask = 0;
size_t buf_size = CHUNK_NX * CHUNK_NY * sizeof(int);
/* For H5DOread_chunk() */
void *readbuf = NULL; /* Buffer for reading data */
const Bytef *pt_readbuf; /* Point to the buffer for data read */
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) read_chunk_nbytes; /* Size of chunk on disk */
int read_dst_buf[CHUNK_NX][CHUNK_NY]; /* Buffer to hold un-compressed data */
/* Create the data space */
if ((dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dims, maxdims)) < 0)
goto error;
/* Create a new file */
if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(FILE_NAME5, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0)
goto error;
/* Modify dataset creation properties, i.e. enable chunking and compression */
if ((cparms = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102))) < 0)
goto error;
if ((status = [H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)(cparms, [RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), chunk_dims)) < 0)
goto error;
if ((status = [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d)(cparms, aggression)) < 0)
goto error;
/* Create a new dataset within the file using cparms creation properties */
if ((dset_id = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file, DATASETNAME, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), cparms,
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0)
goto error;
/* Initialize data for one chunk */
for (i = n = 0; i < CHUNK_NX; i++)
for (j = 0; j < CHUNK_NY; j++)
direct_buf[i][j] = n++;
/* Allocate output (compressed) buffer */
outbuf = malloc(z_dst_nbytes);
z_dst = (Bytef *)outbuf;
/* Perform compression from the source to the destination buffer */
ret = compress2(z_dst, &z_dst_nbytes, z_src, z_src_nbytes, aggression);
/* Check for various zlib errors */
if (Z_BUF_ERROR == ret) {
fprintf(stderr, "overflow");
goto error;
}
else if (Z_MEM_ERROR == ret) {
fprintf(stderr, "deflate memory error");
goto error;
}
else if (Z_OK != ret) {
fprintf(stderr, "other deflate error");
goto error;
}
/* Write the compressed chunk data repeatedly to cover all the
* * chunks in the dataset, using the direct write function. */
for (i = 0; i < NX / CHUNK_NX; i++) {
for (j = 0; j < NY / CHUNK_NY; j++) {
status = [H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)(dset_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), filter_mask, offset, z_dst_nbytes, outbuf);
offset[1] += CHUNK_NY;
}
offset[0] += CHUNK_NX;
offset[1] = 0;
}
if ([H5Fflush](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae686870f0a276c4d06bbc667b2c24124)(dataset, [H5F_SCOPE_LOCAL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#ac9db1b1211555797021daed9b54b8cdfa3d09ee2d1551f31b008a7dc7d80b42bd)) < 0)
goto error;
if ([H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dataset) < 0)
goto error;
if ((dataset = [H5Dopen2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga04198c4cf0b849ed3a8921f6c7169ee2)(file, DATASETNAME1, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0)
goto error;
offset[0] = CHUNK_NX;
offset[1] = CHUNK_NY;
/* Get the size of the compressed chunk */
ret = [H5Dget_chunk_storage_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gaaeea958861de082db9051fc4bf215234)(dataset, offset, &read_chunk_nbytes);
readbuf = malloc(read_chunk_nbytes);
pt_readbuf = (const Bytef *)readbuf;
/* Use H5DOread_chunk() to read the chunk back */
if ((status = [H5DOread_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76)(dataset, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), offset, &read_filter_mask, readbuf)) < 0)
goto error;
ret =
uncompress((Bytef *)read_dst_buf, (uLongf *)&buf_size, pt_readbuf, (uLong)read_chunk_nbytes);
/* Check for various zlib errors */
if (Z_BUF_ERROR == ret) {
fprintf(stderr, "error: not enough room in output buffer");
goto error;
}
else if (Z_MEM_ERROR == ret) {
fprintf(stderr, "error: not enough memory");
goto error;
}
else if (Z_OK != ret) {
fprintf(stderr, "error: corrupted input data");
goto error;
}
/* Data verification here */
:
: 

Version
    1.10.3 Function deprecated in favor of H5Dread_chunk. 

Since
    1.10.2, 1.8.19 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)H5DOwrite_chunk()
H5HL_DLL [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) H5DOwrite_chunk  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dset_id_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _dxpl_id_ ,   
|  | uint32_t |  _filters_ ,   
|  | const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) * |  _offset_ ,   
|  | size_t |  _data_size_ ,   
|  | const void * |  _buf_ )  
Writes a raw data chunk from a buffer directly to a dataset in a file. 
* * * 

Parameters
     [in] | dset_id | Identifier for the dataset to write to   
---|---|---  
[in] | dxpl_id | Transfer property list identifier for this I/O operation   
[in] | filters | Mask for identifying the filters in use   
[in] | offset | Logical position of the chunk's first element in the dataspace   
[in] | data_size | Size of the actual data to be written in bytes   
[in] | buf | Buffer containing data to be written to the chunk 

Returns
    Returns a non-negative value if successful; otherwise, returns a negative value. 

**[Deprecated](https://support.hdfgroup.org/documentation/hdf5/latest/deprecated.html#_deprecated000002)**
      
This function was deprecated in favor of the function [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.") of HDF5-1.10.3. The functionality of [H5DOwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") was moved to [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file."). 
For compatibility, this API call has been left as a stub which simply calls [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file."). New code should use [H5Dwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga416ccd200929b11386a10e9024977109 "Writes a raw data chunk from a buffer directly to a dataset in a file.").
The [H5DOwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") writes a raw data chunk as specified by its logical `offset` in a chunked dataset `dset_id` from the application memory buffer `buf` to the dataset in the file. Typically, the data in `buf` is preprocessed in memory by a custom transformation, such as compression. The chunk will bypass the library's internal data transfer pipeline, including filters, and will be written directly to the file.
`dxpl_id` is a data transfer property list identifier.
`filters` is a mask providing a record of which filters are used with the chunk. The default value of the mask is zero (`0`), indicating that all enabled filters are applied. A filter is skipped if the bit corresponding to the filter's position in the pipeline (`0 ≤ position < 32`) is turned on. This mask is saved with the chunk in the file.
`offset` is an array specifying the logical position of the first element of the chunk in the dataset's dataspace. The length of the offset array must equal the number of dimensions, or rank, of the dataspace. The values in `offset` must not exceed the dimension limits and must specify a point that falls on a dataset chunk boundary.
`data_size` is the size in bytes of the chunk, representing the number of bytes to be read from the buffer `buf`. If the data chunk has been precompressed, `data_size` should be the size of the compressed data.
`buf` is the memory buffer containing data to be written to the chunk in the file. 

Attention
    Exercise caution when using [H5DOread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76 "Reads a raw data chunk directly from a dataset in a file into a buffer.") and [H5DOwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file."), as they read and write data chunks directly in a file. [H5DOwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") bypasses hyperslab selection, the conversion of data from one datatype to another, and the filter pipeline to write the chunk. Developers should have experience with these processes before using this function. Please see [HDF5 High Level Optimizations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html) for more information. 

Note
     [H5DOread_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga42ee2888be9fe225803bf50641faae76 "Reads a raw data chunk directly from a dataset in a file into a buffer.") and [H5DOwrite_chunk()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29 "Writes a raw data chunk from a buffer directly to a dataset in a file.") are not supported under parallel and do not support variable length types. 

Example
    The following code illustrates the use of H5DOwrite_chunk to write an entire dataset, chunk by chunk: 
#include <zlib.h>
#include <math.h>
#define DEFLATE_SIZE_ADJUST(s) (ceil(((double)(s)) * 1.001) + 12)
:
:
size_t buf_size = CHUNK_NX*CHUNK_NY*sizeof(int);
const Bytef *z_src = (const Bytef *)(direct_buf);
Bytef *z_dst; /* Destination buffer */
uLongf z_dst_nbytes = (uLongf)DEFLATE_SIZE_ADJUST(buf_size);
uLong z_src_nbytes = (uLong)buf_size;
int aggression = 9; /* Compression aggression setting */
uint32_t filter_mask = 0;
size_t buf_size = CHUNK_NX * CHUNK_NY * sizeof(int);
/* Create the data space */
if ((dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dims, maxdims)) < 0)
goto error;
/* Create a new file */
if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(FILE_NAME5, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0)
goto error;
/* Modify dataset creation properties, i.e. enable chunking and compression */
if ((cparms = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102))) < 0)
goto error;
if ((status = [H5Pset_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga3584d592e377da3604b7604e266dcf5b)(cparms, [RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), chunk_dims)) < 0)
goto error;
if ((status = [H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d)(cparms, aggression)) < 0)
goto error;
/* Create a new dataset within the file using cparms creation properties */
if ((dset_id = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(file, DATASETNAME, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), cparms,
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))) < 0)
goto error;
/* Initialize data for one chunk */
for (i = n = 0; i < CHUNK_NX; i++)
for (j = 0; j < CHUNK_NY; j++)
direct_buf[i][j] = n++;
/* Allocate output (compressed) buffer */
outbuf = malloc(z_dst_nbytes);
z_dst = (Bytef *)outbuf;
/* Perform compression from the source to the destination buffer */
ret = compress2(z_dst, &z_dst_nbytes, z_src, z_src_nbytes, aggression);
/* Check for various zlib errors */
if (Z_BUF_ERROR == ret) {
fprintf(stderr, "overflow");
goto error;
}
else if (Z_MEM_ERROR == ret) {
fprintf(stderr, "deflate memory error");
goto error;
}
else if (Z_OK != ret) {
fprintf(stderr, "other deflate error");
goto error;
}
/* Write the compressed chunk data repeatedly to cover all the
* * chunks in the dataset, using the direct write function. */
for (i = 0; i < NX / CHUNK_NX; i++) {
for (j = 0; j < NY / CHUNK_NY; j++) {
status =
[H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)(dset_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), filter_mask, offset, z_dst_nbytes, outbuf);
offset[1] += CHUNK_NY;
}
offset[0] += CHUNK_NX;
offset[1] = 0;
}
/* Overwrite the first chunk with uncompressed data. Set the filter mask to
* * indicate the compression filter is skipped */
filter_mask = 0x00000001;
offset[0] = offset[1] = 0;
if ([H5DOwrite_chunk](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_o.html#ga873b08130d88e81487f24bc3c1ce0d29)(dset_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), filter_mask, offset, buf_size, direct_buf) < 0)
goto error;
/* Read the entire dataset back for data verification converting ints to longs */
if ([H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, [H5T_NATIVE_LONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga290b9655882754ee0ec9f31b42ac04f6), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), outbuf_long) < 0)
goto error;
/* Data verification here */
:
: 

Version
    1.10.3 Function deprecated in favor of H5Dwrite_chunk. 

Since
    1.8.11 
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


