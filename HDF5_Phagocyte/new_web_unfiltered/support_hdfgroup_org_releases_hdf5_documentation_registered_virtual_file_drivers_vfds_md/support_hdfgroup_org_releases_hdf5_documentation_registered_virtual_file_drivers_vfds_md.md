```
---
title: Registered Virtual File Drivers (VFDs)
---

# Registered Virtual File Drivers (VFDs)

Members of the HDF5 users community can register VFDs for use with HDF5. To register a VFD please contact [The HDF Helpdesk](https://help.hdfgroup.org) with the following information: 

* Contact information for the developer requesting a new identifier
* Short description of the new VFD
* Links to any relevant information including licensing information 

Here is the current policy regarding VFD identifier assignment: Valid VFD identifiers can have values from 0 through 255 for VFDs defined by the HDF5 library. Values 256 through 511 are available for testing new VFDs.  Subsequent values should be obtained by contacting the The HDF Help Desk.

Please contact the maintainer of a VFD for help implementing the plugin.

## List of VFDs Registered with The HDF Group

| Driver	| Driver Identifier| Search Name\*	| Short Description	| URL	| Contacts| 
| --- | --- | --- | --- | --- | --- | 
| CUDA GPU	| 512	| gds	| The HDF5 GPUDirect Storage VFD is a Virtual File Driver (VFD) for HDF5 that can be used to interface with Nvidia's GPUDirect Storage (GDS) API. The driver is built as a plugin library that is external to HDF5.	| [https://github.com/hpc-io/vfd-gds](ttps://github.com/hpc-io/vfd-gds)	|  Suren Byna (sbyna at lbl dot gov)| 
| GDAL vsil	| 513	| vsil	| The HDF5 GDAL vsil Storage VFD is a Virtual File Driver (VFD) for the GDAL HDF5 driver that can be used to access any file supported by the GDAL Virtual File System Interface ([https://gdal.org/user/virtual_file_systems.html](https://gdal.org/user/virtual_file_systems.html)). | [https://github.com/OSGeo/gdal/blob/master/frmts/hdf5/hdf5vfl.h](https://github.com/OSGeo/gdal/blob/master/frmts/hdf5/hdf5vfl.h) | Even Rouault (even dot rouault at spatialys dot com)| 
| Unidata/UCAR NetCDF-C ByteRange	| 514	| byte-range	| The Unidata H5FDhttp.[ch] VFD driver is used to support accessing remote files using the HTTP byte range mechanism. It is part of the Unidata Netcdf-C library. | [https://github.com/Unidata/netcdf-c/blob/main/libhdf5/H5FDhttp.c](https://github.com/Unidata/netcdf-c/blob/main/libhdf5/H5FDhttp.c)	| Dennis Heimbigner (dmh at ucar dot edu) | 
| MathWorks (MW) VFD | 515 | no search name | The MathWorks (MW) VFD is a Virtual File Driver (VFD) for HDF5 which can be used for reading and writing hdf5 files from/to remote locations including (but not limited to) AWS S3, Azure Blob Storage and Hadoop Distributed File System (HDFS). | [](https://url.com) | Abhi Baruah (abaruah at mathworks dot com) |

*The Search Name provides a mechanism for searching for a VFD.

```

