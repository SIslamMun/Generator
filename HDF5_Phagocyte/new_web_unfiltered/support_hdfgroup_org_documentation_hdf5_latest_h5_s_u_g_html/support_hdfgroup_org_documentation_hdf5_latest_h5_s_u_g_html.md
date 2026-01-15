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
  * [ HDF5 Dataspaces and Partial I/O](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title1 "H2")
  * [ Dataspace Function Summaries](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title2 "H2")
  * [ Definition of Dataspace Objects and the Dataspace Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title3 "H2")
  * [ Dataspaces and Data Transfer](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title4 "H2")
  * [ Dataspace Selection Operations and Data Transfer](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title5 "H2")
  * [ References](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title6 "H2")
  * [ Deprecated References to Dataset Regions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title7 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title8 "H2")
  * [ Sample Programs](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Dataspaces and Partial I/O
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 Dataspaces and Partial I/O
HDF5 dataspaces describe the _shape_ of datasets in memory or in HDF5 files. Dataspaces can be empty ([H5S_NULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aa3d83ec70c1a824a2d821bf8464488bc5)), a singleton ([H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8)), or a multi-dimensional, regular grid ([H5S_SIMPLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aac4ee937dcccfd99f885626983e7d2750)). Dataspaces can be re-shaped.
Subsets of dataspaces can be "book-marked" or used to restrict I/O operations using _selections_. Furthermore, certain set operations are supported for selections.
##  Introduction
The HDF5 _dataspace_ is a required component of an HDF5 dataset or attribute definition. The dataspace defines the size and shape of the dataset or attribute raw data. In other words, a dataspace defines the number of dimensions and the size of each dimension of the multidimensional array in which the raw data is represented. The dataspace must be defined when the dataset or attribute is created.
The _dataspace_ is also used during dataset I/O operations, defining the elements of the dataset that participate in the I/O operation.
This chapter explains the _dataspace_ object and its use in dataset and attribute creation and data transfer. It also describes selection operations on a dataspace used to implement sub‐setting, sub‐sampling, and scatter‐gather access to datasets.
##  Dataspace Function Summaries
see [Dataspaces (H5S)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html) reference manual provides a reference list of dataspace functions, the H5S APIs.
##  Definition of Dataspace Objects and the Dataspace Programming Model
This section introduces the notion of the HDF5 dataspace object and a programming model for creating and working with dataspaces.
###  Dataspace Objects
An HDF5 dataspace is a required component of an HDF5 dataset or attribute. A dataspace defines the size and the shape of a dataset's or an attribute's raw data. Currently, HDF5 supports the following types of the dataspaces: 
  * Scalar dataspaces 
  * Simple dataspaces 
  * Null dataspaces


A scalar dataspace, [H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8), represents just one element, a scalar. Note that the datatype of this one element may be very complex; example would be a compound structure with members being of any allowed HDF5 datatype, including multidimensional arrays, strings, and nested compound structures. By convention, the rank of a scalar dataspace is always 0 (zero); think of it geometrically as a single, dimensionless point, though that point may be complex.
A simple dataspace, [H5S_SIMPLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aac4ee937dcccfd99f885626983e7d2750) , is a multidimensional array of elements. The dimensionality of the dataspace (or the rank of the array) is fixed and is defined at creation time. The size of each dimension can grow during the life time of the dataspace from the current size up to the maximum size. Both the current size and the maximum size are specified at creation time. The sizes of dimensions at any particular time in the life of a dataspace are called the current dimensions, or the dataspace extent. They can be queried along with the maximum sizes.
A null dataspace, [H5S_NULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aa3d83ec70c1a824a2d821bf8464488bc5), contains no data elements. Note that no selections can be applied to a null dataset as there is nothing to select.
As shown in the UML diagram in the figure below, an HDF5 simple dataspace object has three attributes: the rank or number of dimensions; the current sizes, expressed as an array of length rank with each element of the array denoting the current size of the corresponding dimension; and the maximum sizes, expressed as an array of length rank with each element of the array denoting the maximum size of the corresponding dimension.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_simple.gif) A simple dataspace  
---  
_Note:_ A simple dataspace is defined by its rank, the current size of each dimension, and the maximum size of each dimension.
The size of a current dimension cannot be greater than the maximum size, which can be unlimited, specified as [H5S_UNLIMITED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5af9ab788797b2ea9a4843857674ac18). Note that while the HDF5 file format and library impose no maximum size on an unlimited dimension, practically speaking its size will always be limited to the biggest integer available on the particular system being used.
Dataspace rank is restricted to 32, the standard limit in C on the rank of an array, in the current implementation of the HDF5 Library. The HDF5 file format, on the other hand, allows any rank up to the maximum integer value on the system, so the library restriction can be raised in the future if higher dimensionality is required.
Note that most of the time Fortran applications calling HDF5 will work with dataspaces of rank less than or equal to seven, since seven is the maximum number of dimensions in a Fortran array. But dataspace rank is not limited to seven for Fortran applications.
The current dimensions of a dataspace, also referred to as the dataspace extent, define the bounding box for dataset elements that can participate in I/O operations.
###  Dataspace Programming Model
The programming model for creating and working with HDF5 dataspaces can be summarized as follows: 
  * 1. Create a dataspace 
  * 2. Use the dataspace to create a dataset in the file or to describe a data array in memory 
  * 3. Modify the dataspace to define dataset elements that will participate in I/O operations 
  * 4. Use the modified dataspace while reading/writing dataset raw data or to create a region reference 
  * 5. Close the dataspace when no longer needed


The rest of this section will address steps 1, 2, and 5 of the programming model; steps 3 and 4 will be discussed in later sections of this chapter.
#### Creating a Dataspace
A dataspace can be created by calling the [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083) function. Since the definition of a simple dataspace requires the specification of dimensionality (or rank) and initial and maximum dimension sizes, the HDF5 Library provides a convenience API, [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae) to create a simple dataspace in one step.
The following examples illustrate the usage of these APIs.
#### Creating a Scalar Dataspace
Creating a Scalar Dataspace 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id;
. . .
space_id = [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)([H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5S_SCALAR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aaf6a34a2439db8aa7bb63ed0c4aaa5eb8)
@ H5S_SCALAR
**Definition** H5Spublic.h:90
[H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)
hid_t H5Screate(H5S_class_t type)
Creates a new dataspace of a specified type.
As mentioned above, the dataspace will contain only one element. Scalar dataspaces are used more often for describing attributes that have just one value. For example, the attribute temperature with the value Celsius is used to indicate that the dataset with this attribute stores temperature values using the Celsius scale.
#### Creating a Null Dataspace
A null dataspace is created with the [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083) function. 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id;
. . .
space_id = [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)([H5S_NULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aa3d83ec70c1a824a2d821bf8464488bc5));
[H5S_NULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aa3d83ec70c1a824a2d821bf8464488bc5)
@ H5S_NULL
**Definition** H5Spublic.h:92
As mentioned above, the dataspace will contain no elements.
#### Creating a Simple Dataspace
Let's assume that an application wants to store a two‐dimensional array of data, A(20,100). During the life of the application, the first dimension of the array can grow up to 30; there is no restriction on the size of the second dimension. The following steps are used to declare a dataspace for the dataset in which the array data will be stored. 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id;
int rank = 2;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) current_dims[2] = {20, 100};
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) max_dims[2] = {30, [H5S_UNLIMITED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5af9ab788797b2ea9a4843857674ac18)};
. . .
space_id = [H5Screate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gabee514327cba34ca9951b24fa14fb083)([H5S_NULL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#ae53f3c6a52563646fbac9ead8ecdbf0aa3d83ec70c1a824a2d821bf8464488bc5));
[H5Sset_extent_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gaf2526a41d2f4506e2c52098510517343)(space_id, rank, current_dims, max_dims);
[H5S_UNLIMITED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5af9ab788797b2ea9a4843857674ac18)
#define H5S_UNLIMITED
**Definition** H5Spublic.h:52
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5Sset_extent_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gaf2526a41d2f4506e2c52098510517343)
herr_t H5Sset_extent_simple(hid_t space_id, int rank, const hsize_t dims[], const hsize_t max[])
Sets or resets the size of an existing dataspace.
Alternatively, the convenience APIs H5Screate_simple/h5screate_simple_f can replace the H5Screate/h5screate_f and H5Sset_extent_simple/h5sset_extent_simple_f calls. 
space_id = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(rank, current_dims, max_dims);
[H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
hid_t H5Screate_simple(int rank, const hsize_t dims[], const hsize_t maxdims[])
Creates a new simple dataspace and opens it for access.
In this example, a dataspace with current dimensions of 20 by 100 is created. The first dimension can be extended only up to 30. The second dimension, however, is declared unlimited; it can be extended up to the largest available integer value on the system.
Note that when there is a difference between the current dimensions and the maximum dimensions of an array, then chunking storage must be used. In other words, if the number of dimensions may change over the life of the dataset, then chunking must be used. If the array dimensions are fixed (if the number of current dimensions is equal to the maximum number of dimensions when the dataset is created), then contiguous storage can be used. For more information, see "Data Transfer".
Maximum dimensions can be the same as current dimensions. In such a case, the sizes of dimensions cannot be changed during the life of the dataspace object. In C, `NULL` can be used to indicate to the [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae) and [H5Sset_extent_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gaf2526a41d2f4506e2c52098510517343) functions that the maximum sizes of all dimensions are the same as the current sizes. 
space_id = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(rank, current_dims, NULL);
The created dataspace will have current and maximum dimensions of 20 and 100 correspondingly, and the sizes of those dimensions cannot be changed.
#### C versus Fortran Dataspaces
Dataspace dimensions are numbered from 1 to rank. HDF5 uses C storage conventions, assuming that the last listed dimension is the fastest‐changing dimension and the first‐listed dimension is the slowest changing. The HDF5 file format storage layout specification adheres to the C convention and the HDF5 Library adheres to the same convention when storing dataspace dimensions in the file. This affects how C programs and tools interpret data written from Fortran programs and vice versa. The example below illustrates the issue.
When a Fortran application describes a dataspace to store an array as A(20,100), it specifies the value of the first dimension to be 20 and the second to be 100. Since Fortran stores data by columns, the first‐listed dimension with the value 20 is the fastest‐changing dimension and the last‐listed dimension with the value 100 is the slowest‐changing. In order to adhere to the HDF5 storage convention, the HDF5 Fortran wrapper transposes dimensions, so the first dimension becomes the last. The dataspace dimensions stored in the file will be 100,20 instead of 20,100 in order to correctly describe the Fortran data that is stored in 100 columns, each containing 20 elements.
When a Fortran application reads the data back, the HDF5 Fortran wrapper transposes the dimensions once more, returning the first dimension to be 20 and the second to be 100, describing correctly the sizes of the array that should be used to read data in the Fortran array A(20,100).
When a C application reads data back, the dimensions will come out as 100 and 20, correctly describing the size of the array to read data into, since the data was written as 100 records of 20 elements each. Therefore C tools such as [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) and [h5ls](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html#sec_cltools_h5ls) always display transposed dimensions and values for the data written by a Fortran application.
Consider the following simple example of equivalent C 3 x 5 and Fortran 5 x 3 arrays. As illustrated in the figure below, a C application will store a 3 x 5 2‐dimensional array as three 5‐element rows. In order to store the same data in the same order, a Fortran application must view the array as a 5 x 3 array with three 5‐element columns. The dataspace of this dataset, as written from Fortran, will therefore be described as 5 x 3 in the application but stored and described in the file according to the C convention as a 3 x 5 array. This ensures that C and Fortran applications will always read the data in the order in which it was written. The HDF5 Fortran interface handles this transposition automatically. 
// C
\\#define NX 3 // dataset dimensions
\\#define NY 5
. . .
int data[NX][NY]; // data to write
. . .
// Data and output buffer initialization.
for (j = 0; j < NX; j++)
for (i = 0; i < NY; i++)
data[j][i] = i + j;
//
// 1 2 3 4 5
// 6 7 8 9 10
// 11 12 13 14 15
//
. . .
dims[0] = NX;
dims[1] = NY;
dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dims, NULL);
[RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5)
#define RANK
**Definition** h5import.h:343
! Fortran
INTEGER, PARAMETER :: NX = 3
INTEGER, PARAMETER :: NX = 5
. . .
INTEGER(HSIZE_T), DIMENSION(2) :: dims = (/NY, NX/) ! Dataset dimensions
. . .
!
! Initialize data
!
do i = 1, NY
do j = 1, NX
data(i,j) = i + (j-1)*NY
enddo
enddo
!
! Data
!
! 1 6 11
! 2 7 12
! 3 8 13
! 4 9 14
! 5 10 15
. . .
CALL h5screate_simple_f(rank, dims, dspace_id, error)
Comparing C and Fortran dataspaces A dataset stored by a C program in a 3 x 5 array:   
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_CvsF1.gif)  
The same dataset stored by a Fortran program in a 5 x 3 array:   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_CvsF2.gif)  
The first dataset above as written to an HDF5 file from C or the second dataset above as written from Fortran:   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_CvsF3.gif)  
The first dataset above as written to an HDF5 file from Fortran:   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_CvsF4.gif)  
_Note: The HDF5 Library stores arrays along the fastest‐changing dimension. This approach is often referred to as being “in C order.” C, C++, and Java work with arrays in row‐major order. In other words, the row, or the last dimension, is the fastest‐changing dimension. Fortran, on the other hand, handles arrays in column‐major order making the column, or the first dimension, the fastest‐changing dimension. Therefore, the C and Fortran arrays illustrated in the top portion of this figure are stored identically in an HDF5 file. This ensures that data written by any language can be meaningfully read, interpreted, and manipulated by any other._
#### Finding Dataspace Characteristics
The HDF5 Library provides several APIs designed to query the characteristics of a dataspace.
The function [H5Sis_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gab0b1560f7c8402986f332522e2adae1d) returns information about the type of a dataspace. This function is rarely used and currently supports only simple and scalar dataspaces.
To find out the dimensionality, or rank, of a dataspace, use [H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gae5282a81692b80b5b19dd12d05b9b28e). [H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3) can also be used to find out the rank. See the example below. If both functions return 0 for the value of rank, then the dataspace is scalar.
To query the sizes of the current and maximum dimensions, use [H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3).
The following example illustrates querying the rank and dimensions of a dataspace using these functions. 
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) space_id;
int rank;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *current_dims;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *max_dims;
. . .
rank = [H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gae5282a81692b80b5b19dd12d05b9b28e)(space_id);
// (or rank = H5Sget_simple_extent_dims(space_id, NULL, NULL);)
current_dims = ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c))malloc(rank * sizeof([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)));
max_dims = ([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c))malloc(rank * sizeof([hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)));
[H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3)(space_id, current_dims, max_dims);
// Print values here
[H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3)
int H5Sget_simple_extent_dims(hid_t space_id, hsize_t dims[], hsize_t maxdims[])
Retrieves dataspace dimension size and maximum size.
[H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gae5282a81692b80b5b19dd12d05b9b28e)
int H5Sget_simple_extent_ndims(hid_t space_id)
Determines the dimensionality of a dataspace.
##  Dataspaces and Data Transfer
Read and write operations transfer data between an HDF5 file on disk and in memory. The shape that the array data takes in the file and in memory may be the same, but HDF5 also allows users the ability to represent data in memory in a different shape than in the file. If the shape of an array in the file and in memory will be the same, then the same dataspace definition can be used for both. If the shape of an array in memory needs to be different than the shape in the file, then the dataspace definition for the shape of the array in memory can be changed. During a read operation, the array will be read into the different shape in memory, and during a write operation, the array will be written to the file in the shape specified by the dataspace in the file. The only qualification is that the number of elements read or written must be the same in both the source and the destination dataspaces.
Item a in the figure below shows a simple example of a read operation in which the data is stored as a 3 by 4 array in the file (item b) on disk, but the program wants it to be a 4 by 3 array in memory. This is accomplished by setting the memory dataspace to describe the desired memory layout, as in item c. The read operation reads the data in the file array into the memory array.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_read.gif) Data layout before and after a read operation  
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_move.gif) Moving data from disk to memory  
---  
Both the source and destination are stored as contiguous blocks of storage with the elements in the order specified by the dataspace. The figure above shows one way the elements might be organized. In item a, the elements are stored as 3 blocks of 4 elements. The destination is an array of 12 elements in memory (see item c). As the figure suggests, the transfer reads the disk blocks into a memory buffer (see item b), and then writes the elements to the correct locations in memory. A similar process occurs in reverse when data is written to disk.
###  Data Selection
In addition to rearranging data, the transfer may select the data elements from the source and destination.
Data selection is implemented by creating a dataspace object that describes the selected elements (within the hyper rectangle) rather than the whole array. Two dataspace objects with selections can be used in data transfers to read selected elements from the source and write selected elements to the destination. When data is transferred using the dataspace object, only the selected elements will be transferred.
This can be used to implement partial I/O, including: 
  * Sub‐setting ‐ reading part of a large dataset 
  * Sampling ‐ reading selected elements (for example, every second element) of a dataset 
  * Scatter‐gather ‐ read non‐contiguous elements into contiguous locations (gather) or read contiguous elements into non‐contiguous locations (scatter) or both


To use selections, the following steps are followed: 
  * 1. Get or define the dataspace for the source and destination 
  * 2. Specify one or more selections for source and destination dataspaces 
  * 3. Transfer data using the dataspaces with selections


A selection is created by applying one or more selections to a dataspace. A selection may override any other selections ([H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550)) or may be “Ored” with previous selections on the same dataspace ([H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8)). In the latter case, the resulting selection is the union of the selection and all previously selected selections. Arbitrary sets of points from a dataspace can be selected by specifying an appropriate set of selections.
Two selections are used in data transfer, so the source and destination must be compatible, as described below.
There are two forms of selection, hyperslab and point. A selection must be either a point selection or a set of hyperslab selections. Selections cannot be mixed.
The definition of a selection within a dataspace, not the data in the selection, cannot be saved to the file unless the selection definition is saved as a region reference. For more information, see [References](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#subsec_dataspace_refer).
#### Hyperslab Selection
A hyperslab is a selection of elements from a hyper rectangle. An HDF5 hyperslab is a rectangular pattern defined by four arrays. The four arrays are summarized in the table below.
The offset defines the origin of the hyperslab in the original dataspace.
The stride is the number of elements to increment between selected elements. A stride of ‘1’ is every element, a stride of ‘2’ is every second element, etc. Note that there may be a different stride for each dimen‐sion of the dataspace. The default stride is 1.
The count is the number of elements in the hyperslab selection. When the stride is 1, the selection is a hyper rectangle with a corner at the offset and size count[0] by count[1] by.... When stride is greater than one, the hyperslab bounded by the offset and the corners defined by stride[n] * count[n].
Hyperslab elements Parameter  | Description   
---|---  
Offset  | The starting location for the hyperslab.   
Stride  | The number of elements to separate each element or block to be selected.   
Count  | The number of elements or blocks to select along each dimension.   
Block  | The size of the block selected from the dataspace.   
The block is a count on the number of repetitions of the hyperslab. The default block size is '1', which is one hyperslab. A block of 2 would be two hyperslabs in that dimension, with the second starting at offset[n] + (count[n] * stride[n]) + 1.
A hyperslab can be used to access a sub‐set of a large dataset. The figure below shows an example of a hyperslab that reads a rectangle from the middle of a larger two dimensional array. The destination is the same shape as the source.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_subset.gif) Access a sub‐set of data with a hyperslab  
---  
Hyperslabs can be combined to select complex regions of the source and destination. The figure below shows an example of a transfer from one non‐rectangular region into another non‐rectangular region. The source is defined as the union of two hyperslabs, and the destination is the union of three hyperslabs.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_complex.gif) Build complex regions with hyperslab unions  
---  
Hyperslabs may also be used to collect or scatter data from regular patterns. The figure below shows an example where the source is a repeating pattern of blocks, and the destination is a single, one dimensional array.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_combine.gif) Use hyperslabs to combine or disperse data  
---  
#### Select Points
The second type of selection is an array of points such as coordinates. Essentially, this selection is a list of all the points to include. The figure below shows an example of a transfer of seven elements from a two dimensional dataspace to a three dimensional dataspace using a point selection to specify the points.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_point.gif) Point selection  
---  
#### Rules for Defining Selections
A selection must have the same number of dimensions (rank) as the dataspace it is applied to, although it may select from only a small region such as a plane from a 3D dataspace. Selections do not affect the extent of the dataspace, the selection may be larger than the dataspace. The boundaries of selections are reconciled with the extent at the time of the data transfer.
#### Data Transfer with Selections
A data transfer (read or write) with selections is the same as any read or write, except the source and destination dataspace have compatible selections.
During the data transfer, the following steps are executed by the library: 
  * The source and destination dataspaces are checked to assure that the selections are compatible. 
    * Each selection must be within the current extent of the dataspace. A selection may be defined to extend outside the current extent of the dataspace, but the dataspace cannot be accessed if the selection is not valid at the time of the access. 
    * The total number of points selected in the source and destination must be the same. Note that the dimensionality of the source and destination can be different (for example, the source could be 2D, the destination 1D or 3D), and the shape can be different, but the number of elements selected must be the same.
  * The data is transferred, element by element.


Selections have an iteration order for the points selected, which can be any permutation of the dimensions involved (defaulting to 'C' array order) or a specific order for the selected points, for selections composed of single array elements with [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d).
The elements of the selections are transferred in row‐major, or C order. That is, it is assumed that the first dimension varies slowest, the second next slowest, and so forth. For hyperslab selections, the order can be any permutation of the dimensions involved (defaulting to ‘C’ array order). When multiple hyperslabs are combined, the hyperslabs are coalesced into contiguous reads and writes.
In the case of point selections, the points are read and written in the order specified.
###  Programming Model
#### Selecting Hyperslabs
Suppose we want to read a 3x4 hyperslab from a dataset in a file beginning at the element <1,2> in the dataset, and read it into a 7 x 7 x 3 array in memory. See the figure below. In order to do this, we must create a dataspace that describes the overall rank and dimensions of the dataset in the file as well as the position and size of the hyperslab that we are extracting from that dataset.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_select.gif) Selecting a hyperslab  
---  
The code in the first example below illustrates the selection of the hyperslab in the file dataspace. The second example below shows the definition of the destination dataspace in memory. Since the in‐memory dataspace has three dimensions, the hyperslab is an array with three dimensions with the last dimension being 1: <3,4,1>. The third example below shows the read using the source and destination dataspaces with selections.
_Selecting a hyperslab_
//get the file dataspace.
dataspace = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)(dataset); // dataspace identifier
// Define hyperslab in the dataset.
offset[0] = 1;
offset[1] = 2;
count[0] = 3;
count[1] = 4;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(dataspace, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, NULL, count, NULL);
[H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550)
@ H5S_SELECT_SET
**Definition** H5Spublic.h:100
[H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)
hid_t H5Dget_space(hid_t dset_id)
Returns an identifier for a copy of the dataspace for a dataset.
[H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)
herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op, const hsize_t start[], const hsize_t stride[], const hsize_t count[], const hsize_t block[])
Selects a hyperslab region to add to the current selected region.
_Defining the destination memory_
// Define memory dataspace.
dimsm[0] = 7;
dimsm[1] = 7;
dimsm[2] = 3;
memspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(3,dimsm,NULL);
// Define memory hyperslab.
offset_out[0] = 3;
offset_out[1] = 0;
offset_out[2] = 0;
count_out[0] = 3;
count_out[1] = 4;
count_out[2] = 1;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(memspace, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset_out, NULL, count_out, NULL);
_A sample read specifying source and destination dataspaces_
ret = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), memspace,dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data);
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)
herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void *buf)
Reads raw data from a dataset into a provided buffer.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
#### Example with Strides and Blocks
Consider an 8 x 12 dataspace into which we want to write eight 3 x 2 blocks in a two dimensional array from a source dataspace in memory that is a 50‐element one dimensional array. See the figure below.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_write1to2.gif) Write from a one dimensional array to a two dimensional array  
---  
The example below shows code to write 48 elements from the one dimensional array to the file dataset starting with the second element in vector. The destination hyperslab has the following parameters: offset=(0,1), stride=(4,3), count=(2,4), block=(3,2). The source has the parameters: offset=(1), stride=(1), count=(48), block=(1). After these operations, the file dataspace will have the values shown in item b in the figure above. Notice that the values are inserted in the file dataset in row‐major order.
_Write from a one dimensional array to a two dimensional array_
// Select hyperslab for the dataset in the file, using 3 x 2 blocks, (4,3) stride (2,4)
// count starting at the position (0,1).
offset[0] = 0; offset[1] = 1;
stride[0] = 4; stride[1] = 3;
count[0] = 2; count[1] = 4;
block[0] = 3; block[1] = 2;
ret = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(fid, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, stride, count, block);
// Create dataspace for the first dataset.
mid1 = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(MSPACE1_RANK, dim1, NULL);
// Select hyperslab.
// We will use 48 elements of the vector buffer starting
// at the second element. Selected elements are
// 1 2 3 . . . 48
offset[0] = 1;
stride[0] = 1;
count[0] = 48;
block[0] = 1;
ret = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(mid1, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, stride, count, block);
// Write selection from the vector buffer to the dataset in the file.
ret = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), midd1, fid, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), vector)
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
#### Selecting a Union of Hyperslabs
The HDF5 Library allows the user to select a union of hyperslabs and write or read the selection into another selection. The shapes of the two selections may differ, but the number of elements must be equal.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_transfer.gif) Transferring hyperslab unions  
---  
The figure above shows the transfer of a selection that is two overlapping hyperslabs from the dataset into a union of hyperslabs in the memory dataset. Note that the destination dataset has a different shape from the source dataset. Similarly, the selection in the memory dataset could have a different shape than the selected union of hyperslabs in the original file. For simplicity, the selection is that same shape at the destination.
To implement this transfer, it is necessary to: 
  * 1. Get the source dataspace 
  * 2. Define one hyperslab selection for the source 
  * 3. Define a second hyperslab selection, unioned with the first 
  * 4. Get the destination dataspace 
  * 5. Define one hyperslab selection for the destination 
  * 6. Define a second hyperslab selection, unioned with the first 
  * 7. Execute the data transfer (H5Dread or H5Dwrite) using the source and destination dataspaces


The example below shows example code to create the selections for the source dataspace (the file). The first hyperslab is size 3 x 4 and the left upper corner at the position (1,2). The hyperslab is a simple rectangle, so the stride and block are 1. The second hyperslab is 6 x 5 at the position (2,4). The second selection is a union with the first hyperslab ([H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8)).
_Select source hyperslabs_
fid = [H5Dget_space](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gad42a46be153d895d8c28a11ebf5a0d0a)(dataset);
// Select first hyperslab for the dataset in the file.
offset[0] = 1; offset[1] = 2;
block[0] = 1; block[1] = 1;
stride[0] = 1; stride[1] = 1;
count[0] = 3; count[1] = 4;
ret = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(fid, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, stride, count, block);
// Add second selected hyperslab to the selection.
offset[0] = 2; offset[1] = 4;
block[0] = 1; block[1] = 1;
stride[0] = 1; stride[1] = 1;
count[0] = 6; count[1] = 5;
ret = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(fid, [H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8), offset, stride, count, block);
[H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8)
@ H5S_SELECT_OR
**Definition** H5Spublic.h:101
The example below shows example code to create the selection for the destination in memory. The steps are similar. In this example, the hyperslabs are the same shape, but located in different positions in the dataspace. The first hyperslab is 3 x 4 and starts at (0,0), and the second is 6 x 5 and starts at (1,2). Finally, the H5Dread call transfers the selected data from the file dataspace to the selection in memory. In this example, the source and destination selections are two overlapping rectangles. In general, any number of rectangles can be OR’ed, and they do not have to be contiguous. The order of the selections does not matter, but the first should use [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550) ; subsequent selections are unioned using [H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8).
It is important to emphasize that the source and destination do not have to be the same shape (or number of rectangles). As long as the two selections have the same number of elements, the data can be transferred.
_Select destination hyperslabs_
// Create memory dataspace.
mid = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(MSPACE_RANK, mdim, NULL);
// Select two hyperslabs in memory. Hyperslabs has the
// same size and shape as the selected hyperslabs for
// the file dataspace.
offset[0] = 0; offset[1] = 0;
block[0] = 1; block[1] = 1;
stride[0] = 1; stride[1] = 1;
count[0] = 3; count[1] = 4;
ret = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(mid, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, stride, count, block);
offset[0] = 1; offset[1] = 2;
block[0] = 1; block[1] = 1;
stride[0] = 1; stride[1] = 1;
count[0] = 6; count[1] = 5;
ret = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(mid, [H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8), offset, stride, count, block);
ret = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), mid, fid, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), matrix_out);
#### Selecting a List of Independent Points
It is also possible to specify a list of elements to read or write using the function H5Sselect_elements.
The procedure is similar to hyperslab selections. 
  * 1. Get the source dataspace 
  * 2. Set the selected points 
  * 3. Get the destination dataspace 
  * 4. Set the selected points 
  * 5. Transfer the data using the source and destination dataspaces


The figure below shows an example where four values are to be written to four separate points in a two dimensional dataspace. The source dataspace is a one dimensional array with the values 53, 59, 61, 67. The destination dataspace is an 8 x 12 array. The elements are to be written to the points (0,0), (3,3), (3,5), and (5,6). In this example, the source does not require a selection. The example below the figure shows example code to implement this transfer.
A point selection lists the exact points to be transferred and the order they will be transferred. The source and destination are required to have the same number of elements. A point selection can be used with a hyperslab (for example, the source could be a point selection and the destination a hyperslab, or vice versa), so long as the number of elements selected are the same.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_separate.gif) Write data to separate points  
---  
_Write data to separate points_
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dim2[] = {4};
int values[] = {53, 59, 61, 67};
// file dataspace
[hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280) coord[4][2];
// Create dataspace for the second dataset.
mid2 = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, dim2, NULL);
// Select sequence of NPOINTS points in the file dataspace.
coord[0][0] = 0; coord[0][1] = 0;
coord[1][0] = 3; coord[1][1] = 3;
coord[2][0] = 3; coord[2][1] = 5;
coord[3][0] = 5; coord[3][1] = 6;
ret = [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d)(fid, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), NPOINTS, (const [hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280) **)coord);
ret = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), mid2, fid, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), values);
[hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280)
int64_t hssize_t
**Definition** H5public.h:332
[H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d)
herr_t H5Sselect_elements(hid_t space_id, H5S_seloper_t op, size_t num_elem, const hsize_t *coord)
Selects array elements to be included in the selection for a dataspace.
#### Combinations of Selections
Selections are a very flexible mechanism for reorganizing data during a data transfer. With different combinations of dataspaces and selections, it is possible to implement many kinds of data transfers including sub‐setting, sampling, and reorganizing the data. The table below gives some example combinations of source and destination, and the operations they implement.
Selection operations Source |  Destination |  Operation  
---|---|---  
All |  All |  Copy whole array  
All |  All (different shape) |  Copy and reorganize array  
Hyperslab |  All |  Sub-set  
Hyperslab |  Hyperslab (same shape) |  Selection  
Hyperslab |  Hyperslab (different shape) |  Select and rearrange  
Hyperslab with stride or block |  All or hyperslab with stride 1 |  Sub-sample, scatter  
Hyperslab |  Points |  Scatter  
Points |  Hyperslab or all |  Gather  
Points |  Points (same) |  Selection  
Points |  Points (different) |  Reorder points  
##  Dataspace Selection Operations and Data Transfer
Dataspace selections play a critical role in HDF5 data transfer operations. When reading or writing data with [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") or [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset."), selections determine which elements are transferred between memory and the dataset. The HDF5 Library supports independent specification of selections for both the dataset (file dataspace) and the memory buffer (memory dataspace).
During data transfer, selections work as follows: 
  * The selection in the file dataspace identifies which elements to read from or write to in the dataset. 
  * The selection in the memory dataspace defines where to place the read data or where to retrieve the write data. 
  * Both selections must contain the same number of elements and should not exceed the dataset dimensions.


Additionally, as data is transferred, HDF5 automatically performs data type conversion between the file and memory representations if the source and destination data types differ.
Selection operations are designed to work with commonly structured patterns while also allowing for arbitrary point and hyperslab selections to provide maximum flexibility. These selections can be combined using set operations, such as [H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8) for a union and [H5S_SELECT_AND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0faae87b1594f2efe5f2717a865211d9418) for an intersection. You can then pass these combined selections to [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d "Selects a hyperslab region to add to the current selected region.") or [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d "Selects array elements to be included in the selection for a dataspace.") to efficiently create complex selection patterns.
For parallel I/O operations, collective data transfers can optimize performance when multiple processes access different selections of the same dataset simultaneously. See the parallel HDF5 documentation for details on collective I/O with selections.
##  References
Another use of selections is to store a reference to a region of a dataset in the file or an external file. An HDF5 object reference object is a pointer to an object (attribute, dataset, group, or committed datatype) in the file or an external file. A selection can be used to create a pointer to a set of selected elements of a dataset, called a region reference. The selection can be either a point selection or a hyperslab selection.
A region reference is an object maintained by the HDF5 Library. The region reference can be stored in a dataset or attribute, and then read. The dataset or attribute is defined to have the special datatype, [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc).
To discover the elements and/or read the data, the region reference can be dereferenced to obtain the identifiers for the dataset and dataspace.
For more information,  

See also
     [Reference](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#subsubsec_datatype_other_refs).
###  Example Uses for Region References
Region references are used to implement stored pointers to data within a dataset. For example, features in a large dataset might be indexed by a table. See the figure below. This table could be stored as an HDF5 dataset with a compound datatype, for example, with a field for the name of the feature and a region reference to point to the feature in the dataset. See the second figure below.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_features.gif) Features indexed by a table  
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_features_cmpd.gif) Storing the table with a compound datatype  
---  
###  Creating References to Regions
To create a region reference: 
  * 1. Create or open the dataset that contains the region 
  * 2. Get the dataspace for the dataset 
  * 3. Define a selection that specifies the region 
  * 4. Create a region reference using the dataset and dataspace with selection 
  * 5. Write the region reference(s) to the desired dataset or attribute 
  * 6. Release the region reference(s)


The figure below shows a diagram of a file with three datasets. Dataset D1 and D2 are two dimensional arrays of integers. Dataset R1 is a one dimensional array of references to regions in D1 and D2. The regions can be any valid selection of the dataspace of the target dataset. 
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_three_datasets.gif) A file with three datasets  
---  
_Note: In the figure above, R1 is a 1 D array of region pointers; each pointer refers to a selection in one dataset._
The example below shows code to create the array of region references. The references are created in an array of type [H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html). Each region is defined as a selection on the dataspace of the dataset, and a reference is created using [H5Rcreate_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b). The call to [H5Rcreate_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b) specifies the file, dataset, and the dataspace with selection.
_Create an array of region references_
// create an array of 4 region references
[H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) ref[4];
// Create a reference to the first hyperslab in the first Dataset.
offset[0] = 1; offset[1] = 1;
count[0] = 3; count[1] = 2;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(space_id, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, NULL, count, NULL);
status = [H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)(file_id, "D1", space_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &ref[0]);
// The second reference is to a union of hyperslabs in the first Dataset
offset[0] = 5; offset[1] = 3;
count[0] = 1; count[1] = 4;
status = [H5Sselect_none](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac59773d4a0953cb2db0ed57a05699095)(space_id);
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(space_id, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, NULL, count, NULL);
offset[0] = 6; offset[1] = 5;
count[0] = 1; count[1] = 2;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(space_id, [H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8), offset, NULL, count, NULL);
status = [H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)(file_id, "D1", space_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &ref[1]);
// the fourth reference is to a selection of points in the first Dataset
status = [H5Sselect_none](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac59773d4a0953cb2db0ed57a05699095)(space_id);
coord[0][0] = 4; coord[0][1] = 4;
coord[1][0] = 2; coord[1][1] = 6;
coord[2][0] = 3; coord[2][1] = 7;
coord[3][0] = 1; coord[3][1] = 5;
coord[4][0] = 5; coord[4][1] = 8;
status = [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d)(space_id, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), num_points, (const [hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280) **)coord);
status = [H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)(file_id, "D1", space_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &ref[3]);
// the third reference is to a hyperslab in the second Dataset
offset[0] = 0; offset[1] = 0;
count[0] = 4; count[1] = 6;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(space_id2, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, NULL, count, NULL);
status = [H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)(file_id, "D2", space_id2, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), &ref[2]);
[H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b)
herr_t H5Rcreate_region(hid_t loc_id, const char *name, hid_t space_id, hid_t oapl_id, H5R_ref_t *ref_ptr)
Creates a region reference.
[H5Sselect_none](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac59773d4a0953cb2db0ed57a05699095)
herr_t H5Sselect_none(hid_t spaceid)
Resets the selection region to include no elements.
[H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html)
**Definition** H5Rpublic.h:97
When all the references are created, the array of references is written to the dataset R1. The dataset is declared to have datatype [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc). See the example below. Also, note the release of the references afterwards.
_Write the array of references to a dataset_
Hsize_t dimsr[1];
dimsr[0] = 4;
// Dataset with references.
spacer_id = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, dimsr, NULL);
dsetr_id = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file_id, "R1", [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e), spacer_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f),
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Write dataset with the references.
status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dsetr_id, [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), ref);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&ref[0]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&ref[1]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&ref[0]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&ref[1]);
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484)
#define H5S_ALL
**Definition** H5Spublic.h:33
[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
#define H5Dcreate
**Definition** H5version.h:1124
[H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)
herr_t H5Rdestroy(H5R_ref_t *ref_ptr)
Closes a reference.
[H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e)
#define H5T_STD_REF_DSETREG
**Definition** H5Tpublic.h:571
When creating region references, the following rules are enforced. 
  * The selection must be a valid selection for the target dataset, just as when transferring data 
  * The dataset must exist in the file when the reference is created; [H5Rcreate_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga60134eb917afbe89aa23eb25a30d249b "Creates a region reference.")
  * The target dataset must be in the same file as the stored reference


###  Reading References to Regions
To retrieve data from a region reference, the reference must be read from the file, and then the data can be retrieved. The steps are: 
  * 1. Open the dataset or attribute containing the reference objects 
  * 2. Read the reference object(s) 
  * 3. For each region reference, get the dataset ([H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd "Opens the HDF5 object referenced.")) and dataspace ([H5Ropen_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga400154b014acd1736bae9a127f4e9a1a "Sets up a dataspace and selection as specified by a region reference.")) 
  * 4. Use the dataspace and datatype to discover what space is needed to store the data, allocate the correct storage and create a dataspace and datatype to define the memory data layout 
  * 5. Release the region reference(s)


The example below shows code to read an array of region references from a dataset, and then read the data from the first selected region. Note that the region reference has information that records the dataset (within the file) and the selection on the dataspace of the dataset. After dereferencing the regions reference, the datatype, number of points, and some aspects of the selection can be discovered. (For a union of hyperslabs, it may not be possible to determine the exact set of hyperslabs that has been combined.) The table below the code example shows the inquiry functions.
When reading data from a region reference, the following rules are enforced: 
  * The target dataset must be present and accessible in the file 
  * The selection must be a valid selection for the dataset


_Read an array of region references; read from the first selection_
dsetr_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) (file_id, "R1", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dsetr_id, H5T_STD, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), ref_out);
// Dereference the first reference.
// 1) get the dataset (H5Ropen_object)
// 2) get the selected dataspace (H5Ropen_region)
dsetv_id = [H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd)(&ref_out[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
space_id = [H5Ropen_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga400154b014acd1736bae9a127f4e9a1a)(&ref_out[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Discover how many points and shape of the data
ndims = [H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gae5282a81692b80b5b19dd12d05b9b28e)(space_id);
[H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3)(space_id,dimsx,NULL);
// Read and display hyperslab selection from the dataset.
dimsy[0] = [H5Sget_select_npoints](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga1a44dde97206f40f366f99d9c39b6046)(space_id);
spacex_id = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, dimsy, NULL);
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dsetv_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), space_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data_out);
printf("Selected hyperslab: ");
for (i = 0; i < 8; i++) {
printf("\n");
for (j = 0; j < 10; j++)
printf("%d ", data_out[i][j]);
}
printf("\n");
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&ref_out[0]);
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
[H5Ropen_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga400154b014acd1736bae9a127f4e9a1a)
hid_t H5Ropen_region(H5R_ref_t *ref_ptr, hid_t rapl_id, hid_t oapl_id)
Sets up a dataspace and selection as specified by a region reference.
[H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd)
hid_t H5Ropen_object(H5R_ref_t *ref_ptr, hid_t rapl_id, hid_t oapl_id)
Opens the HDF5 object referenced.
[H5Sget_select_npoints](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga1a44dde97206f40f366f99d9c39b6046)
hssize_t H5Sget_select_npoints(hid_t spaceid)
Determines the number of elements in a dataspace selection.
##  Deprecated References to Dataset Regions
The API described in this section was deprecated since HDF5 1.12.0. Shown are examples and usage in use by applications written before 1.12.0.
Another use of selections is to store a reference to a region of a dataset. An HDF5 object reference object is a pointer to an object (dataset, group, or committed datatype) in the file. A selection can be used to create a pointer to a set of selected elements of a dataset, called a region reference. The selection can be either a point selection or a hyperslab selection.
A region reference is an object maintained by the HDF5 Library. The region reference can be stored in a dataset or attribute, and then read. The dataset or attribute is defined to have the special datatype, [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e).
To discover the elements and/or read the data, the region reference can be dereferenced. The [H5Rdereference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a) call returns an identifier for the dataset, and then the selected dataspace can be retrieved with a call to [H5Rget_region()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f "Sets up a dataspace and selection as specified by a region reference."). The selected dataspace can be used to read the selected data elements.
###  Deprecated Creating References to Regions
To create a region reference: 
  * 1. Create or open the dataset that contains the region 
  * 2. Get the dataspace for the dataset 
  * 3. Define a selection that specifies the region 
  * 4. Create a region reference using the dataset and dataspace with selection 
  * 5. Write the region reference(s) to the desired dataset or attribute


The figure below shows a diagram of a file with three datasets. Dataset D1 and D2 are two dimensional arrays of integers. Dataset R1 is a one dimensional array of references to regions in D1 and D2. The regions can be any valid selection of the dataspace of the target dataset. 
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dspace_three_datasets.gif) A file with three datasets  
---  
_Note: In the figure above, R1 is a 1 D array of region pointers; each pointer refers to a selection in one dataset._
The example below shows code to create the array of region references. The references are created in an array of type [hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html). Each region is defined as a selection on the dataspace of the dataset, and a reference is created using [H5Rcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d). The call to [H5Rcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d) specifies the file, dataset, and the dataspace with selection.
_Deprecated Create an array of region references_
// create an array of 4 region references
[hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html) ref[4];
// Create a reference to the first hyperslab in the first Dataset.
offset[0] = 1; offset[1] = 1;
count[0] = 3; count[1] = 2;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(space_id, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, NULL, count, NULL);
status = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&ref[0], file_id, "D1", [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), space_id);
// The second reference is to a union of hyperslabs in the first Dataset
offset[0] = 5; offset[1] = 3;
count[0] = 1; count[1] = 4;
status = [H5Sselect_none](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac59773d4a0953cb2db0ed57a05699095)(space_id);
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(space_id, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, NULL, count, NULL);
offset[0] = 6; offset[1] = 5;
count[0] = 1; count[1] = 2;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(space_id, [H5S_SELECT_OR](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fa1a8dc2bbb8d8268d6e7665a4664b9ad8), offset, NULL, count, NULL);
status = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&ref[1], file_id, "D1", [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), space_id);
// the fourth reference is to a selection of points in the first Dataset
status = [H5Sselect_none](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac59773d4a0953cb2db0ed57a05699095)(space_id);
coord[0][0] = 4; coord[0][1] = 4;
coord[1][0] = 2; coord[1][1] = 6;
coord[2][0] = 3; coord[2][1] = 7;
coord[3][0] = 1; coord[3][1] = 5;
coord[4][0] = 5; coord[4][1] = 8;
status = [H5Sselect_elements](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2f4407dd73d0ec37e5d9e80e4382483d)(space_id, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), num_points, (const [hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280) **)coord);
status = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&ref[3], file_id, "D1", [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), space_id);
// the third reference is to a hyperslab in the second Dataset
offset[0] = 0; offset[1] = 0;
count[0] = 4; count[1] = 6;
status = [H5Sselect_hyperslab](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga6adfdf1b95dc108a65bf66e97d38536d)(space_id2, [H5S_SELECT_SET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a10093bab27cc5720efdab3186993da0fab90faf3dc59ecf6f28197ef471141550), offset, NULL, count, NULL);
status = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&ref[2], file_id, "D2", [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), space_id2);
[H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0)
#define H5R_DATASET_REGION
**Definition** H5Rpublic.h:625
[H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)
herr_t H5Rcreate(void *ref, hid_t loc_id, const char *name, H5R_type_t ref_type, hid_t space_id)
Creates a reference.
[hdset_reg_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhdset__reg__ref__t.html)
**Definition** H5Rpublic.h:86
When all the references are created, the array of references is written to the dataset R1. The dataset is declared to have datatype [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e). See the example below.
_Deprecated Write the array of references to a dataset_
Hsize_t dimsr[1];
dimsr[0] = 4;
// Dataset with references.
spacer_id = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, dimsr, NULL);
dsetr_id = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file_id, "R1", [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e), spacer_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f),
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Write dataset with the references.
status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dsetr_id, [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), ref);
When creating region references, the following rules are enforced. 
  * The selection must be a valid selection for the target dataset, just as when transferring data 
  * The dataset must exist in the file when the reference is created; [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d "Creates a reference.")
  * The target dataset must be in the same file as the stored reference


###  Deprecated Reading References to Regions
To retrieve data from a region reference, the reference must be read from the file, and then the data can be retrieved. The steps are: 
  * 1. Open the dataset or attribute containing the reference objects 
  * 2. Read the reference object(s) 
  * 3. For each region reference, get the dataset ([H5Rdereference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a)) and dataspace ([H5Rget_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f "Sets up a dataspace and selection as specified by a region reference.")) 
  * 4. Use the dataspace and datatype to discover what space is needed to store the data, allocate the correct storage and create a dataspace and datatype to define the memory data layout


The example below shows code to read an array of region references from a dataset, and then read the data from the first selected region. Note that the region reference has information that records the dataset (within the file) and the selection on the dataspace of the dataset. After dereferencing the regions reference, the datatype, number of points, and some aspects of the selection can be discovered. (For a union of hyperslabs, it may not be possible to determine the exact set of hyperslabs that has been combined.) The table below the code example shows the inquiry functions.
When reading data from a region reference, the following rules are enforced: 
  * The target dataset must be present and accessible in the file 
  * The selection must be a valid selection for the dataset


_Deprecated Read an array of region references; read from the first selection_
dsetr_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) (file_id, "R1", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dsetr_id, [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), ref_out);
// Dereference the first reference.
// 1) get the dataset (H5Rdereference)
// 2) get the selected dataspace (H5Rget_region)
dsetv_id = [H5Rdereference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a)(dsetr_id, [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), &ref_out[0]);
space_id = [H5Rget_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f)(dsetr_id, [H5R_DATASET_REGION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#a2a28c48c03a4616a02f9157fca4b2df0), &ref_out[0]);
// Discover how many points and shape of the data
ndims = [H5Sget_simple_extent_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gae5282a81692b80b5b19dd12d05b9b28e)(space_id);
[H5Sget_simple_extent_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gac494409b615d8e67c5edd9eb2848b2f3)(space_id,dimsx,NULL);
// Read and display hyperslab selection from the dataset.
dimsy[0] = [H5Sget_select_npoints](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga1a44dde97206f40f366f99d9c39b6046)(space_id);
spacex_id = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)(1, dimsy, NULL);
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dsetv_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), space_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data_out);
printf("Selected hyperslab: ");
for (i = 0; i < 8; i++) {
printf("\n");
for (j = 0; j < 10; j++)
printf("%d ", data_out[i][j]);
}
printf("\n");
[H5Rget_region](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga1702d609e85b9edd3d1e526a0276484f)
hid_t H5Rget_region(hid_t dataset, H5R_type_t ref_type, const void *ref)
Sets up a dataspace and selection as specified by a region reference.
[H5Rdereference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a)
#define H5Rdereference
**Definition** H5version.h:1471
##  Functions
The inquiry functions Function |  Information  
---|---  
[H5Sget_select_npoints](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga1a44dde97206f40f366f99d9c39b6046) |  The number of elements in the selection (hyperslab or point selection).  
[H5Sget_select_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga645591ec939b89732c10efd5867a6205) |  The bounding box that encloses the selected points (hyperslab or point selection).  
[H5Sget_select_hyper_nblocks](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#gad873b2f3b82ee8c426c26ceeb1c67f86) |  The number of blocks in the selection.  
[H5Sget_select_hyper_blocklist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8534829a8db2eca8e987bb9fe8a3d628) |  A list of the blocks in the selection.  
[H5Sget_select_elem_npoints](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga217b839584cd7c7995b47fc30fe92f4c) |  The number of points in the selection.  
[H5Sget_select_elem_pointlist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga61459c488147254d1d06537a9ab6e2d4) |  The points.  
##  Sample Programs
This section contains the full programs from which several of the code examples in this chapter were derived. The [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) output from the program's output file immediately follows each program.
_h5_write.c_
#include "hdf5.h"
#define H5FILE_NAME "SDS.h5"
#define DATASETNAME "C Matrix"
#define NX 3
#define NY 5
#define RANK 2 // dataset dimensions
int
main (void)
{
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, dataset; // file and dataset identifiers
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) datatype, dataspace; // identifiers
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dims[2]; // dataset dimensions
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) status;
int data[NX][NY]; // data to write
int i, j;
//
// Data and output buffer initialization.
for (j = 0; j < NX; j++) {
for (i = 0; i < NY; i++)
data[j][i] = i + 1 + j*NY;
}
// 1 2 3 4 5
// 6 7 8 9 10
// 11 12 13 14 15
// Create a new file using H5F_ACC_TRUNC access,
// default file creation properties, and default file
// access properties.
file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(H5FILE_NAME, [H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Describe the size of the array and create the data space for fixed
// size dataset.
dims[0] = NX;
dims[1] = NY;
dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dims, NULL);
// Create a new dataset within the file using defined dataspace and
// datatype and default dataset creation properties.
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, DATASETNAME, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f),
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Write the data to the dataset using default transfer properties.
status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data);
// Close/release resources.
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)(dataspace);
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)(dataset);
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
return 0;
}
SDS.out
-------
HDF5 "SDS.h5" {
GROUP "/" {
DATASET "C Matrix" {
DATATYPE [H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8)
DATASPACE SIMPLE { ( 3, 5 ) / ( 3, 5 ) }
DATA {
1, 2, 3, 4, 5,
6, 7, 8, 9, 10,
11, 12, 13, 14, 15
}
}
[H5F_ACC_TRUNC](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a5a2d6726f9ad8d2bca8df2b817e5ad6a)
#define H5F_ACC_TRUNC
**Definition** H5Fpublic.h:30
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Dclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gae47c3f38db49db127faf221624c30609)
herr_t H5Dclose(hid_t dset_id)
Closes the specified dataset.
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)
herr_t H5Fclose(hid_t file_id)
Terminates access to an HDF5 file.
[H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)
hid_t H5Fcreate(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
Creates an HDF5 file.
[H5Sclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga2b53128a39c8f104c1c9c2a91590fcc1)
herr_t H5Sclose(hid_t space_id)
Releases and terminates access to a dataspace.
[H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8)
#define H5T_STD_I32BE
**Definition** H5Tpublic.h:466
_h5_write.f90_
----------
PROGRAM DSETEXAMPLE
USE HDF5 ! This module contains all necessary modules
IMPLICIT NONE
CHARACTER(LEN=7), PARAMETER :: filename = "SDSf.h5" ! File name
CHARACTER(LEN=14), PARAMETER :: dsetname = "Fortran Matrix" ! Dataset name
INTEGER, PARAMETER :: NX = 3
INTEGER, PARAMETER :: NY = 5
INTEGER(HID_T) :: file_id ! File identifier
INTEGER(HID_T) :: dset_id ! Dataset identifier
INTEGER(HID_T) :: dspace_id ! Dataspace identifier
INTEGER(HSIZE_T), DIMENSION(2) :: dims = (/3,5/) ! Dataset dimensions
INTEGER :: rank = 2 ! Dataset rank
INTEGER :: data(NX,NY)
INTEGER :: error ! Error flag
INTEGER :: i, j
!
! Initialize data
!
do i = 1, NX
do j = 1, NY
data(i,j) = j + (i-1)*NY
enddo
enddo
!
! Data
!
! 1 2 3 4 5
! 6 7 8 9 10
! 11 12 13 14 15
!
! Initialize FORTRAN interface.
!
CALLh5open_f(error)
!
! Create a new file using default properties.
!
CALL h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, error)
!
! Create the dataspace.
!
CALL h5screate_simple_f(rank, dims, dspace_id, error)
!
! Create and write dataset using default properties.
!
CALL h5dcreate_f(file_id, dsetname, H5T_NATIVE_INTEGER, dspace_id, &
dset_id, error, H5P_DEFAULT_F, H5P_DEFAULT_F, &
H5P_DEFAULT_F)
CALL h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, data, dims, error)
!
! End access to the dataset and release resources used by it.
!
CALL h5dclose_f(dset_id, error)
!
! Terminate access to the data space.
!
CALL h5sclose_f(dspace_id, error)
!
! Close the file.
!
CALL h5fclose_f(file_id, error)
!
! Close FORTRAN interface.
!
CALL h5close_f(error)
END PROGRAM DSETEXAMPLE
SDSf.out
--------
HDF5 "SDSf.h5" {
GROUP "/" {
DATASET "Fortran Matrix" {
DATATYPE [H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8)
DATASPACE SIMPLE { ( 5, 3 ) / ( 5, 3 ) }
DATA {
1, 6, 11,
2, 7, 12,
3, 8, 13,
4, 9, 14,
5, 10, 15
}
}
}
}
_h5_write_tr.f90_
PROGRAM DSETEXAMPLE
USE HDF5 ! This module contains all necessary modules
IMPLICIT NONE
CHARACTER(LEN=10), PARAMETER :: filename = "SDSf_tr.h5" ! File name
CHARACTER(LEN=24), PARAMETER :: dsetname = "Fortran Transpose Matrix"! Dataset name
INTEGER, PARAMETER :: NX = 3
INTEGER, PARAMETER :: NY = 5
INTEGER(HID_T) :: file_id ! File identifier
INTEGER(HID_T) :: dset_id ! Dataset identifier
INTEGER(HID_T) :: dspace_id ! Dataspace identifier
INTEGER(HSIZE_T), DIMENSION(2) :: dims = (/NY, NX/) ! Dataset dimensions
INTEGER :: rank = 2 ! Dataset rank
INTEGER :: data(NY,NX)
INTEGER :: error ! Error flag
INTEGER :: i, j
!
! Initialize data
!
do i = 1, NY
do j = 1, NX
data(i,j) = i + (j-1)*NY
enddo
enddo
!
! Data
!
! 1 6 11
! 2 7 12
! 3 8 13
! 4 9 14
! 5 10 15
!
! Initialize FORTRAN interface.
!
CALL h5open_f(error)
!
! Create a new file using default properties.
!
CALL h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, error)
!
! Create the dataspace.
!
CALL h5screate_simple_f(rank, dims, dspace_id, error)
!
! Create and write dataset using default properties.
!
CALL h5dcreate_f(file_id, dsetname, H5T_NATIVE_INTEGER, dspace_id, &
dset_id, error, H5P_DEFAULT_F, H5P_DEFAULT_F, &
H5P_DEFAULT_F)
CALL h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, data, dims, error)
!
! End access to the dataset and release resources used by it.
!
CALL h5dclose_f(dset_id, error)
!
! Terminate access to the data space.
!
CALL h5sclose_f(dspace_id, error)
!
! Close the file.
!
CALL h5fclose_f(file_id, error)
!
! Close FORTRAN interface.
!
CALL h5close_f(error)
END PROGRAM DSETEXAMPLE
SDSf_tr.out
-----------
HDF5 "SDSf_tr.h5" {
GROUP "/" {
DATASET "Fortran Transpose Matrix" {
DATATYPE [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
DATASPACE SIMPLE { ( 3, 5 ) / ( 3, 5 ) }
DATA {
1, 2, 3, 4, 5,
6, 7, 8, 9, 10,
11, 12, 13, 14, 15
}
}
}
}
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
#define H5T_STD_I32LE
**Definition** H5Tpublic.h:471
Previous Chapter [HDF5 Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#sec_datatype) - Next Chapter [HDF5 Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_a__u_g.html#sec_attribute)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


