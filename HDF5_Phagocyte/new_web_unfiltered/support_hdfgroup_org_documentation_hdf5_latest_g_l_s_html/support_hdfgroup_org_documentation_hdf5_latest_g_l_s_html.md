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
  * [ A](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title0 "H1")
  * [ B](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title1 "H1")
  * [ C](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title2 "H1")
  * [ D](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title3 "H1")
  * [ E](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title4 "H1")
  * [ F](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title5 "H1")
  * [ G](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title6 "H1")
  * [ H](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title7 "H1")
  * [ I](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title8 "H1")
  * [ L](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title9 "H1")
  * [ M](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title10 "H1")
  * [ N](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title11 "H1")
  * [ O](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title12 "H1")
  * [ P](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title13 "H1")
  * [ R](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title14 "H1")
  * [ S](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title15 "H1")
  * [ U](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title16 "H1")
  * [ V](https://support.hdfgroup.org/documentation/hdf5/latest/_g_l_s.html#title17 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Glossary
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html)
* * *
#  A 

Array datatype 
    A family of HDF5 datatypes whose elements are arrays of a fixed rank (≤ 32) and fixed finite extent. All array elements must be of the same HDF5 datatype.  

Array variable 
    
A variable that can store (logically) dense, rectilinear, multidimensional arrays of elements of a given HDF5 datatype.
The combination of array rank (dimensionality) and extent is called an array variable's shape. This includes the degenerate array shapes of a singleton (scalar) and the empty array (null).
The array element datatype is sometimes referred to as the array variable's type, which is not entirely accurate because the array variable's type is 'array of element type' rather than 'element type'.
In HDF5, there are two kinds of array variables, attributes and datasets, and the distinction is functional (i.e., how they can be used) rather than conceptual. Attributes are commonly used for descriptive "light-weight" HDF5 object metadata while datasets are HDF5 objects used to store "heavy-weight" problem-sized data. 

Attribute 
    
A named array variable that is associated with an HDF5 object, its owner or attributee, and used to represent application domain-specific metadata of the object. Intuitively, the set of an object's attributes can be thought of as its key-value pair collection. Attribute names (keys) can be arbitrary Unicode strings, but must be unique per object, i.e., an object can have at most one attribute with a given name.
A scalar attribute is an attribute backed by a singleton array variable. A null attribute is attribute backed by an empty array variable.
#  B 

Bitfield datatype 
    A family of HDF5 datatypes whose elements are fixed-width bit fields. 
#  C 

Chunked layout 
    
A dataset storage layout where the dataset elements are partitioned into fixed-size multidimensional chunks or tiles. Chunked layout is mandatory for datasets with one or more dimensions of indefinite (infinite) extent or where compression or other filters are applied to the dataset elements.
Chunked layout may improve I/O performance for certain access patterns. 

Committed datatype 
    An immutable kind of HDF5 object that is used to store an HDF5 datatype definition, which can be referenced by multiple array variables. When linked to an HDF5 group, a committed datatype can be located by an HDF5 path name, and is sometimes called a named datatype.  

Compact layout 
    
A dataset storage layout where the dataset elements are stored in the object header of the dataset. This layout is suitable for very small datasets that can easily fit in the object header.
Compact layout can improve storage and access performance for files that have many very small datasets. 

Compound datatype 
    
A family of HDF5 datatypes whose elements are records with named fields of other HDF5 datatypes. Currently, on ASCII field names are supported.
Similar to a `struct` in C or a `COMMON` block in Fortran. 

Contiguous layout 
    A dataset storage layout where the dataset elements are physically stored in an HDF5 file as a contiguous block of bytes. 
#  D 

Dataset 
    
A kind of HDF5 object, a linked array variable. which can be located in an HDF5 file through a path name. Datasets are commonly used to store "heavy-weight" problem-sized data.
The HDF5 library offers a lot of features aimed at optimized dataset access and storage, including compression and partial I/O. 

Dataspace 
    The shape of an array variable. With the exception of degenerate cases (empty set, singleton), this is a rectilinear lattice or grid of a certain rank (dimensionality) and extent.  

Datatype 
    
An HDF5 datatype consists of an abstract data type (a set of elements) and a bit-level representation of these elements in storage such as an HDF5 file or memory.
The HDF5 library comes with a large set of predefined datatypes and offers mechanisms for creating user-defined datatypes.
The ten major families or classes of HDF5 datatypes are:
  * Integer datatypes 
  * Floating-point number datatypes 
  * String datatypes 
  * Bitfield datatypes 
  * Opaque datatypes 
  * Compound datatypes 
  * Reference datatypes 
  * Enumerated datatypes 
  * Variable-length sequence datatypes 
  * Array datatypes 


#  E 

Enumeration datatype 
    A family of HDF5 datatypes whose elements represent named integer values called members or enumerators. Currently, only ASCII names are supported.  

External layout 
    A form of contiguous layout where a dataset's elements are physically stored in unformatted binary files outside the HDF5 file.  

External link 
    An HDF5 link whose destination is specified as a pair of an HDF5 file name and an HDF5 path name in that file. 
#  F 

Field 
    See compound datatype.  

File 
    
  1. A byte stream (in a storage context such as a file system or in memory) formatted according to the HDF5 File Format Specification. 
  2. A (logical) container for HDF5 objects. 



File format 
    HDF5 file format refers to the structure and organization of the HDF5 data being stored within the file.  

Fill value 
    A default value assigned to data elements that have not yet been written.  

Filter 
    Filters are optional stages that can be inserted in the data pipeline to implement compression and error checking. User applications may also add custom filters. 
#  G 

Group 
    
A kind of HDF5 object that stores a collection of HDF5 links. Each HDF5 file contains at least one group, it's root group.
Among the destinations of an HDF5 group's links may be other HDF5 groups (including the group itself!). This ability is sometimes referred to as the closure property of groups. It is the basis for creating hierarchical or more general graph-like structures.
#  H 

Hard link 
    An HDF5 link whose destination is specified (internally) as the address of an HDF5 object in the same HDF5 file.  

Hierarchy 
    See group.  

Hyperslab 
    
A regular multidimensional pattern described by four vectors whose length equals the rank of the pattern.
  1. `start` - the offset where the first block of the hyperslab begins 
  2. `stride` - the offset between pattern blocks 
  3. `count` - the number of blocks 
  4. `block` - the extent of an individual pattern block 


For example, the black squares on a (two-dimensional) chessboard with origin at `(0,0)` can be represented as the union of two hyperslabs representing the even `(0,2,4,6)` and odd `(1,3,5,7)` rows, respectively.
![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Chessboard480.svg/176px-Chessboard480.svg.png)
The hyperslab parameters for the even rows are: `start (0,0)`, `stride (2,2)`, `count (4,4)`, `block (1,1)`. Likewise the parameters for the odd rows are: `start (1,1)`, `stride (2,2)`, `count (4,4)`, `block (1,1)`.
#  I 

Identifier 
    An opaque, transient handle used by the HDF5 library to manipulate in-memory representations of HDF5 items. 
#  L 

Library 


Link 
    
A named, uni-directional association between a source and a destination. In HDF5, the source is always the HDF5 group that hosts the link in its link collection.
There are several ways to specify a link's destination:
  * The address of an HDF5 object in the same HDF5 file; so-called hard link. 
  * A path name in the same or a different file; so-called soft or external link. 
  * User-defined 


A link name can be any Unicode string that does not contain slashes (`"/"`) or consists of a single dot character (`"."`). A link name must be unique in a group's link collection.
#  M 

Metadata 
    Data that in a given context has a descriptive or documentation function for other data. Typically, the metadata is small compared to the data it describes.  

Member 
    
A link destination is sometimes referred to as a member of the link's source (group). This way of speaking invites confusion: A destination (e.g., object) can be the destination of multiple links in the same (!) or different groups. It would then be a "member" of a given group with multiplicity greater than one and be a member of multiple groups.
It is the link that is a member of the group's link collection and not the link destination.
#  N 

Name 
    
A Unicode string that depending on the item it names might be subject to certain character restrictions, such as ASCII-encoded only. In HDF5, the user might encounter the following names:
  * A link name 
  * A path name 
  * An attribute name 
  * A field name (compound datatypes) 
  * A constant name (enumeration datatypes) 
  * A tag name (opaque datatypes) 
  * A file name 



Named datatype 
    See committed datatype.  

Null dataspace 
    A shape which represents the empty set. Array variables with this shape cannot store any values. 
#  O 

Object 
    An HDF5 group, dataset or named datatype; an HDF5 item that can be linked to zero or more groups and decorated with zero or more HDF5 attributes.  

Object reference 
    
  1. A datatype for representing references to objects in a file. 
  2. A value of the object reference datatype. 



Opaque datatype 
    A family of HDF5 datatypes whose elements are byte sequences of a given fixed length. An opaque datatype can be tagged with a sequence of up to 256 ASCII characters, e.g., MIME code. 
#  P 

Path name 
    A Unicode string that is the concatenation of link names separated by slashes (`'/'`). In HDF5, path names are used to locate and refer to HDF5 objects.  

Plugin 
    An HDF5 library feature or capability that can be added dynamically at application run time rather than library compilation time. Plugins are usually implemented as shared libraries, and their discovery and loading behavior can be controlled programmatically or through environment variables.  

Point selection 
    A dataspace selection that consists of a set of points (coordinates) in the same dataspace.  

Property list 
    
An HDF5 API construct, a means of customizing the behavior of the HDF5 library when creating, accessing or modifying HDF5 items.
While the default property settings are sufficient in many cases, certain HDF5 features, such as compression, can be reasonably controlled only by the user who has to provide the desired settings via property lists.
#  R 

Rank 
    The number of dimensions of a non-null dataspace.  

Reference 
    
  1. An HDF5 object reference 
  2. An HDF5 dataset region reference 



Reference datatype 
    
  1. An HDF5 datatype whose elements represent references to HDF5 objects. 
  2. An HDF5 datatype whose elements represent references to regions of an HDF5 dataset. 



Region reference 
    See dataset region reference.  

Root group 
    
An HDF5 group that is present in all HDF5 files and that acts as the entry or base point for all other data stored in an HDF5 file.
The root group is "the mother of all objects" in an HDF5 file in the sense that all objects (and their attributes) can be discovered, beginning at the root group, by combinations of the following operations:
  * Link traversal 
  * De-referencing of object references 


This discovery is portable and robust with respect to file-internal storage reorganization.
#  S 

Scalar dataspace 
    A kind of HDF5 dataspace that has the shape of a singleton, i.e., a set containing a single element. Array variables with this shape store exactly one element.  

Selection 
    
  1. A subset of points of an HDF5 dataspace. The subset might be a point selection or a combination (union, intersection, etc.) of hyperslabs. 
  2. A subset of dataset elements associated with a dataspace selection as described under 1. 



Serialization 
    
  1. The flattening of an N-dimensional array into a 1-dimensional array. 
  2. The encoding of a complex data item as a linear byte stream. 



Soft link 
    A kind of HDF5 link in which the link destination is specified as an HDF5 path name. The path name may or may not refer to an actual object.  

Storage layout 
    The storage arrangement for dataset elements, links in a group's link collection, or attributes in an object's attribute collection.  

String datatype 


Super block 
    An HDF5 file format primitive; a block of data which contains information required to access HDF5 files in a portable manner on multiple platforms. The super block contains information such as version numbers, the size of offsets and lengths, and the location of the root group.  

SWMR 
    Single Writer Multiple Reader, a file access mode in which a single process is permitted to write data to an HDF5 file while other processes are permitted to read data from the same file without the need of inter-process communication or synchronization.  

Symbolic link 
    An external link or a soft link. 
#  U 

User block 
    An HDF5 file format primitive that allows one to set aside a fixed-size (at least 512 bytes or any power of 2 thereafter) contiguous range of bytes at the beginning of an HDF5 file for application purposes which will be skipped/ignored by the HDF5 library.  

UTF-8 
    
A variable-length (1-4 bytes per code point) encoding of the Unicode set of code points. This is the encoding supported by HDF5 to represent Unicode strings.
The ASCII encoding is a proper subset of UTF-8.
#  V 

Variable-length (sequence) datatype 
    A family of HDF5 datatypes whose elements are variable-length sequences of a given datatype.  

Virtual Dataset (VDS) 
    An HDF5 dataset with virtual storage layout. A dataset whose elements are partially or entirely stored physically in other datasets.  

Virtual File Driver (VFD) 


Virtual layout 


Virtual Object Layer (VOL) 

* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


