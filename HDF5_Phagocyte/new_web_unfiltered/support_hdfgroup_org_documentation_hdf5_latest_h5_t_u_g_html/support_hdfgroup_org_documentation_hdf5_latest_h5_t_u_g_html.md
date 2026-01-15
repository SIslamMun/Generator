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
  * [ HDF5 Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title0 "H1")
  * [ Introduction and Definitions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title1 "H2")
  * [ Datatype Model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title2 "H2")
  * [ How Datatypes are Used](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title3 "H2")
  * [ Datatype Function Summaries](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title4 "H2")
  * [ Programming Model for Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title5 "H2")
  * [ Other Non-numeric Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title6 "H2")
  * [ Fill Values](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title7 "H2")
  * [ Complex Combinations of Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title8 "H2")
  * [ Life Cycle of the Datatype Object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title9 "H2")
  * [ Data Transfer: Datatype Conversion and Selection](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title10 "H2")
  * [ Text Descriptions of Datatypes: Conversion to and from](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t__u_g.html#title11 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Datatypes
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 Datatypes
HDF5 datatypes describe the element type of HDF5 datasets and attributes. There's a large set of predefined datatypes, but users may find it useful to define new datatypes through a process called _derivation_.
The element type is automatically persisted as part of the HDF5 metadata of attributes and datasets. Additionally, datatype definitions can be persisted to HDF5 files and linked to groups as HDF5 datatype objects or so-called _committed datatypes_.
##  Introduction and Definitions
An HDF5 dataset is an array of data elements, arranged according to the specifications of the dataspace. In general, a data element is the smallest addressable unit of storage in the HDF5 file. (Compound datatypes are the exception to this rule.) The HDF5 datatype defines the storage format for a single data element. See the figure below.
The model for HDF5 attributes is extremely similar to datasets: an attribute has a dataspace and a data type, as shown in the figure below. The information in this chapter applies to both datasets and attributes.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig1.gif) Datatypes, dataspaces, and datasets  
---  
Abstractly, each data element within the dataset is a sequence of bits, interpreted as a single value from a set of values (for example, a number or a character). For a given datatype, there is a standard or convention for representing the values as bits, and when the bits are represented in a particular storage the bits are laid out in a specific storage scheme such as 8-bit bytes with a specific ordering and alignment of bytes within the storage array.
HDF5 datatypes implement a flexible, extensible, and portable mechanism for specifying and discovering the storage layout of the data elements, determining how to interpret the elements (for example, as floating point numbers), and for transferring data from different compatible layouts.
An HDF5 datatype describes one specific layout of bits. A dataset has a single datatype which applies to every data element. When a dataset is created, the storage datatype is defined. After the dataset or attribute is created, the datatype cannot be changed. 
  * The datatype describes the storage layout of a single data element 
  * All elements of the dataset must have the same type 
  * The datatype of a dataset is immutable


When data is transferred (for example, a read or write), each end point of the transfer has a datatype, which describes the correct storage for the elements. The source and destination may have different (but compatible) layouts, in which case the data elements are automatically transformed during the transfer.
HDF5 datatypes describe commonly used binary formats for numbers (integers and floating point) and characters (ASCII). A given computing architecture and programming language supports certain number and character representations. For example, a computer may support 8-, 16-, 32-, and 64-bit signed integers, stored in memory in little-endian byte order. These would presumably correspond to the C programming language types _char_ , _short_ , _int_ , and _long_.
When reading and writing from memory, the HDF5 library must know the appropriate datatype that describes the architecture specific layout. The HDF5 library provides the platform independent _NATIVE_ types, which are mapped to an appropriate datatype for each platform. So the type [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) is an alias for the appropriate descriptor for each platform.
Data in memory has a datatype: 
  * The storage layout in memory is architecture-specific 
  * The HDF5 _NATIVE_ types are predefined aliases for the architecture-specific memory layout 
  * The memory datatype need not be the same as the stored datatype of the dataset


In addition to numbers and characters, an HDF5 datatype can describe more abstract classes of types including enumerations, strings, bit strings, and references (pointers to objects in the HDF5 file). HDF5 supports several classes of composite datatypes which are combinations of one or more other datatypes. In addition to the standard predefined datatypes, users can define new datatypes within the datatype classes.
The HDF5 datatype model is very general and flexible: 
  * For common simple purposes, only predefined types will be needed 
  * Datatypes can be combined to create complex structured datatypes 
  * If needed, users can define custom atomic datatypes 
  * Committed datatypes can be shared by datasets or attributes


##  Datatype Model
The HDF5 library implements an object-oriented model of datatypes. HDF5 datatypes are organized as a logical set of base types, or datatype classes. Each datatype class defines a format for representing logical values as a sequence of bits. For example the [H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037) class is a format for representing twos complement integers of various sizes.
A datatype class is defined as a set of one or more datatype properties. A datatype property is a property of the bit string. The datatype properties are defined by the logical model of the datatype class. For example, the integer class (twos complement integers) has properties such as “signed or unsigned”, “length”, and “byte-order”. The float class (IEEE floating point numbers) has these properties, plus “exponent bits”, “exponent sign”, etc.
A datatype is derived from one datatype class: a given datatype has a specific value for the datatype properties defined by the class. For example, for 32-bit signed integers, stored big-endian, the HDF5 datatype is a sub-type of integer with the properties set to signed=1, size=4(bytes), and byte-order=BE.
The HDF5 datatype API (H5T functions) provides methods to create datatypes of different datatype classes, to set the datatype properties of a new datatype, and to discover the datatype properties of an existing datatype.
The datatype for a dataset is stored in the HDF5 file as part of the metadata for the dataset. A datatype can be shared by more than one dataset in the file if the datatype is saved to the file with a name. This shareable datatype is known as a committed datatype. In the past, this kind of datatype was called a named datatype.
When transferring data (for example, a read or write), the data elements of the source and destination storage must have compatible types. As a general rule, data elements with the same datatype class are compatible while elements from different datatype classes are not compatible. When transferring data of one datatype to another compatible datatype, the HDF5 Library uses the datatype properties of the source and destination to automatically transform each data element. For example, when reading from data stored as 32-bit signed integers, big endian into 32-bit signed integers, little-endian, the HDF5 Library will automatically swap the bytes.
Thus, data transfer operations ([H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c), [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906), [H5Aread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaacb27a997f7c98e8a833d0fd63b58f1c), [H5Awrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gab70871e205d57450c83efd9912be2b5c)) require a datatype for both the source and the destination.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig2.gif) The datatype model  
---  
The HDF5 library defines a set of predefined datatypes, corresponding to commonly used storage formats, such as twos complement integers, IEEE Floating point numbers, etc., 4- and 8-byte sizes, big-endian and little-endian byte orders. In addition, a user can derive types with custom values for the properties. For example, a user program may create a datatype to describe a 6-bit integer, or a 600-bit floating point number.
In addition to atomic datatypes, the HDF5 library supports composite datatypes. A composite datatype is an aggregation of one or more datatypes. Each class of composite datatypes has properties that describe the organization of the composite datatype. See the figure below. Composite datatypes include: 
  * Compound datatypes: structured records 
  * Array: a multidimensional array of a datatype 
  * Variable-length: a one-dimensional array of a datatype 
  * Enumeration: a set of (name, value) pairs, similar to the C/C++ enum type 
  * Complex: an aggregate of two similar floating-point datatypes

![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig3.gif) Composite datatypes  
---  
###  Datatype Classes and Properties
The figure below shows the HDF5 datatype classes. Each class is defined to have a set of properties which describe the layout of the data element and the interpretation of the bits. The table below lists the properties for the datatype classes.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig4.gif) Datatype classes  
---  
Datatype classes and their properties Class  | Description  | Properties  | Notes   
---|---|---|---  
Integer  | Twos complement integers  | Size (bytes), precision (bits), offset (bits), pad, byte order, signed/unsigned  |   
Float  | Floating Point numbers  | Size (bytes), precision (bits), offset (bits), pad, byte order, sign position, exponent position, exponent size (bits), exponent sign, exponent bias, mantissa position, mantissa (size) bits, mantissa sign, mantissa normalization, internal padding  | See IEEE 754 for a definition of these properties. These properties describe non-IEEE 754 floating point formats as well.   
Character  | Array of 1-byte character encoding  | Size (characters), Character set, byte order, pad/no pad, pad character  | Currently, ASCII and UTF-8 are supported.   
Bitfield  | String of bits  | Size (bytes), precision (bits), offset (bits), pad, byte order  | A sequence of bit values packed into one or more bytes.   
Opaque  | Uninterpreted data  | Size (bytes), precision (bits), offset (bits), pad, byte order, tag  | A sequence of bytes, stored and retrieved as a block. The ‘tag’ is a string that can be used to label the value.   
Enumeration  | A list of discrete values, with symbolic names in the form of strings.  | Number of elements, element names, element values  | Enumeration is a list of pairs (name, value). The name is a string; the value is an unsigned integer.   
Reference  | Reference to object or region within the HDF5 file  |  |  

See also
     [References (H5R)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html)  
Array  | Array (1-4 dimensions) of data elements  | Number of dimensions, dimension sizes, base datatype  | The array is accessed atomically: no selection or sub-setting.   
Variable-length  | A variable-length 1-dimensional array of data elements  | Current size, base type  |   
Compound  | A Datatype of a sequence of Datatypes  | Number of members, member names, member types, member offset, member class, member size, byte order  |   
Complex  | Data elements of two floating point numbers  | Base floating point datatype  | Other properties inherited from base floating point datatype   
###  Predefined Datatypes
The HDF5 library predefines a modest number of commonly used datatypes. These types have standard symbolic names of the form H5T_arch_base where arch is an architecture name and base is a programming type name **Table 2**. New types can be derived from the predefined types by copying the predefined type [H5Tcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e) and then modifying the result.
The base name of most types consists of a letter to indicate the class **Table 3** , a precision in bits, and an indication of the byte order **Table 4**.
**Table 5** shows examples of predefined datatypes. The full list can be found in the [Predefined Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t.html) section of the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
Table 2. Architectures used in predefined datatypes Architecture Name  | Description   
---|---  
IEEE  | IEEE-754 standard floating point types in various byte orders.   
STD  | This is an architecture that contains semi-standard datatypes like signed two's complement integers, unsigned integers, and bitfields in various byte orders.   
C <br > FORTRAN  | Types which are specific to the C or Fortran programming languages are defined in these architectures. For instance, [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d) defines a base string type with null termination which can be used to derive string types of other lengths.   
NATIVE  | This architecture contains C-like datatypes for the machine for which the library was compiled. In order to be portable, applications should almost always use this architecture to describe things in memory.   
CRAY  | Cray architectures. These are word-addressable, big-endian systems with non-IEEE floating point.   
INTEL  | All Intel and compatible CPUs. These are little-endian systems with IEEE floating-point.   
MIPS  | All MIPS CPUs commonly used in SGI systems. These are big-endian systems with IEEE floating-point.   
ALPHA  | All DEC Alpha CPUs, little-endian systems with IEEE floating-point.   
Table 3. Base types Base  | Description   
---|---  
B  | Bitfield   
F  | Floating point   
I  | Signed integer   
R  | References   
S  | Character string   
U  | Unsigned integer   
Table 4. Byte order Order  | Description   
---|---  
BE  | Big-endian   
LE  | Little-endian   
Table 5. Some predefined datatypes Example  | Description   
---|---  
[H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e) | Eight-byte, little-endian, IEEE floating-point   
[H5T_IEEE_F32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga71d24a7d4c373ed9a003d7a0d8133f1e) | Four-byte, big-endian, IEEE floating point   
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b) | Four-byte, little-endian, signed two's complement integer   
[H5T_STD_U16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga10e46ab72ac0330c51ed6cf709db4697) | Two-byte, big-endian, unsigned integer   
[H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d) | One-byte,null-terminated string of eight-bit characters   
[H5T_INTEL_B64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_x86.html#ga4eb389ac84af50af09998391169328bf) | Eight-byte bit field on an Intel CPU   
[H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc) | Reference to an object in a file   
The HDF5 library predefines a set of _NATIVE_ datatypes which are similar to C type names. The native types are set to be an alias for the appropriate HDF5 datatype for each platform. For example, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) corresponds to a C int type. On an Intel based PC, this type is the same as [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b), while on a MIPS system this would be equivalent to [H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8). Table 6 shows examples of _NATIVE_ types and corresponding C types for a common 32-bit workstation.
Table 6. Native and 32-bit C datatypes Example  | Corresponding C Type   
---|---  
[H5T_NATIVE_CHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga7439560bc4ef6d995a4e35b30262e660) | char   
[H5T_NATIVE_SCHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga273ddc372a6d574c67415917bb5232e5) | signed char   
[H5T_NATIVE_UCHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga39cadf50f1701a2af3afa86eba3c7cbd) | unsigned char   
[H5T_NATIVE_SHORT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae0625357fa121eca117f7fa9c4701810) | short   
[H5T_NATIVE_USHORT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gad0a240282e54647b83fe28ef65b40655) | unsigned short   
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) | int   
[H5T_NATIVE_UINT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga904b507c7b8aa4838fbb7c6ce71a37c5) | unsigned   
[H5T_NATIVE_LONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga290b9655882754ee0ec9f31b42ac04f6) | long   
[H5T_NATIVE_ULONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gacaea6c2c381167bacf67f6d6c43eb718) | unsigned long   
[H5T_NATIVE_LLONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gafb9c5d393d693869d7d21f71753a532c) | long long   
[H5T_NATIVE_ULLONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga91ed0d2ce3289ef3707449cf5babe928) | unsigned long long   
[H5T_NATIVE_FLOAT16](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga596ee6c1b8226b318eef1bca8ac7d20f) | _Float16   
[H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c) | float   
[H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09) | double   
[H5T_NATIVE_LDOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga6b68c680440c463ce03c5efbf8cdf1c0) | long double   
[H5T_NATIVE_FLOAT_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga957f6b409daca9447dfcf671c9b18610) | float _Complex (MSVC _Fcomplex)   
[H5T_NATIVE_DOUBLE_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga00cdc9734afbfead614191bd9736bc37) | double _Complex (MSVC _Dcomplex)   
[H5T_NATIVE_LDOUBLE_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga841b6107417a1bb8bae085333ee7a6c8) | long double _Complex (MSVC _Lcomplex)   
[H5T_NATIVE_HSIZE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3a3d25a1c3ccc59534276c2634423965) |  [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)  
[H5T_NATIVE_HSSIZE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga459210d1b4edad1f2b53c1b42ffb85e2) |  [hssize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7d9d4293176a8d7535ea6d4038235280)  
[H5T_NATIVE_HERR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga142d44a537dca7ec39da37bca48f7e40) |  [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)  
[H5T_NATIVE_HBOOL](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga8bd74ace7938e1732ad74d7332de3df2) | bool   
[H5T_NATIVE_B8](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga2ed3827a2add0f6f2252c17e231b84b1) | 8-bit unsigned integer or 8-bit buffer in memory   
[H5T_NATIVE_B16](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga81aba8febe34dc8ae2c9fa06aa4c4800) | 16-bit unsigned integer or 16-bit buffer in memory   
[H5T_NATIVE_B32](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga87ec5cf19bdbf8203ca1e9a19bc59a9f) | 32-bit unsigned integer or 32-bit buffer in memory   
[H5T_NATIVE_B64](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga4fe14f1ec3a8bd68112df6859e2db783) | 64-bit unsigned integer or 64-bit buffer in memory   
##  How Datatypes are Used
###  The Datatype Object and the HDF5 Datatype API
The HDF5 library manages datatypes as objects. The HDF5 datatype API manipulates the datatype objects through C function calls. New datatypes can be created from scratch or copied from existing datatypes. When a datatype is no longer needed its resources should be released by calling [H5Tclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0).
The datatype object is used in several roles in the HDF5 data model and library. Essentially, a datatype is used whenever the form at of data elements is needed. There are four major uses of datatypes in the HDF5 library: at dataset creation, during data transfers, when discovering the contents of a file, and for specifying user-defined datatypes. See the table below.
Table 7. Datatype uses Use  | Description   
---|---  
Dataset creation  | The datatype of the data elements must be declared when the dataset is created.   
Dataset transfer  | The datatype (format) of the data elements must be defined for both the source and destination.   
Discovery  | The datatype of a dataset can be interrogated to retrieve a complete description of the storage layout.   
Creating user-defined datatypes  | Users can define their own datatypes by creating datatype objects and setting their properties.   
###  Dataset Creation
All the data elements of a dataset have the same datatype. When a dataset is created, the datatype for the data elements must be specified. The datatype of a dataset can never be changed. The example below shows the use of a datatype to create a dataset called “/dset”. In this example, the dataset will be stored as 32-bit signed integers in big-endian order.
_Using a datatype to create a dataset_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dt;
dt = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8));
dataset_id = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file_id, “/dset”, dt, dataspace_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
#define H5Dcreate
**Definition** H5version.h:1124
[H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)
hid_t H5Tcopy(hid_t type_id)
Copies an existing datatype.
[H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8)
#define H5T_STD_I32BE
**Definition** H5Tpublic.h:466
###  Data Transfer (Read and Write)
Probably the most common use of datatypes is to write or read data from a dataset or attribute. In these operations, each data element is transferred from the source to the destination (possibly rearranging the order of the elements). Since the source and destination do not need to be identical (in other words, one is disk and the other is memory), the transfer requires both the format of the source element and the destination element. Therefore, data transfers use two datatype objects, for the source and destination.
When data is written, the source is memory and the destination is disk (file). The memory datatype describes the format of the data element in the machine memory, and the file datatype describes the desired format of the data element on disk. Similarly, when reading, the source datatype describes the format of the data element on disk, and the destination datatype describes the format in memory.
In the most common cases, the file datatype is the datatype specified when the dataset was created, and the memory datatype should be the appropriate _NATIVE_ type. The examples below show samples of writing data to and reading data from a dataset. The data in memory is declared C type ‘int’, and the datatype [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) corresponds to this type. The datatype of the dataset should be of datatype class [H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037).
_Writing to a dataset_
int dset_data[DATA_SIZE];
status = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dset_data);
[H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484)
#define H5S_ALL
**Definition** H5Spublic.h:33
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, const void *buf)
Writes raw data from a buffer to a dataset.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
_Reading from a dataset_
int dset_data[DATA_SIZE];
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), dset_data);
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)
herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void *buf)
Reads raw data from a dataset into a provided buffer.
###  Discovery of Data Format
The HDF5 Library enables a program to determine the datatype class and properties for any datatype. In order to discover the storage format of data in a dataset, the datatype is obtained, and the properties are determined by queries to the datatype object. The example below shows code that analyzes the datatype for an integer and prints out a description of its storage properties (byte order, signed, size).
_Discovering datatype properties_
switch ([H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898)(type)) {
case [H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037):
ord = [H5Tget_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaeb5bd7ec46787a4b6d33947dc73c2a5f)(type);
sgn = [H5Tget_sign](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga636f7655e706ccf7a3f23566ca561e90)(type);
printf(“Integer ByteOrder= ”);
switch (ord) {
case [H5T_ORDER_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0ae5668f73f6c28feddb7af175ac53012d):
printf(“LE”);
break;
case [H5T_ORDER_BE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0acb00548c30987f873e6836c16dbccec2):
printf(“BE”);
break;
}
printf(“ Sign= ”);
switch (sgn) {
case [H5T_SGN_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71aca59fce1697506dd3cbc7955eca77a12):
printf(“false”);
break;
case [H5T_SGN_2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71a1152d5238ff7af5c9d50edfea1ed1357):
printf(“true”);
break;
}
printf(“ Size= ”);
sz = [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(type);
printf(“%d”, sz);
printf(“\n”);
break;
case H5T_????
...
break;
}
[H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037)
@ H5T_INTEGER
**Definition** H5Tpublic.h:32
[H5T_ORDER_BE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0acb00548c30987f873e6836c16dbccec2)
@ H5T_ORDER_BE
**Definition** H5Tpublic.h:56
[H5T_ORDER_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0ae5668f73f6c28feddb7af175ac53012d)
@ H5T_ORDER_LE
**Definition** H5Tpublic.h:55
[H5T_SGN_2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71a1152d5238ff7af5c9d50edfea1ed1357)
@ H5T_SGN_2
**Definition** H5Tpublic.h:71
[H5T_SGN_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71aca59fce1697506dd3cbc7955eca77a12)
@ H5T_SGN_NONE
**Definition** H5Tpublic.h:70
[H5Tget_sign](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga636f7655e706ccf7a3f23566ca561e90)
H5T_sign_t H5Tget_sign(hid_t type_id)
Retrieves the sign type for an integer type.
[H5Tget_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaeb5bd7ec46787a4b6d33947dc73c2a5f)
H5T_order_t H5Tget_order(hid_t type_id)
Returns the byte order of an atomic datatype.
[H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)
size_t H5Tget_size(hid_t type_id)
Returns the size of a datatype.
[H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898)
H5T_class_t H5Tget_class(hid_t type_id)
Returns a datatype class.
###  Creating and Using User‐defined Datatypes
Most programs will primarily use the predefined datatypes described above, possibly in composite data types such as compound or array datatypes. However, the HDF5 datatype model is extremely general; a user program can define a great variety of atomic datatypes (storage layouts). In particular, the datatype properties can define signed and unsigned integers of any size and byte order, and floating point numbers with different formats, size, and byte order. The HDF5 datatype API provides methods to set these properties.
User-defined types can be used to define the layout of data in memory; examples might match some platform specific number format or application defined bit-field. The user-defined type can also describe data in the file such as an application-defined format. The user-defined types can be translated to and from standard types of the same class, as described above.
##  Datatype Function Summaries
see [Datatypes (H5T)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html) reference manual provides a reference list of datatype functions, the H5T APIs.
##  Programming Model for Datatypes
The HDF5 Library implements an object-oriented model of datatypes. HDF5 datatypes are organized as a logical set of base types, or datatype classes. The HDF5 Library manages datatypes as objects. The HDF5 datatype API manipulates the datatype objects through C function calls. The figure below shows the abstract view of the datatype object. The table below shows the methods (C functions) that operate on datatype objects. New datatypes can be created from scratch or copied from existing datatypes.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig5.gif) The datatype object  
---  
Table 8. General operations on datatype objects API Function  | Description   
---|---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2) class, size_t size)  | Create a new datatype object of the specified datatype class with the specified size. This function is only used with the following datatype classes: 
  * [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6)
  * [H5T_OPAQUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aaf11325a64ed5369e88d8d0d600b5cce)
  * [H5T_ENUM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5ee305303f12787367ac271d8f28f2e6)
  * [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa)
  * Other datatypes are created with a specialized datatype creation function such as [H5Tarray_create2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618) or are copied from an existing predefined datatype with [H5Tcopy()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e). 

  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Tarray_create2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) base_id, unsigned ndims, const [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) dim[]);  | Create a new array datatype object. `base_id` is the datatype of every element of the array, i.e., of the number at each position in the array. `ndims` is the number of dimensions and the size of each dimension is specified in the array `dim`.   
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Tvlen_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) base_id);  | Create a new one-dimensional variable-length array datatype object. `base_id` is the datatype of every element of the array.   
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Tenum_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gafd8d8cfead535791b3f753d21e79991f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) base_id);  | Create a new enumeration datatype object. `base_id` is the datatype of every element of the enumeration datatype.   
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Tcomplex_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_l_e_x.html#ga94d0f9813a6a69487ea71b20acafd9d4) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) base_type_id);  | Create a new complex number datatype object. `base_type_id` is the datatype of both parts of the complex number datatype and must be a floating point datatype.   
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | Obtain a modifiable transient datatype which is a copy of type. If type is a dataset identifier then the type returned is a modifiable transient copy of the datatype of the specified dataset.   
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) location, const char *name, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))  | Open a committed datatype. The committed datatype returned by this function is read-only.   
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) [H5Tequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa92250f289b557b63cba974defa20b0f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type1, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type2)  | Determines if two types are equal.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | Releases resources associated with a datatype obtained from [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e), [H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209), or [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) / [H5Tarray_create2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618) / etc. It is illegal to close an immutable transient datatype (for example, predefined types).   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tcommit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) location, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f))  | Commit a transient datatype (not immutable) to a file to become a committed datatype. Committed datatypes can be shared.   
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) [H5Tcommitted](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga0eba38d8c49784269e71ac9fa79b0f0a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | Test whether the datatype is transient or committed (named).   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tlock](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga523642dbf4c60a83127fff87664a965b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | Make a transient datatype immutable (read-only and not closable). Predefined types are locked.   
In order to use a datatype, the object must be created ([H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) / [H5Tarray_create2](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga9d9aea590106fdab7a2c07c04346f618) / etc.), or a reference obtained by cloning from an existing type ([H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)), or opened ([H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209)). In addition, a reference to the datatype of a dataset or attribute can be obtained with [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb) or [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44). For composite datatypes a reference to the datatype for members or base types can be obtained ([H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb), [H5Tget_super](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga331e8f7b388a50af77294018db788de3)). When the datatype object is no longer needed, the reference is discarded with [H5Tclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0).
Two datatype objects can be tested to see if they are the same with [H5Tequal](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa92250f289b557b63cba974defa20b0f). This function returns true if the two datatype references refer to the same datatype object. However, if two datatype objects define equivalent datatypes (the same datatype class and datatype properties), they will not be considered ‘equal’.
A datatype can be written to the file as a first class object ([H5Tcommit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd)). This is a committed datatype and can be used in the same way as any other datatype.
###  Discovery of Datatype Properties
Any HDF5 datatype object can be queried to discover all of its datatype properties. For each datatype class, there are a set of API functions to retrieve the datatype properties for this class.
#### Properties of Atomic Datatypes
Table 9 lists the functions to discover the properties of atomic datatypes. Table 10 lists the queries relevant to specific numeric types. Table 11 gives the properties for atomic string datatype, and Table 12 gives the property of the opaque datatype.
Table 9. Functions to discover properties of atomic datatypes API Function  | Description   
---|---  
[H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2) [H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | The datatype class: [H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037), [H5T_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2e92f1a42a19de186a139ab8ff0745a9), [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa), [H5T_BITFIELD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aae32f37ec15a835aa08d9277ad7ffaa2), [H5T_OPAQUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aaf11325a64ed5369e88d8d0d600b5cce), [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), [H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd), [H5T_ENUM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5ee305303f12787367ac271d8f28f2e6), [H5T_VLEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2ad8ff83b6b7ca22575263561221193028), [H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854), [H5T_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2af2675ccd96bbe2c60daf885185774120)  
size_t [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | The total size of the element in bytes, including padding which may appear on either side of the actual value.   
[H5T_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0) [H5Tget_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaeb5bd7ec46787a4b6d33947dc73c2a5f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | The byte order describes how the bytes of the datatype are laid out in memory. If the lowest memory address contains the least significant byte of the datum then it is said to be little-endian or [H5T_ORDER_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0ae5668f73f6c28feddb7af175ac53012d). If the bytes are in the opposite order then they are said to be big-endianor [H5T_ORDER_BE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0acb00548c30987f873e6836c16dbccec2).   
size_t [H5Tget_precision](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaac9f5410c8cf456f048011030b7f90f9) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | The precision property identifies the number of significant bits of a datatype and the offset property (defined below) identifies its location. Some datatypes occupy more bytes than what is needed to store the value. For instance, a short on a Cray is 32 significant bits in an eight-byte field.   
int [H5Tget_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga225f0b6d173f90d3696bb68b88ae07c1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | The offset property defines the bit location of the least significant bit of a bit field whose length is precision.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tget_pad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga26a22811b56f5a63b6cb638f6773d872) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, [H5T_pad_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aa) *lsb, [H5T_pad_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aa) *msb)  | Padding is the bits of a data element which are not significant as defined by the precision and offset properties. Padding in the low-numbered bits is lsb padding and padding in the high-numbered bits is msb padding. Padding bits can be set to zero ([H5T_PAD_ZERO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaaed1384c65a60f4d623fe6bc852b72823)) or one ([H5T_PAD_ONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaa2020ae42345fc8236811593c59ac4fe8)).   
Table 10. Functions to discover properties of atomic datatypes API Function  | Description   
---|---  
[H5T_sign_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71) [H5Tget_sign](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga636f7655e706ccf7a3f23566ca561e90) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | (INTEGER)Integer data can be signed two's complement ([H5T_SGN_2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71a1152d5238ff7af5c9d50edfea1ed1357)) or unsigned ([H5T_SGN_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71aca59fce1697506dd3cbc7955eca77a12)).   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tget_fields](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga42e62cb497fdec8f08cb9ac3c6de0e14) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, size_t *spos, size_t *epos, size_t *esize, size_t*mpos, size_t *msize)  | (FLOAT)A floating-point data element has bit fields which are the exponent and mantissa as well as a mantissa sign bit. These properties define the location (bit position of least significant bit of the field) and size (in bits) of each field. The sign bit is always of length one and none of the fields are allowed to overlap.   
size_t [H5Tget_ebias](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga302b1c22cc6007ca69724a9e387e3888) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | (FLOAT)A floating-point data element has bit fields which are the exponent and mantissa as well as a mantissa sign bit. These properties define the location (bit position of least significant bit of the field) and size (in bits) of each field. The sign bit is always of length one and none of the fields are allowed to overlap.   
[H5T_norm_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a609b101af0343a4a76d8c3e182cdda80) [H5Tget_norm](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gad43c79f15565465a3559f5faf2c87b75) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | (FLOAT)This property describes the normalization method of the mantissa. 
  * [H5T_NORM_MSBSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a609b101af0343a4a76d8c3e182cdda80a9dc5e01d17abf41c619e154150de8dde): the mantissa is shifted left (if non-zero) until the first bit after the radix point is set and the exponent is adjusted accordingly. All bits of the mantissa after the radix point are stored. 
  * [H5T_NORM_IMPLIED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a609b101af0343a4a76d8c3e182cdda80a5b649062dea480101917cc2d6b58f65d): the mantissa is shifted left (if non-zero) until the first bit after the radix point is set and the exponent is adjusted accordingly. The first bit after the radix point is not stored since it's always set. 
  * [H5T_NORM_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a609b101af0343a4a76d8c3e182cdda80ab1a219215c45144cf317f2ea846a861c): the fractional part of the mantissa is stored without normalizing it.

  
[H5T_pad_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aa) [H5Tget_inpad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaaea02cfeb3e749d0983563b4d510a321) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | (FLOAT)If any internal bits (that is, bits between the sign bit, the mantissa field, and the exponent field but within the precision field) are unused, then they will be filled according to the value of this property. The padding can be: [H5T_PAD_BACKGROUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaac2ca8836f78fc3e7f524098857c42e64), [H5T_PAD_ZERO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaaed1384c65a60f4d623fe6bc852b72823),or [H5T_PAD_ONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaa2020ae42345fc8236811593c59ac4fe8).   
Table 11. Functions to discover properties of atomic string datatypes API Function  | Description   
---|---  
[H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) [H5Tget_cset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga5bc2f3e8f708f5bcdd0d8667950310c1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | Two character sets are currently supported: ASCII ([H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663)) and UTF-8 ([H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)).   
[H5T_str_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514) [H5Tget_strpad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga564b21cc269467c39f59462feb0d5903) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | The string datatype has a fixed length, but the string may be shorter than the length. This property defines the storage mechanism for the left over bytes. The options are: 
  * [H5T_STR_NULLTERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a23c685afc240bbac4da23b36d8fd7e13)
  * [H5T_STR_NULLPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a128d51156e51b7a2c9db0fe8787b4547)
  * [H5T_STR_SPACEPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a3f73f8dae99444798f5efd7d2d2a5e5c). 

  
Table 12. Functions to discover properties of atomic opaque datatypes API Function  | Description   
---|---  
char* [H5Tget_tag](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#gaf1d0f634ac1a3b4220b8fe0197b93832)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id)  | A user-defined string.   
#### Properties of Composite Datatypes
The composite datatype classes can also be analyzed to discover their datatype properties and the datatypes that are members or base types of the composite datatype. The member or base type can, in turn, be analyzed. The table below lists the functions that can access the datatype properties of the different composite datatypes.
Table 13. Functions to discover properties of composite datatypes API Function  | Description   
---|---  
int [H5Tget_nmembers](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id)  | (COMPOUND)The number of fields in the compound datatype.   
[H5T_class_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2) [H5Tget_member_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gac8476d164fb972fbf7b8c4584b8e916b) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) cdtype_id, unsigned member_no)  | (COMPOUND)The datatype class of compound datatype member member_no.   
char* [H5Tget_member_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, unsigned field_idx)  | (COMPOUND)The name of field field_idx of a compound datatype.   
size_t [H5Tget_member_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, unsigned memb_no)  | (COMPOUND)The byte offset of the beginning of a field within a compound datatype.   
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, unsigned field_idx)  | (COMPOUND)The datatype of the specified member.   
int [H5Tget_array_ndims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#gadec89de23da8efaba4677abfd818a9c0) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) adtype_id)  | (ARRAY)The number of dimensions (rank) of the array datatype object.   
int [H5Tget_array_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) adtype_id, [hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) *dims[])  | (ARRAY)The sizes of the dimensions and the dimension permutations of the array datatype object.   
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Tget_super](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga331e8f7b388a50af77294018db788de3)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type)  | (ARRAY, VL, ENUM)The base datatype from which the datatype type is derived.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tenum_nameof](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gab45f148151595716006ad8b603d55623)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, const void *value, char *name, size_t size)  | (ENUM)The symbol name that corresponds to the specified value of the enumeration datatype.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tenum_valueof](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga5a50f4172640de713e16f0ecd12aeb30)([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, const char *name, void *value)  | (ENUM)The value that corresponds to the specified name of the enumeration datatype.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tget_member_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga4fc2d29dcde5af45b905bbc7355d2b76) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type unsigned memb_no, void *value)  | (ENUM)The value of the enumeration datatype member memb_no.   
###  Definition of Datatypes
The HDF5 library enables user programs to create and modify datatypes. The essential steps are: 
  * 1. Create a new datatype object of a specific composite datatype class, or copy an existing atomic datatype object 
  * 2. Set properties of the datatype object 
  * 3. Use the datatype object 
  * 4. Close the datatype object


To create a user-defined atomic datatype, the procedure is to clone a predefined datatype of the appropriate datatype class ([H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)), and then set the datatype properties appropriate to the datatype class. The table below shows how to create a datatype to describe a 1024-bit unsigned integer.
_Create a new datatype_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_type = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e) ([H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
[H5Tset_precision](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab0f4dccfc2fb47bf2c7e06c9bf84c1f7)(new_type, 1024);
[H5Tset_sign](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga3ac9a4781cd3c4a8b5df4ff549ec8aec)(new_type, [H5T_SGN_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71aca59fce1697506dd3cbc7955eca77a12));
[H5Tset_sign](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga3ac9a4781cd3c4a8b5df4ff549ec8aec)
herr_t H5Tset_sign(hid_t type_id, H5T_sign_t sign)
Sets the sign property for an integer type.
[H5Tset_precision](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab0f4dccfc2fb47bf2c7e06c9bf84c1f7)
herr_t H5Tset_precision(hid_t type_id, size_t prec)
Sets the precision of an atomic datatype.
Composite datatypes are created with a specific API call for each datatype class. The table below shows the creation method for each datatype class. A newly created datatype cannot be used until the datatype properties are set. For example, a newly created compound datatype has no members and cannot be used.
Table 14. Functions to create each datatype class Datatype Class  | Function to Create   
---|---  
COMPOUND  |  [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65 "Creates a new datatype.")  
OPAQUE  |  [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65 "Creates a new datatype.")  
ENUM  |  [H5Tenum_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#gafd8d8cfead535791b3f753d21e79991f "Creates a new enumeration datatype.")  
ARRAY  |  [H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96)  
VL  |  [H5Tvlen_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f "Creates a new variable-length array datatype.")  
COMPLEX  |  [H5Tcomplex_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_l_e_x.html#ga94d0f9813a6a69487ea71b20acafd9d4 "Creates a new complex number datatype.")  
Once the datatype is created and the datatype properties set, the datatype object can be used.
Predefined datatypes are defined by the library during initialization using the same mechanisms as described here. Each predefined datatype is locked ([H5Tlock](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga523642dbf4c60a83127fff87664a965b)), so that it cannot be changed or destroyed. User-defined datatypes may also be locked using [H5Tlock](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga523642dbf4c60a83127fff87664a965b).
#### User-defined Atomic Datatypes
Table 15 summarizes the API methods that set properties of atomic types. Table 16 shows properties specific to numeric types, Table 17 shows properties specific to the string datatype class. Note that offset, pad, etc. do not apply to strings. Table 18 shows the specific property of the OPAQUE datatype class.
Table 15. API methods that set properties of atomic datatypes Functions  | Description   
---|---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, size_t size)  | Set the total size of the element in bytes. This includes padding which may appear on either side of the actual value. If this property is reset to a smaller value which would cause the significant part of the data to extend beyond the edge of the datatype, then the offset property is decremented a bit at a time. If the offset reaches zero and the significant part of the data still extends beyond the edge of the datatype then the precision property is decremented a bit at a time. Decreasing the size of a datatype may fail if the [H5T_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2e92f1a42a19de186a139ab8ff0745a9) bit fields would extend beyond the significant part of the type.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab1aab76b1214a819281f2156c6d45d71) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, [H5T_order_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0) order)  | Set the byte order to little-endian ([H5T_ORDER_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0ae5668f73f6c28feddb7af175ac53012d)) or big-endian ([H5T_ORDER_BE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0acb00548c30987f873e6836c16dbccec2)).   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_precision](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab0f4dccfc2fb47bf2c7e06c9bf84c1f7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, size_t precision)  | Set the number of significant bits of a datatype. The offset property (defined below) identifies its location. The size property defined above represents the entire size (in bytes) of the datatype. If the precision is decreased then padding bits are inserted on the MSB side of the significant bits (this will fail for [H5T_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2e92f1a42a19de186a139ab8ff0745a9) types if it results in the sign,mantissa, or exponent bit field extending beyond the edge of the significant bit field). On the other hand, if the precision is increased so that it “hangs over” the edge of the total size then the offset property is decremented a bit at a time. If the offset reaches zero and the significant bits still hang over the edge, then the total size is increased a byte at a time.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gafd22e4b0aecbe6dad9a899c5bf567e2f) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, size_t offset)  | Set the bit location of the least significant bit of a bit field whose length is precision. The bits of the entire data are numbered beginning at zero at the least significant bit of the least significant byte (the byte at the lowest memory address for a little-endian type or the byte at the highest address for a big-endian type). The offset property defines the bit location of the least significant bit of a bit field whose length is precision. If the offset is increased so the significant bits “hang over” the edge of the datum, then the size property is automatically incremented.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_pad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga1089a9f454052d0038a06a432ce8e1e1) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, [H5T_pad_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aa) lsb, [H5T_pad_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aa) msb)  | Set the padding to zeros ([H5T_PAD_ZERO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaaed1384c65a60f4d623fe6bc852b72823)) or ones ([H5T_PAD_ONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaa2020ae42345fc8236811593c59ac4fe8)). Padding is the bits of a data element which are not significant as defined by the precision and offset properties. Padding in the low-numbered bits is lsb padding and padding in the high-numbered bits is msb padding.   
Table 16. API methods that set properties of numeric datatypes Functions  | Description   
---|---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_sign](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga3ac9a4781cd3c4a8b5df4ff549ec8aec) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, [H5T_sign_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71) sign)  | (INTEGER)Integer data can be signed two's complement ([H5T_SGN_2](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71a1152d5238ff7af5c9d50edfea1ed1357)) or unsigned ([H5T_SGN_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af7bfee2db210a12b9290eba85d730a71aca59fce1697506dd3cbc7955eca77a12)).   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_fields](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gafbdc98b45749e5cfbaf1a8689f3c403d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, size_t spos, size_t epos, size_t esize, size_t mpos, size_t msize)  | (FLOAT)Set the properties define the location (bit position of least significant bit of the field) and size (in bits) of each field. The sign bit is always of length one and none of the fields are allowed to overlap.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_ebias](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gad2c4a8f09672f4166f39efe83d44dba2) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, size_t ebias)  | (FLOAT)The exponent is stored as a non-negative value which is ebias larger than the true exponent.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_norm](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaa415a17c98bf32c357f5a35ba657beab) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, [H5T_norm_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a609b101af0343a4a76d8c3e182cdda80) norm)  | (FLOAT)This property describes the normalization method of the mantissa. 
  * [H5T_NORM_MSBSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a609b101af0343a4a76d8c3e182cdda80a9dc5e01d17abf41c619e154150de8dde): the mantissa is shifted left (if non-zero) until the first bit after theradix point is set and the exponent is adjusted accordingly. All bits of the mantissa after the radix point are stored. 
  * [H5T_NORM_IMPLIED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a609b101af0343a4a76d8c3e182cdda80a5b649062dea480101917cc2d6b58f65d): the mantissa is shifted left (if non-zero) until the first bit after the radix point is set and the exponent is adjusted accordingly. The first bit after the radix point is not stored since it is always set. 
  * [H5T_NORM_NONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a609b101af0343a4a76d8c3e182cdda80ab1a219215c45144cf317f2ea846a861c): the fractional part of the mantissa is stored without normalizing it.

  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_inpad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga6dc8e6ba49a24f56f0912539cf9e0481) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, [H5T_pad_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aa) inpad)  | (FLOAT) If any internal bits (that is, bits between the sign bit, the mantissa field, and the exponent field but within the precision field) are unused, then they will be filled according to the value of this property. The padding can be: 
  * [H5T_PAD_BACKGROUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaac2ca8836f78fc3e7f524098857c42e64)
  * [H5T_PAD_ZERO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaaed1384c65a60f4d623fe6bc852b72823)
  * [H5T_PAD_ONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaa2020ae42345fc8236811593c59ac4fe8)

  
Table 17. API methods that set properties of string datatypes Functions  | Description   
---|---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, size_t size)  | Set the length of the string, in bytes. The precision is automatically set to 8*size.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_precision](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab0f4dccfc2fb47bf2c7e06c9bf84c1f7) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, size_t precision)  | The precision must be a multiple of 8.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_cset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga4909c0c3d97c3d212fee032cc8dc031a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [H5T_cset_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71a) cset)  | Two character sets are currently supported: 
  * ASCII ([H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663)) 
  * UTF-8 ([H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)). 

  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_strpad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gaec9ebf44e766cc5b932d0bf26dcf8700) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, [H5T_str_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514) strpad)  | The string datatype has a fixed length, but the string may be shorter than the length. This property defines the storage mechanism for the left over bytes. The method used to store character strings differs with the programming language: 
  * C usually null terminates strings 
  * Fortran left-justifies and space-pads strings

Valid string padding values, as passed in the parameter strpad, are as follows: 
  * [H5T_STR_NULLTERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a23c685afc240bbac4da23b36d8fd7e13): Null terminate (as C does) 
  * [H5T_STR_NULLPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a128d51156e51b7a2c9db0fe8787b4547): Pad with zeros 
  * [H5T_STR_SPACEPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a3f73f8dae99444798f5efd7d2d2a5e5c): Pad with spaces (as FORTRAN does) 

  
Table 18. API methods that set properties of opaque datatypes Functions  | Description   
---|---  
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tset_tag](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#ga3543ad909983a2a20e651d16502de43d) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type_id, const char *tag)  | Tags the opaque datatype type_id with an ASCII identifier tag.   
#### Examples
The example below shows how to create a 128-bit little-endian signed integer type. Increasing the precision of a type automatically increases the total size. Note that the proper procedure is to begin from a type of the intended datatype class which in this case is a NATIVE INT.
_Create a new 128-bit little-endian signed integer datatype_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_type = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e) ([H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
[H5Tset_precision](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab0f4dccfc2fb47bf2c7e06c9bf84c1f7) (new_type, 128);
[H5Tset_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab1aab76b1214a819281f2156c6d45d71) (new_type, [H5T_ORDER_LE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a2a6a8eb856a0829fecaac60f803c9fd0ae5668f73f6c28feddb7af175ac53012d));
[H5Tset_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab1aab76b1214a819281f2156c6d45d71)
herr_t H5Tset_order(hid_t type_id, H5T_order_t order)
Sets the byte order of a datatype.
The figure below shows the storage layout as the type is defined. The [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e) creates a datatype that is the same as [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2). In this example, suppose this is a 32-bit big-endian number (Figure a). The precision is set to 128 bits, which automatically extends the size to 8 bytes (Figure b). Finally, the byte order is set to little-endian (Figure c).
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig6.gif) The storage layout for a new 128-bit little-endian signed integer datatype  
---  
The significant bits of a data element can be offset from the beginning of the memory for that element by an amount of padding. The offset property specifies the number of bits of padding that appear to the “right of” the value. The table and figure below show how a 32-bit unsigned integer with 16-bits of precision having the value 0x1122 will be laid out in memory.
Table 19. Memory Layout for a 32-bit unsigned integer Byte Position  | Big-Endian  
Offset=0  | Big-Endian  
Offset=16  | Little-Endian  
Offset=0  | Little-Endian  
Offset=16   
---|---|---|---|---  
0:  | [pad]  | [0x11]  | [0x22]  | [pad]   
1:  | [pad]  | [0x22]  | [0x11]  | [pad]   
2:  | [0x11]  | [pad]  | [pad]  | [0x22]   
3:  | [0x22]  | [pad]  | [pad]  | [0x11]   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig7.gif) Memory Layout for a 32-bit unsigned integer  
---  
If the offset is incremented then the total size is incremented also if necessary to prevent significant bits of the value from hanging over the edge of the datatype.
The bits of the entire data are numbered beginning at zero at the least significant bit of the least significant byte (the byte at the lowest memory address for a little-endian type or the byte at the highest address for a big-endian type). The offset property defines the bit location of the least significant bit of a bit field whose length is precision. If the offset is increased so the significant bits “hang over” the edge of the datum, then the size property is automatically incremented.
To illustrate the properties of the integer datatype class, the example below shows how to create a user-defined datatype that describes a 24-bit signed integer that starts on the third bit of a 32-bit word. The datatype is specialized from a 32-bit integer, the precision is set to 24 bits, and the offset is set to 3.
_A user-defined datatype with a 24-bit signed integer_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dt;
dt = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)(H5T_SDT_I32LE);
[H5Tset_precision](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab0f4dccfc2fb47bf2c7e06c9bf84c1f7)(dt, 24);
[H5Tset_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gafd22e4b0aecbe6dad9a899c5bf567e2f)(dt,3);
[H5Tset_pad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga1089a9f454052d0038a06a432ce8e1e1)(dt, [H5T_PAD_ZERO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaaed1384c65a60f4d623fe6bc852b72823), [H5T_PAD_ONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaa2020ae42345fc8236811593c59ac4fe8));
[H5T_PAD_ONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaa2020ae42345fc8236811593c59ac4fe8)
@ H5T_PAD_ONE
**Definition** H5Tpublic.h:147
[H5T_PAD_ZERO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaaed1384c65a60f4d623fe6bc852b72823)
@ H5T_PAD_ZERO
**Definition** H5Tpublic.h:146
[H5Tset_pad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga1089a9f454052d0038a06a432ce8e1e1)
herr_t H5Tset_pad(hid_t type_id, H5T_pad_t lsb, H5T_pad_t msb)
Sets the least and most-significant bits padding types.
[H5Tset_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gafd22e4b0aecbe6dad9a899c5bf567e2f)
herr_t H5Tset_offset(hid_t type_id, size_t offset)
Sets the bit offset of the first significant bit.
The figure below shows the storage layout for a data element. Note that the unused bits in the offset will be set to zero and the unused bits at the end will be set to one, as specified in the [H5Tset_pad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga1089a9f454052d0038a06a432ce8e1e1) call. 
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig8.gif) A user-defined integer datatype with a range of -1,048,583 to 1,048,584  
---  
To illustrate a user-defined floating point number, the example below shows how to create a 24-bit floating point number that starts 5 bits into a 4 byte word. The floating point number is defined to have a mantissa of 19 bits (bits 5-23), an exponent of 3 bits (25-27), and the sign bit is bit 28. (Note that this is an illustration of what can be done and is not necessarily a floating point format that a user would require.)
_A user-defined datatype with a 24-bit floating point datatype_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dt;
dt = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)(H5T_SDT_F32LE);
[H5Tset_precision](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gab0f4dccfc2fb47bf2c7e06c9bf84c1f7)(dt, 24);
[H5Tset_fields](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gafbdc98b45749e5cfbaf1a8689f3c403d) (dt, 28, 25, 3, 5, 19);
[H5Tset_pad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga1089a9f454052d0038a06a432ce8e1e1)(dt, [H5T_PAD_ZERO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaaed1384c65a60f4d623fe6bc852b72823), [H5T_PAD_ONE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaa2020ae42345fc8236811593c59ac4fe8));
[H5Tset_inpad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga6dc8e6ba49a24f56f0912539cf9e0481)(dt, [H5T_PAD_ZERO](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a361ad902a75bcf442c17bf3d0bc103aaaed1384c65a60f4d623fe6bc852b72823));
[H5Tset_inpad](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga6dc8e6ba49a24f56f0912539cf9e0481)
herr_t H5Tset_inpad(hid_t type_id, H5T_pad_t pad)
Fills unused internal floating-point bits.
[H5Tset_fields](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#gafbdc98b45749e5cfbaf1a8689f3c403d)
herr_t H5Tset_fields(hid_t type_id, size_t spos, size_t epos, size_t esize, size_t mpos, size_t msize)
Sets locations and sizes of floating point bit fields.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig9.gif) A user-defined floating point datatype  
---  
The figure above shows the storage layout of a data element for this datatype. Note that there is an unused bit (24) between the mantissa and the exponent. This bit is filled with the inpad value which in this case is 0.
The sign bit is always of length one and none of the fields are allowed to overlap. When expanding a floating-point type one should set the precision first; when decreasing the size one should set the field positions and sizes first.
#### Composite Datatypes
All composite datatypes must be user-defined; there are no predefined composite datatypes.
#### Compound Datatypes
The subsections below describe how to create a compound datatype and how to write and read data of a compound datatype.
#### Defining Compound Datatypes
Compound datatypes are conceptually similar to a C struct or Fortran derived types. The compound datatype defines a contiguous sequence of bytes, which are formatted using one up to 2^16 datatypes (members). A compound datatype may have any number of members in any order, and the members may have any datatype, including compound. Thus, complex nested compound datatypes can be created. The total size of the compound datatype is greater than or equal to the sum of the size of its members, up to a maximum of 2^32 bytes. HDF5 does not support datatypes with distinguished records or the equivalent of C unions or Fortran EQUIVALENCE statements.
Typically, a C struct or Fortran derived type is defined to store a data point in memory. The offsets of the members in memory represent their positions relative to the beginning of an instance of the struct. The HDF5 C library includes a macro, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) (s, m), which calculates the offset of member _m_ within struct _s_. Alternatively, the `offsetof(s, m)` macro, defined in _stddef.h_ , serves the same purpose as the `HOFFSET` macro. For Fortran users, the HDF5 library provides the function [h5offsetof](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_h5.html#ga01e3a81f299b9c9cb4bc088bab795412) to determine the offset of a member. To find the size of a scalar derived type, the Fortran function equivalent of the _sizeof_ can be used. Note, in the past, the HDF5 Fortran applications had to calculate offsets by using sizes of members datatypes and by considering the order of members in the Fortran derived type, thus offsets of Fortran structure members corresponded to the offsets within a packed datatype (see explanation below) stored in an HDF5 file.
Each member of a compound datatype must have a descriptive name which is the key used to uniquely identify the member within the compound datatype. A member name in an HDF5 datatype does not necessarily have to be the same as the name of the member in the C struct or Fortran derived type, although this is often the case. Nor does one need to define all members of the C struct or Fortran derived type in the HDF5 compound datatype (or vice versa).
Unlike atomic datatypes which are derived from other atomic datatypes, compound datatypes are created from scratch. First, one creates an empty compound datatype and specifies its total size. Then members are added to the compound datatype in any order. Each member type is inserted at a designated offset. Each member has a name which is the key used to uniquely identify the member within the compound datatype.
The example below shows a way of creating an HDF5 C compound datatype to describe a complex number. This is a structure with two components, “real” and “imaginary”, and each component is a double. An equivalent C struct whose type is defined by the complex_tstruct is shown.
_A compound datatype for complex numbers in C_
typedef struct {
double re; //real part
double im; //imaginary part
} complex_t;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) complex_id = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof (complex_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (complex_id, “real”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(complex_t,re),
[H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (complex_id, “imaginary”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(complex_t,im),
[H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6)
@ H5T_COMPOUND
**Definition** H5Tpublic.h:38
[HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)
#define HOFFSET(S, M)
**Definition** H5Tpublic.h:22
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)
herr_t H5Tinsert(hid_t parent_id, const char *name, size_t offset, hid_t member_id)
Adds a new member to a compound datatype.
[H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65)
hid_t H5Tcreate(H5T_class_t type, size_t size)
Creates a new datatype.
[H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09)
#define H5T_NATIVE_DOUBLE
**Definition** H5Tpublic.h:1036
The example below shows a way of creating an HDF5 Fortran compound datatype to describe a complex number. This is a Fortran derived type with two components, “real” and “imaginary”, and each component is DOUBLE PRECISION. An equivalent Fortran TYPE whose type is defined by the TYPE complex_t is shown.
_A compound datatype for complex numbers in Fortran_
**Fortran 2003****Fortran (Obsolete)**
  * TYPE complex_t
DOUBLE PRECISION re ! real part
DOUBLE PRECISION im ! imaginary part
END TYPE complex_t
TYPE(complex_t), DIMENSION(1:8), TARGET :: cmplx
CalcSize = H5OFFSETOF(C_LOC(cmplx(1)), C_LOC(cmplx(2))
CALL h5tcreate_f(H5T_COMPOUND_F, CalcSize, type_id, error)
offset = H5OFFSETOF(C_LOC(cmplx),C_LOC(cmplx%re))
CALL h5tinsert_f(type_id, “real”, offset, [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), error)
offset = H5OFFSETOF(C_LOC(cmplx),C_LOC(cmplx%im))
CALL h5tinsert_f(type_id, “imaginary”, offset, [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), error)
  * TYPE complex_t
DOUBLE PRECISION re ! real part
DOUBLE PRECISION im ! imaginary part
END TYPE complex_t
CALL h5tget_size_f([H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), re_size, error)
CALL h5tget_size_f([H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), im_size, error)
complex_t_size = re_size + im_size
CALL h5tcreate_f(H5T_COMPOUND_F, complex_t_size, type_id, error)
offset = 0
CALL h5tinsert_f(type_id, “real”, offset, [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), error)
offset = offset + re_size
CALL h5tinsert_f(type_id, “imaginary”, offset, [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), error)


Important Note: The compound datatype is created with a size sufficient to hold all its members. In the C example above, the size of the C struct and the [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro are used as a convenient mechanism to determine the appropriate size and offset. Alternatively, the size and offset could be manually determined: the size can be set to 16 with “real” at offset 0 and “imaginary” at offset 8. However, different platforms and compilers have different sizes for “double” and may have alignment restrictions which require additional padding within the structure. It is much more portable to use the [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro which assures that the values will be correct for any platform.
The figure below shows how the compound datatype would be laid out assuming that NATIVE_DOUBLE are 64-bit numbers and that there are no alignment requirements. The total size of the compound datatype will be 16 bytes, the “real” component will start at byte 0, and “imaginary” will start at byte 8.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig10.gif) Layout of a compound datatype  
---  
The members of a compound datatype may be any HDF5 datatype including the compound, array, and variable-length (VL) types. The figure and example below show the memory layout and code which creates a compound datatype composed of two complex values, and each complex value is also a compound datatype as in the figure above.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig11.gif) Layout of a compound datatype nested in a compound datatype  
---  
_TText for a compound datatype nested in a compound datatype_
typedef struct {
complex_t x;
complex_t y;
} surf_t;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) complex_id, surf_id; // hdf5 datatypes
complex_id = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(complex_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (complex_id, “re”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(complex_t, re), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (complex_id, “im”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(complex_t, im), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
surf_id = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(surf_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (surf_id, “x”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(surf_t, x), complex_id);
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (surf_id, “y”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(surf_t, y), complex_id);
Note that a similar result could be accomplished by creating a compound datatype and inserting four fields. See the figure below. This results in the same layout as the figure above. The difference would be how the fields are addressed. In the first case, the real part of ‘y’ is called ‘y.re’; in the second case it is ‘y-re’.
_Another compound datatype nested in a compound datatype_
typedef struct {
complex_t x;
complex_t y;
} surf_t;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) surf_id = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(surf_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (surf_id, “x-re”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(surf_t, x.re), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (surf_id, “x-im”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(surf_t, x.im), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (surf_id, “y-re”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(surf_t, y.re), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (surf_id, “y-im”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(surf_t, y.im), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
The members of a compound datatype do not always fill all the bytes. The [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) macro assures that the members will be laid out according to the requirements of the platform and language. The example below shows an example of a C struct which requires extra bytes of padding on many platforms. The second element, ‘b’, is a 1-byte character followed by an 8 byte double, ‘c’. On many systems, the 8-byte value must be stored on a 4-or 8-byte boundary. This requires the struct to be larger than the sum of the size of its elements.
In the example below, sizeof and [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac) are used to assure that the members are inserted at the correct offset to match the memory conventions of the platform. The figure below shows how this data element would be stored in memory, assuming the double must start on a 4-byte boundary. Notice the extra bytes between ‘b’ and ‘c’.
_A compound datatype that requires padding_
typedef struct {
int a;
char b;
double c;
} s1_t;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) s1_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(s1_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (s1_tid, “x-im”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, a), [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (s1_tid, “y-re”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, b), [H5T_NATIVE_CHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga7439560bc4ef6d995a4e35b30262e660));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081) (s1_tid, “y-im”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, c), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
[H5T_NATIVE_CHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga7439560bc4ef6d995a4e35b30262e660)
#define H5T_NATIVE_CHAR
**Definition** H5Tpublic.h:958
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig12.gif) Memory layout of a compound datatype that requires padding  
---  
However, data stored on disk does not require alignment, so unaligned versions of compound data structures can be created to improve space efficiency on disk. These unaligned compound datatypes can be created by computing offsets by hand to eliminate inter-member padding, or the members can be packed by calling [H5Tpack](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga1a28ac30f83a4920aba49bb1b0a6a0f3 "Recursively removes padding from within a compound datatype.") (which modifies a datatype directly, so it is usually preceded by a call to [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e "Copies an existing datatype.")).
The example below shows how to create a disk version of the compound datatype from the figure above in order to store data on disk in as compact a form as possible. Packed compound datatypes should generally not be used to describe memory as they may violate alignment constraints for the architecture being used. Note also that using a packed datatype for disk storage may involve a higher data conversion cost.
_Create a packed compound datatype in C_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) s2_tid = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e) (s1_tid);
[H5Tpack](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga1a28ac30f83a4920aba49bb1b0a6a0f3) (s2_tid);
[H5Tpack](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga1a28ac30f83a4920aba49bb1b0a6a0f3)
herr_t H5Tpack(hid_t type_id)
Recursively removes padding from within a compound datatype.
The example below illustrates the sequence of Fortran calls used to create a packed compound datatype. Before Fortran 2003, an HDF5 Fortran compound datatype did not represent a compound datatype in memory. Therefore, compound data was ALWAYS written by field, as explained in the next section. Packing was only necessary if the offset of each consecutive member was not equal to the sum of the sizes of the previous members. However, with the introduction of Fortran 2003, this is no longer the case, and the same considerations that apply to C also apply to Fortran.
_Create a packed compound datatype in Fortran_
CALL h5tcopy_f(s1_id, s2_id, error)
CALL h5tpack_f(s2_id, error)
#### Creating and Writing Datasets with Compound Datatypes
Creating datasets with compound datatypes is similar to creating datasets with any other HDF5 datatypes. But writing and reading may be different since datasets that have compound datatypes can be written or read by a field (member) or subsets of fields (members). The compound datatype is the only composite datatype that supports “sub-setting” by the elements the datatype is built from.
The example below shows a C example of creating and writing a dataset with a compound datatype.
_Create and write a dataset with a compound datatype in C_
typedef struct s1_t {
int a;
float b;
double c;
} s1_t;
s1_t data[LENGTH];
// Initialize data
for (i = 0; i < LENGTH; i++) {
data[i].a = i;
data[i].b = i*i;
data[i].c = 1./(i+1);
}
...
s1_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(s1_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “a_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, a), [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “b_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, b), [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “c_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, c), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
...
dataset_id = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file_id, “SDScompound.h5”, s1_t,
space_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) (dataset_id, s1_tid, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data);
[H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c)
#define H5T_NATIVE_FLOAT
**Definition** H5Tpublic.h:1030
The example below shows the content of the file written on a little-endian machine.
_Create and write a little-endian dataset with a compound datatype in C_
HDF5 “SDScompound.h5” {
GROUP “/” {
DATASET “ArrayOfStructures” {
DATATYPE [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6) {
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b) “a_name”;
[H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea) “b_name”;
[H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e) “c_name”;
}
DATASPACE SIMPLE { ( 3 ) / ( 3 ) }
DATA {
(0): {
0,
0,
1
},
(1): {
0,
1,
0.5
},
(2): {
0,
4,
0.333333
}
}
}
}
}
[H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e)
#define H5T_IEEE_F64LE
**Definition** H5Tpublic.h:283
[H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea)
#define H5T_IEEE_F32LE
**Definition** H5Tpublic.h:271
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b)
#define H5T_STD_I32LE
**Definition** H5Tpublic.h:471
It is not necessary to write the whole data at once. Datasets with compound datatypes can be written by field or by subsets of fields. In order to do this one has to remember to set the transfer property of the dataset using the H5Pset_preserve call and to define the memory datatype that corresponds to a field. The example below shows how float and double fields are written to the dataset.
_Writing floats and doubles to a dataset_
typedef struct sb_t {
float b;
double c;
} sb_t;
typedef struct sc_t {
float b;
double c;
} sc_t;
sb_t data1[LENGTH];
sc_t data2[LENGTH];
// Initialize data
for (i = 0; i < LENGTH; i++) {
data1.b = i * i;
data2.c = 1./(i + 1);
}
...
// Create dataset as in example 15
...
// Create memory datatypes corresponding to float
// and double datatype fields
sb_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(sb_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(sb_tid, “b_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(sb_t, b), [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c));
sc_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(sc_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(sc_tid, “c_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(sc_t, c), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
...
// Set transfer property
xfer_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d));
[H5Pset_preserve](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga55445d27d058dadcd34fdba93344855e)(xfer_id, 1);
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) (dataset_id, sb_tid, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), xfer_id, data1);
[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906) (dataset_id, sc_tid, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), xfer_id, data2);
[H5P_DATASET_XFER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a6f9c8a5aba72c0445fff384bf418a80d)
#define H5P_DATASET_XFER
**Definition** H5Ppublic.h:68
[H5Pset_preserve](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga55445d27d058dadcd34fdba93344855e)
herr_t H5Pset_preserve(hid_t plist_id, bool status)
Sets the dataset transfer property list status.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
The figure below shows the content of the file written on a little-endian machine. Only float and double fields are written. The default fill value is used to initialize the unwritten integer field.
_Writing floats and doubles to a dataset on a little-endian system_
HDF5 “SDScompound.h5” {
GROUP “/” {
DATASET “ArrayOfStructures” {
DATATYPE [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6) {
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b) “a_name”;
[H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea) “b_name”;
[H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e) “c_name”;
}
DATASPACE SIMPLE { ( 3 ) / ( 3 ) }
DATA {
(0): {
0,
0,
1
},
(1): {
0,
1,
0.5
},
(2): {
0,
4,
0.333333
}
}
}
}
}
_Create and write a dataset with a compound datatype in Fortran_
**Fortran 2003****Fortran (Obsolete)**
  * The following example demonstrates how to create and write a dataset using a compound datatype in Fortran 2003. 
TYPE s1_t
INTEGER :: a
REAL :: b
DOUBLE PRECISION :: c
END TYPE
TYPE(s1_t), TARGET, DIMENSION(1:LENGTH) :: data
! Initialize data
DO i = 1, LENGTH
data[i].a = i-1
data[i].b = (i-1)*(i-1)
data[i].c = 1./(i)
}
...
type_size = H5OFFSETOF(C_LOC(data(1)), C_LOC(data(2)))
CALL H5Tcreate_f(H5T_COMPOUND_F, type_size, s1_tid, error)
offset = H5OFFSETOF(C_LOC(data(1)), C_LOC(data(1)%a))
CALL H5Tinsert_f(s1_tid, “a_name”, offset, H5T_NATIVE_INTEGER, error)
offset = H5OFFSETOF(C_LOC(data(1)), C_LOC(data(1)%b))
CALL H5Tinsert_f(s1_tid, “b_name”, offset, H5T_NATIVE_REAL, error)
offset = H5OFFSETOF(C_LOC(data(1)), C_LOC(data(1)%c))
CALL H5Tinsert_f(s1_tid, “c_name”, offset, [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), error)
...
CALL H5Dcreate_f(file_id, “SDScompound.h5”, s1_t, space_id, dataset_id, error)
CALL H5Dwrite_f(dataset_id, s1_tid, C_LOC(data(1)), error)
  * The following example demonstrates creating and writing a dataset with a compound datatype using pre-Fortran 2003 standards. As illustrated in Fortran 90, writing and reading compound datatypes is always done by fields. The content of the written file matches the example provided previously. 
! One cannot write an array of a derived datatype in
! Fortran 90.
TYPE s1_t
INTEGER a
REAL b
DOUBLE PRECISION c
END TYPE s1_t
TYPE(s1_t) d(LENGTH)
! Therefore, the following code initializes an array
! corresponding to each field in the derived datatype
! and writesthose arrays to the dataset
INTEGER, DIMENSION(LENGTH) :: a
REAL, DIMENSION(LENGTH) :: b
DOUBLE PRECISION, DIMENSION(LENGTH) :: c
! Initialize data
do i = 1, LENGTH
a(i) = i-1
b(i) = (i-1) * (i-1)
c(i) = 1./i
enddo
...
! Set dataset transfer property to preserve partially
! initialized fields during write/read to/from dataset
! with compound datatype.
!
CALL h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error)
CALL h5pset_preserve_f(plist_id, .TRUE., error)
...
!
! Create compound datatype.
!
! First calculate total size by calculating sizes of
! each member
!
CALL h5tget_size_f(H5T_NATIVE_INTEGER, type_sizei, error)
CALL h5tget_size_f(H5T_NATIVE_REAL, type_sizer, error)
CALL h5tget_size_f([H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), type_sized, error)
type_size = type_sizei + type_sizer + type_sized
CALL h5tcreate_f(H5T_COMPOUND_F, type_size, dtype_id, error)
!
! Insert members
!
!
! INTEGER member
!
offset = 0
CALL h5tinsert_f(dtype_id, “a_name”, offset, H5T_NATIVE_INTEGER, error)
!
! REAL member
!
offset = offset + type_sizei
CALL h5tinsert_f(dtype_id, “b_name”, offset, H5T_NATIVE_REAL, error)
!
! DOUBLE PRECISION member
!
offset = offset + type_sizer
CALL h5tinsert_f(dtype_id, “c_name”, offset, [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), error)
!
! Create the dataset with compound datatype.
!
CALL h5dcreate_f(file_id, dsetname, dtype_id, dspace_id, &dset_id, error, H5P_DEFAULT_F,
H5P_DEFAULT_F, H5P_DEFAULT_F)
!
...
! Create memory types. We have to create a compound
! datatype for each member we want to write.
!
CALL h5tcreate_f(H5T_COMPOUND_F, type_sizei, dt1_id, error)
offset = 0
CALL h5tinsert_f(dt1_id, “a_name”, offset, H5T_NATIVE_INTEGER, error)
!
CALL h5tcreate_f(H5T_COMPOUND_F, type_sizer, dt2_id, error)
offset = 0
CALL h5tinsert_f(dt2_id, “b_name”, offset, H5T_NATIVE_REAL, error)
!
CALL h5tcreate_f(H5T_COMPOUND_F, type_sized, dt3_id, error)
offset = 0
CALL h5tinsert_f(dt3_id, “c_name”, offset, [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09), error)
!
! Write data by fields in the datatype. Fields order
! is not important.
!
CALL h5dwrite_f(dset_id, dt3_id, c, data_dims, error, xfer_prp = plist_id)
CALL h5dwrite_f(dset_id, dt2_id, b, data_dims, error, xfer_prp = plist_id)
CALL h5dwrite_f(dset_id, dt1_id, a, data_dims, error, xfer_prp = plist_id)


#### Reading Datasets with Compound Datatypes
Reading datasets with compound datatypes may be a challenge. For general applications there is no way to know a priori the corresponding C structure. Also, C structures cannot be allocated on the fly during discovery of the dataset's datatype. For general C, C++, Fortran and Java application the following steps will be required to read and to interpret data from the dataset with compound datatype: 
  * 1. Get the identifier of the compound datatype in the file with the [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb "Returns an identifier for a copy of the datatype for a dataset.") call 
  * 2. Find the number of the compound datatype members with the [H5Tget_nmembers](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626 "Retrieves the number of elements in a compound or enumeration datatype.") call 
  * 3. Iterate through compound datatype members 
    * Get member class with the [H5Tget_member_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gac8476d164fb972fbf7b8c4584b8e916b "Returns datatype class of compound datatype member.") call 
    * Get member name with the [H5Tget_member_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66 "Retrieves the name of a compound or enumeration datatype member.") call 
    * Check class type against predefined classes 
      * [H5T_INTEGER](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aba1fc36abc23f073912e337d2291b037)
      * [H5T_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2e92f1a42a19de186a139ab8ff0745a9)
      * [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa)
      * [H5T_BITFIELD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aae32f37ec15a835aa08d9277ad7ffaa2)
      * [H5T_OPAQUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aaf11325a64ed5369e88d8d0d600b5cce)
      * [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6)
      * [H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd)
      * [H5T_ENUM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5ee305303f12787367ac271d8f28f2e6)
      * [H5T_VLEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2ad8ff83b6b7ca22575263561221193028)
      * [H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854)
      * [H5T_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2af2675ccd96bbe2c60daf885185774120)
    * If class is [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), then go to step 2 and repeat all steps under step 3. If class is not [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), then a member is of an atomic class and can be read to a corresponding buffer after discovering all necessary information specific to each atomic type (for example, size of the integer or floats, super class for enumerated and array datatype, and its sizes)


The examples below show how to read a dataset with a known compound datatype.
The first example below shows the steps needed to read data of a known structure. First, build a memory datatype the same way it was built when the dataset was created, and then second use the datatype in an [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") call.
_Read a dataset using a memory datatype_
typedef struct s1_t {
int a;
float b;
double c;
} s1_t;
s1_t *data;
...
s1_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65)([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(s1_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “a_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, a), [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “b_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, b), [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “c_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, c), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
...
dataset_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file_id, “SDScompound.h5”, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
...
data = (s1_t *) malloc (sizeof(s1_t)*LENGTH);
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset_id, s1_tid, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data);
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
Instead of building a memory datatype, the application could use the [H5Tget_native_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga05b99133058637e8daa5d745381ddd3d "Returns the native datatype identifier of a specified datatype.") function. See the example below.
_Read a dataset using H5Tget_native_type_
typedef struct s1_t {
int a;
float b;
double c;
} s1_t;
s1_t *data;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file_s1_t, mem_s1_t;
...
dataset_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file_id, “SDScompound.h5”, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Discover datatype in the file
file_s1_t = [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)(dataset_id);
// Find corresponding memory datatype
mem_s1_t = [H5Tget_native_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga05b99133058637e8daa5d745381ddd3d)(file_s1_t, [H5T_DIR_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a891787104495ea80c677ab7042bfc33da15b446e6a306e3824cf0003108d6b51c));
...
data = (s1_t *) malloc (sizeof(s1_t)*LENGTH);
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c) (dataset_id,mem_s1_tid, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data);
[H5T_DIR_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a891787104495ea80c677ab7042bfc33da15b446e6a306e3824cf0003108d6b51c)
@ H5T_DIR_DEFAULT
**Definition** H5Tpublic.h:159
[H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)
hid_t H5Dget_type(hid_t dset_id)
Returns an identifier for a copy of the datatype for a dataset.
[H5Tget_native_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga05b99133058637e8daa5d745381ddd3d)
hid_t H5Tget_native_type(hid_t type_id, H5T_direction_t direction)
Returns the native datatype identifier of a specified datatype.
The example below shows how to read just one float member of a compound datatype.
_Read one floating point member of a compound datatype_
typedef struct sf_t {
float b;
} sf_t;
sf_t *data;
...
sf_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65)([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(sf_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(sf_tid, “b_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(sf_t, b), [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c));
...
dataset_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file_id, “SDScompound.h5”, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
...
data = (sf_t *) malloc (sizeof(sf_t) * LENGTH);
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset_id, sf_tid, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data);
The example below shows how to read float and double members of a compound datatype into a structure that has those fields in a different order. Please notice that [H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081 "Adds a new member to a compound datatype.") calls can be used in an order different from the order of the structure's members.
_Read float and double members of a compound datatype_
typedef struct sdf_t {
double c;
float b;
} sdf_t;
sdf_t *data;
...
sdf_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65)([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(sdf_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(sdf_tid, “b_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(sdf_t, b), [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(sdf_tid, “c_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(sdf_t, c), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
...
dataset_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file_id, “SDScompound.h5”, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
...
data = (sdf_t *) malloc (sizeof(sdf_t) * LENGTH);
[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset_id, sdf_tid, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), data);
#### Array
Many scientific datasets have multiple measurements for each point in a space. There are several natural ways to represent this data, depending on the variables and how they are used in computation. See the table and the figure below.
Representing data with multiple measurements Storage Strategy |  Stored as |  Remarks  
---|---|---  
Multiple planes  | Several datasets with identical dataspaces  | This is optimal when variables are accessed individually, or when often uses only selected variables.   
Additional dimension  | One dataset, the last “dimension” is a vec-tor of variables  | This can give good performance, although selecting only a few variables may be slow. This may not reflect the science.   
Record with multiple values  | One dataset with compound datatype  | This enables the variables to be read all together or selected. Also handles “vectors” of heterogeneous data.   
Vector or Tensor value  | One dataset, each data element is a small array of values.  | This uses the same amount of space as the previous two, and may represent the science model better.   
Figure 13 Representing data with multiple measurements ![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig13a.gif) |  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig13b.gif)  
---|---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig13c.gif) |  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig13d.gif)  
The HDF5 [H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854) datatype defines the data element to be a homogeneous, multi-dimensional array. See Figure 13 above. The elements of the array can be any HDF5 datatype (including compound and array), and the size of the datatype is the total size of the array. A dataset of array datatype cannot be subdivided for I/O within the data element: the entire array of the data element must be transferred. If the data elements need to be accessed separately, for example, by plane, then the array datatype should not be used. The table below shows advantages and disadvantages of various storage methods.
Storage method advantages and disadvantages Method |  Advantages |  Disadvantages  
---|---|---  
Multiple Datasets  | Easy to access each plane, can select any plane(s)  | Less efficient to access a 'column' through the planes   
N+1 Dimension  | All access patterns supported  | Must be homogeneous datatype  
The added dimension may not make sense in the scientific model   
Compound Datatype  | Can be heterogeneous datatype  | Planes must be named, selection is by plane  
Not a natural representation for a matrix   
Array  | A natural representation for vector or tensor data  | Cannot access elements separately (no access by plane)   
An array datatype may be multi-dimensional with 1 to [H5S_MAX_RANK](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a265cb2343f05cb71831119c90de31a8f) (the maximum rank of a dataset is currently 32) dimensions. The dimensions can be any size greater than 0, but unlimited dimensions are not supported (although the datatype can be a variable-length datatype).
An array datatype is created with the [H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96) call, which specifies the number of dimensions, the size of each dimension, and the base type of the array. The array datatype can then be used in any way that any datatype object is used. The example below shows the creation of a datatype that is a two-dimensional array of native integers, and this is then used to create a dataset. Note that the dataset can be a dataspace that is any number and size of dimensions. The figure below shows the layout in memory assuming that the native integers are 4 bytes. Each data element has 6 elements, for a total of 24 bytes.
_Create a two-dimensional array datatype_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file, dataset;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) datatype, dataspace;
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c) adims[] = {3, 2};
datatype = [H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96)([H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), 2, adims, NULL);
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, datasetname, datatype,
dataspace, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hsize_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a7f81cce70fb546af88da24d9285d3c1c)
uint64_t hsize_t
**Definition** H5public.h:304
[H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96)
#define H5Tarray_create
**Definition** H5version.h:1504
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig14.gif) Memory layout of a two-dimensional array datatype  
---  
#### Variable-length Datatypes
A variable-length (VL) datatype is a one-dimensional sequence of a datatype which are not fixed in length from one dataset location to another. In other words, each data element may have a different number of members. Variable-length datatypes cannot be divided; the entire data element must be transferred.
VL datatypes are useful to the scientific community in many different ways, possibly including: 
  * Ragged arrays: Multi-dimensional ragged arrays can be implemented with the last (fastest changing) dimension being ragged by using a VL datatype as the type of the element stored. 
  * Fractal arrays: A nested VL datatype can be used to implement ragged arrays of ragged arrays, to whatever nesting depth is required for the user. 
  * Polygon lists: A common storage requirement is to efficiently store arrays of polygons with different numbers of vertices. A VL datatype can be used to efficiently and succinctly describe an array of polygons with different numbers of vertices. 
  * Character strings: Perhaps the most common use of VL datatypes will be to store C-like VL character strings in dataset elements or as attributes of objects. 
  * Indices (for example, of objects within a file): An array of VL object references could be used as an index to all the objects in a file which contain a particular sequence of dataset values. 
  * Object Tracking: An array of VL dataset region references can be used as a method of tracking objects or features appearing in a sequence of datasets. 


A VL datatype is created by calling [H5Tvlen_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f "Creates a new variable-length array datatype.") which specifies the base datatype. The first example below shows an example of code that creates a VL datatype of unsigned integers. Each data element is a one-dimensional array of zero or more members and is stored in the [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) structure. See the second example below.
_Create a variable-length datatype of unsigned integers_
tid1 = [H5Tvlen_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f) ([H5T_NATIVE_UINT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga904b507c7b8aa4838fbb7c6ce71a37c5));
dataset=[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(fid1,“Dataset1”, tid1, sid1, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5T_NATIVE_UINT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga904b507c7b8aa4838fbb7c6ce71a37c5)
#define H5T_NATIVE_UINT
**Definition** H5Tpublic.h:994
[H5Tvlen_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6841355fa5b3c924876b121dedb8ed2f)
hid_t H5Tvlen_create(hid_t base_id)
Creates a new variable-length array datatype.
_Data element storage for members of the VL datatype_
typedef struct
{
size_t len; // Length of VL data
//(in base type units)
void *p; // Pointer to VL data
} [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html);
[hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html)
**Definition** H5Tpublic.h:198
The first example below shows how the VL data is written. For each of the 10 data elements, a length and data buffer must be allocated. Below the two examples is a figure that shows how the data is laid out in memory.
An analogous procedure must be used to read the data. See the second example below. An appropriate array of [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) must be allocated, and the data read. It is then traversed one data element at a time. The [H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe "Reclaims the variable length \(VL\) datatype memory buffers.") call frees the data buffer for the buffer. With each element possibly being of different sequence lengths for a dataset with a VL datatype, the memory for the VL datatype must be dynamically allocated. Currently there are two methods of managing the memory for VL datatypes: the standard C malloc/free memory allocation routines or a method of calling user-defined memory management routines to allocate or free memory (set with [H5Pset_vlen_mem_manager](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga2220ab75de470b6a6d5b1173d12aa0cf "Sets the memory manager for variable-length datatype allocation in H5Dread\(\) and H5Dvlen_reclaim\(\)")). Since the memory allocated when reading (or writing) may be complicated to release, the [H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe "Reclaims the variable length \(VL\) datatype memory buffers.") function is provided to traverse a memory buffer and free the VL datatype information without leaking memory.
_Write VL data_
[hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) wdata[10]; // Information to write
// Allocate and initialize VL data to write
for(i = 0; i < 10; i++) {
wdata[i].[p](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html#a117104b82864d3b23ec174af6d392709) = malloc((i + 1) * sizeof(unsigned int));
wdata[i].[len](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html#a7360b55975153b822efc5217b7734e6a) = i + 1;
for(j = 0; j < (i + 1); j++)
((unsigned int *)wdata[i].p)[j]=i * 10 + j;
}
ret = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset, tid1, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), wdata);
[hvl_t::p](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html#a117104b82864d3b23ec174af6d392709)
void * p
**Definition** H5Tpublic.h:200
[hvl_t::len](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html#a7360b55975153b822efc5217b7734e6a)
size_t len
**Definition** H5Tpublic.h:199
_Read VL data_
[hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) rdata[SPACE1_DIM1];
ret = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, tid1, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), xfer_pid, rdata);
for(i = 0; i < SPACE1_DIM1; i++) {
printf(“%d: len %d ”,rdata[i].len);
for(j = 0; j < rdata[i].[len](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html#a7360b55975153b822efc5217b7734e6a); j++) {
printf(“ value: %u\n”,((unsigned int *)rdata[i].p)[j]);
}
}
ret = [H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe)(tid1, sid1, xfer_pid, rdata);
[H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe)
herr_t H5Treclaim(hid_t type_id, hid_t space_id, hid_t plist_id, void *buf)
Reclaims the variable length (VL) datatype memory buffers.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig15.gif) Memory layout of a VL datatype  
---  
The user program must carefully manage these relatively complex data structures. The [H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe "Reclaims the variable length \(VL\) datatype memory buffers.") function performs a standard traversal, freeing all the data. This function analyzes the datatype and dataspace objects, and visits each VL data element, recursing through nested types. By default, the system free is called for the pointer in each [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html). Obviously, this call assumes that all of this memory was allocated with the system malloc.
The user program may specify custom memory manager routines, one for allocating and one for freeing. These may be set with the [H5Pset_vlen_mem_manager](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga2220ab75de470b6a6d5b1173d12aa0cf "Sets the memory manager for variable-length datatype allocation in H5Dread\(\) and H5Dvlen_reclaim\(\)"), and must have the following prototypes: 
  * typedef void *(*H5MM_allocate_t)(size_t size, void *info);
  * typedef void (*[H5MM_free_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m_mpublic_8h.html#aa34c7616be59673cfc3d63fa7d960f25))(void *mem, void *free_info);
[H5MM_free_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_m_mpublic_8h.html#aa34c7616be59673cfc3d63fa7d960f25)
void(* H5MM_free_t)(void *mem, void *free_info)
**Definition** H5MMpublic.h:33


The utility function [H5Dvlen_get_buf_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0e97bbd8a8ee4a8b5b78ccce8547ce76 "Determines the number of bytes required to store variable-length \(VL\) data.") checks the number of bytes required to store the VL data from the dataset. This function analyzes the datatype and dataspace object to visit all the VL data elements, to determine the number of bytes required to store the data for the in the destination storage (memory). The size value is adjusted for data conversion and alignment in the destination.
#### Complex
A complex number datatype represents complex number data elements which consist of two floating point parts. Complex number datatypes cannot be divided for I/O; the entire data element must be transferred.
A complex number datatype is created by calling [H5Tcomplex_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_l_e_x.html#ga94d0f9813a6a69487ea71b20acafd9d4 "Creates a new complex number datatype.") with a specified base floating point datatype. The example below shows code that creates a complex number datatype of 16-bit floating point values.
_Create a complex number datatype of 2 IEEE little-endian 16-bit floating point values_
tid1 = [H5Tcomplex_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_l_e_x.html#ga94d0f9813a6a69487ea71b20acafd9d4) ([H5T_IEEE_F16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#gac1735e1dcc978313a6605b62f935282a));
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(fid1, “Dataset1”, tid1, sid1, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Tcomplex_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_l_e_x.html#ga94d0f9813a6a69487ea71b20acafd9d4)
hid_t H5Tcomplex_create(hid_t base_type_id)
Creates a new complex number datatype.
[H5T_IEEE_F16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#gac1735e1dcc978313a6605b62f935282a)
#define H5T_IEEE_F16LE
**Definition** H5Tpublic.h:259
_Data element storage of a complex number datatype_
Each part of a data element with a complex number datatype is stored contiguously. Complex number datatypes have the same storage representation as an array datatype of 2 elements of a floating point datatype or a compound datatype with 2 fields and no structure padding, where each field is of the same floating point datatype. Thus, the following representations are equivalent:
float _Complex data;
float data[2];
struct
{
float real;
float imaginary;
} data;
##  Other Non-numeric Datatypes
Several datatype classes define special types of objects.
###  Strings
Text data is represented by arrays of characters, called strings. Many programming languages support different conventions for storing strings, which may be fixed or variable-length, and may have different rules for padding unused storage. HDF5 can represent strings in several ways. See the figure below.
The strings to store are “Four score” and “lazy programmers.” 
A string stored as one-character elements in a one-dimensional array a) [H5T_NATIVE_CHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga7439560bc4ef6d995a4e35b30262e660): The dataset is a one-dimensional array with 29 elements, and each element is a single character.   
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig16a.gif)  
b) Fixed-length string: The dataset is a one-dimensional array with two elements, and each element is 20 characters.   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig16b.gif)  
c) Variable-length string: The dataset is a one-dimensional array with two elements, and each element is a variable-length string. This is the same result when stored as a fixed-length string except that the first element of the array will need only 11 bytes for storage instead of 20.   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig16c.gif)  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig16d.gif)  
First, a dataset may have a dataset with datatype [H5T_NATIVE_CHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga7439560bc4ef6d995a4e35b30262e660) with each character of the string as an element of the dataset. This will store an unstructured block of text data, but gives little indication of any structure in the text. See item a in the figure above.
A second alternative is to store the data using the datatype class [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) with each element a fixed length. See item b in the figure above. In this approach, each element might be a word or a sentence, addressed by the dataspace. The dataset reserves space for the specified number of characters, although some strings may be shorter. This approach is simple and usually is fast to access, but can waste storage space if the length of the Strings varies.
A third alternative is to use a variable-length datatype. See item c in the figure above. This can be done using the standard mechanisms described above. The program would use [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) structures to write and read the data.
A fourth alternative is to use a special feature of the string datatype class to set the size of the datatype to [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc). See item c in the figure above. The example below shows a declaration of a datatype of type [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d) which is set to [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc). The HDF5 Library automatically translates between this and the [hvl_t](https://support.hdfgroup.org/documentation/hdf5/latest/structhvl__t.html) structure. Note: the [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc) size can only be used with string datatypes.
_Set the string datatype size to H5T_VARIABLE_
tid1 = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e) ([H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d));
ret = [H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c) (tid1, [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc));
[H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc)
#define H5T_VARIABLE
**Definition** H5Tpublic.h:209
[H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)
herr_t H5Tset_size(hid_t type_id, size_t size)
Sets size for a datatype.
[H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d)
#define H5T_C_S1
**Definition** H5Tpublic.h:646
Variable-length strings can be read into C strings (in other words, pointers to zero terminated arrays of char). See the example below.
_Read variable-length strings into C strings_
char *rdata[SPACE1_DIM1];
ret = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, tid1, [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), xfer_pid, rdata);
for(i = 0; i < SPACE1_DIM1; i++) {
printf(“%d: len: %d, str is: %s\n”, i, strlen(rdata[i]), rdata[i]);
}
ret = [H5Treclaim](https://support.hdfgroup.org/documentation/hdf5/latest/group___v_l_e_n.html#ga6851783a68a0f868c27300cb5622fbbe)(tid1, sid1, xfer_pid, rdata);
###  Reference
In HDF5, objects (groups, datasets, attributes, and committed datatypes) are usually accessed by name. There is another way to access stored objects - by reference. Before HDF5 1.12.0, there were only two reference datatypes: object reference and region reference. Since 1.12.0, attribute references and external references were added. And all references can be stored and retrieved from a file by invoking the [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906 "Writes raw data from a buffer to a dataset.") and [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") functions with a single predefined type: [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc).
The first example below shows an example of code that creates references to four objects, and then writes the array of object references to a dataset. The second example below shows a dataset of datatype reference being read and one of the reference objects being dereferenced to obtain an object pointer.
_Create object references and write to a dataset_
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (fid1, “Dataset3”, [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc), sid1, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Create reference to dataset
ret = [H5Rcreate_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad0fb6ec99ecefa602756a5682addfc69)(fid1,“/Group1/Dataset1”, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), &wbuf[0]);
// Create reference to dataset
ret = [H5Rcreate_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad0fb6ec99ecefa602756a5682addfc69)(fid1, “/Group1/Dataset2”, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), &wbuf[1]);
// Create reference to group
ret = [H5Rcreate_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad0fb6ec99ecefa602756a5682addfc69)(fid1, “/Group1”, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), &wbuf[2]);
// Create reference to committed datatype
ret = [H5Rcreate_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad0fb6ec99ecefa602756a5682addfc69)(fid1, “/Group1/Datatype1”, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), &wbuf[3]);
// Write selection to disk
ret = [H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset, [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), wbuf);
// Release buffers
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&wdata[0]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&wdata[1]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&wdata[2]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&wdata[3]);
[H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2)
#define H5R_OBJECT
**Definition** H5Rpublic.h:624
[H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)
herr_t H5Rdestroy(H5R_ref_t *ref_ptr)
Closes a reference.
[H5Rcreate_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gad0fb6ec99ecefa602756a5682addfc69)
herr_t H5Rcreate_object(hid_t loc_id, const char *name, hid_t oapl_id, H5R_ref_t *ref_ptr)
Creates an object reference.
[H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc)
#define H5T_STD_REF
**Definition** H5Tpublic.h:576
_Read a dataset with a reference datatype_
rbuf = ([H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html) *)malloc(dims[0] * sizeof([H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html)));
// Read selection from disk
ret = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), rbuf);
// Open dataset object
dset2 = [H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd)(&rbuf[0], [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Release buffers
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&rbuf[0]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&rbuf[1]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&rbuf[2]);
status = [H5Rdestroy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga026114e6c23588bf89bb473eb9e4d095)(&rbuf[3]);
[H5Ropen_object](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#gaa6692bd3a5b490c8db05b90a5888d0dd)
hid_t H5Ropen_object(H5R_ref_t *ref_ptr, hid_t rapl_id, hid_t oapl_id)
Opens the HDF5 object referenced.
[H5R_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_r__ref__t.html)
**Definition** H5Rpublic.h:97
###  Deprecated Reference
In HDF5, objects (groups, datasets, and committed datatypes) are usually accessed by name. There is another way to access stored objects - by reference. There are two deprecated reference datatypes: object reference and region reference. Object reference objects are created with [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d "Creates a reference.") and other calls (cross reference). These objects can be stored and retrieved in a dataset as elements with reference datatype. The first example below shows an example of code that creates references to four objects, and then writes the array of object references to a dataset. The second example below shows a dataset of datatype reference being read and one of the reference objects being dereferenced to obtain an object pointer.
In order to store references to regions of a dataset, the datatype should be [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e). Note that a data element must be either an object reference or a region reference: these are different types and cannot be mixed within a single array.
A reference datatype cannot be divided for I/O: an element is read or written completely.
_Create object references and write to a dataset_
dataset= [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (fid1, “Dataset3”, [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1), sid1, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// Create reference to dataset
ret = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&wbuf[0], fid1,“/Group1/Dataset1”, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), -1);
// Create reference to dataset
ret = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&wbuf[1], fid1, “/Group1/Dataset2”, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), -1);
// Create reference to group
ret = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&wbuf[2], fid1, “/Group1”, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), -1);
// Create reference to committed datatype
ret = [H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)(&wbuf[3], fid1, “/Group1/Datatype1”, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), -1);
// Write selection to disk
ret=[H5Dwrite](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga98f44998b67587662af8b0d8a0a75906)(dataset, [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), wbuf);
[H5Rcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga0ac6997b5de26b11d91a95de2869950d)
herr_t H5Rcreate(void *ref, hid_t loc_id, const char *name, H5R_type_t ref_type, hid_t space_id)
Creates a reference.
[H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1)
#define H5T_STD_REF_OBJ
**Definition** H5Tpublic.h:566
_Read a dataset with a reference datatype_
rbuf = malloc(sizeof([hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081))*SPACE1_DIM1);
// Read selection from disk
ret=[H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset, [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5S_ALL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_spublic_8h.html#a5f96eeee84b987f18470737f85af0484), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), rbuf);
// Open dataset object
dset2 = [H5Rdereference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a)(dataset, [H5R_OBJECT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#ae1ffa09875ca6778df3a577592dacbd2), &rbuf[0]);
[hobj_ref_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_rpublic_8h.html#af1f69150fc8b3d6bf182da39dc0fe081)
haddr_t hobj_ref_t
**Definition** H5Rpublic.h:71
[H5Rdereference](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html#ga5dc19b9d1833af66c5e1f819f2c05c4a)
#define H5Rdereference
**Definition** H5version.h:1471
###  ENUM
The enum datatype implements a set of (name, value) pairs, similar to C/C++ enum. The values are currently limited to native integer datatypes. Each name can be the name of only one value, and each value can have only one name.
The data elements of the ENUMERATION are stored according to the datatype. An example would be as an array of integers. The example below shows an example of how to create an enumeration with five elements. The elements map symbolic names to 2-byte integers. See the table below.
_Create an enumeration with five elements_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) hdf_en_colors;
short val;
hdf_en_colors = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65)([H5T_ENUM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5ee305303f12787367ac271d8f28f2e6), sizeof(short));
[H5Tenum_insert](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f)(hdf_en_colors, “RED”, (val=0, &val));
[H5Tenum_insert](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f)(hdf_en_colors, “GREEN”, (val=1, &val));
[H5Tenum_insert](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f)(hdf_en_colors, “BLUE”, (val=2, &val));
[H5Tenum_insert](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f)(hdf_en_colors, “WHITE”, (val=3, &val));
[H5Tenum_insert](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f)(hdf_en_colors, “BLACK”, (val=4, &val));
[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(fileid, datasetname, hdf_en_colors, spaceid, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5T_ENUM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5ee305303f12787367ac271d8f28f2e6)
@ H5T_ENUM
**Definition** H5Tpublic.h:40
[H5Tenum_insert](https://support.hdfgroup.org/documentation/hdf5/latest/group___e_n_u_m.html#ga7bbddcff3a5d18ee983fbe5654fdc41f)
herr_t H5Tenum_insert(hid_t type, const char *name, const void *value)
Inserts a new enumeration datatype member.
An enumeration with five elements Name  | Value   
---|---  
RED  | 0   
GREEN  | 1   
BLUE  | 2   
WHITE  | 3   
BLACK  | 4   
The figure below shows how an array of eight values might be stored. Conceptually, the array is an array of symbolic names [BLACK, RED, WHITE, BLUE, ...] See item a in the figure below. These are stored as the values and are short integers. So, the first 2 bytes are the value associated with “BLACK”, which is the number 4, and so on. See item b in the figure below. 
Storing an enum array a) Logical data to be written - eight elements   
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig17a.gif)  
b) The storage layout. Total size of the array is 16 bytes, 2 bytes per element.   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig17b.gif)  
The order that members are inserted into an enumeration type is unimportant; the important part is the associations between the symbol names and the values. Thus, two enumeration datatypes will be considered equal if and only if both types have the same symbol/value associations and both have equal underlying integer datatypes. Type equality is tested with the H5Tequal function.
If a particular architecture type is required, a little-endian or big-endian datatype for example, use a native integer datatype as the ENUM base datatype and use [H5Tconvert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_n_v.html#ga9442478475a03357ee47fa035df0228a "Converts data from one specified datatype to another.") on values as they are read from or written to a dataset.
###  Opaque
In some cases, a user may have data objects that should be stored and retrieved as blobs with no attempt to interpret them. For example, an application might wish to store an array of encrypted certificates which are 100 bytes long.
While an arbitrary block of data may always be stored as bytes, characters, integers, or whatever, this might mislead programs about the meaning of the data. The opaque datatype defines data elements which are uninterpreted by HDF5. The opaque data may be labeled with [H5Tset_tag](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_p_a_q_u_e.html#ga3543ad909983a2a20e651d16502de43d "Tags an opaque datatype.") with a string that might be used by an application. For example, the encrypted certificates might have a tag to indicate the encryption and the certificate standard.
###  Bitfield
Some data is represented as bits, where the number of bits is not an integral byte and the bits are not necessarily interpreted as a standard type. Some examples might include readings from machine registers (for example, switch positions), a cloud mask, or data structures with several small integers that should be store in a single byte.
This data could be stored as integers, strings, or enumerations. However, these storage methods would likely result in considerable wasted space. For example, storing a cloud mask with one byte per value would use up to eight times the space of a packed array of bits.
The HDF5 bitfield datatype class defines a data element that is a contiguous sequence of bits, which are stored on disk in a packed array. The programming model is the same as for unsigned integers: the datatype object is created by copying a predefined datatype, and then the precision, offset, and padding are set.
While the use of the bitfield datatype will reduce storage space substantially, there will still be wasted space if the bitfield as a whole does not match the 1-, 2-, 4-, or 8-byte unit in which it is written. The remaining unused space can be removed by applying the N-bit filter to the dataset containing the bitfield data. For more information, see "Using the N-bit Filter."
##  Fill Values
The “fill value” for a dataset is the specification of the default value assigned to data elements that have not yet been written. In the case of a dataset with an atomic datatype, the fill value is a single value of the appropriate datatype, such as ‘0’ or ‘-1.0’. In the case of a dataset with a composite datatype, the fill value is a single data element of the appropriate type. For example, for an array or compound datatype, the fill value is a single data element with values for all the component elements of the array or compound datatype.
The fill value is set (permanently) when the dataset is created. The fill value is set in the dataset creation properties in the [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) call. Note that the [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) call must also include the datatype of the dataset, and the value provided for the fill value will be interpreted as a single element of this datatype. The example below shows code which creates a dataset of integers with fill value -1. Any unwritten data elements will be set to -1.
_Create a dataset with a fill value of -1_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist_id;
int filler;
filler = -1;
plist_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
[H5Pset_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga4335bb45b35386daa837b4ff1b9cd4a4)(plist_id, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), &filler);
// Create the dataset with fill value ‘-1’.
dataset_id = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file_id, “/dset”, [H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8), dataspace_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), plist_id,
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102)
#define H5P_DATASET_CREATE
**Definition** H5Ppublic.h:60
[H5Pset_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga4335bb45b35386daa837b4ff1b9cd4a4)
herr_t H5Pset_fill_value(hid_t plist_id, hid_t type_id, const void *value)
Sets the fill value for a dataset.
_Create a fill value for a compound datatype_
typedef struct s1_t {
int a;
char b;
double c;
} s1_t;
s1_t filler;
s1_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(s1_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “a_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, a), [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “b_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, b), [H5T_NATIVE_CHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga7439560bc4ef6d995a4e35b30262e660));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “c_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, c), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
filler.a = -1;
filler.b = ‘*’;
filler.c = -2.0;
plist_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_DATASET_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afcd7f8186c404f3a1d768632eacba102));
[H5Pset_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga4335bb45b35386daa837b4ff1b9cd4a4)(plist_id, s1_tid, &filler);
// Create the dataset with fill value
// (-1, ‘*’, -2.0).
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, datasetname, s1_tid, space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), plist_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
The code above shows how to create a fill value for a compound datatype. The procedure is the same as the previous example except the filler must be a structure with the correct fields. Each field is initialized to the desired fill value.
The fill value for a dataset can be retrieved by reading the dataset creation properties of the dataset and then by reading the fill value with [H5Pget_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga82bbe8c77c7eb9c460bfd1eb26881355 "Retrieves a dataset fill value."). The data will be read into memory using the storage layout specified by the datatype. This transfer will convert data in the same way as [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer."). The example below shows how to get the fill value from the dataset created in the example "Create a dataset with a fill value of -1".
_Retrieve a fill value_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) plist2;
int filler;
dataset_id = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file_id, “/dset”, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
plist2 = [H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163)(dataset_id);
[H5Pget_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga82bbe8c77c7eb9c460bfd1eb26881355)(plist2, [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), &filler);
// filler has the fill value, ‘-1’
[H5Pget_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga82bbe8c77c7eb9c460bfd1eb26881355)
herr_t H5Pget_fill_value(hid_t plist_id, hid_t type_id, void *value)
Retrieves a dataset fill value.
[H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163)
hid_t H5Dget_create_plist(hid_t dset_id)
Returns an identifier for a copy of the dataset creation property list for a dataset.
A similar procedure is followed for any datatype. The example below shows how to read the fill value for the compound datatype created in an example above. Note that the program must pass an element large enough to hold a fill value of the datatype indicated by the argument to [H5Pget_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga82bbe8c77c7eb9c460bfd1eb26881355 "Retrieves a dataset fill value."). Also, the program must understand the datatype in order to interpret its components. This may be difficult to determine without knowledge of the application that created the dataset.
_Read the fill value for a compound datatype_
char *fillbuf;
int sz;
dataset = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)( file, DATASETNAME, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
s1_tid = [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)(dataset);
sz = [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s1_tid);
fillbuf = (char *)malloc(sz);
plist_id = [H5Dget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8848f14f4aba8e6160c3d8bb7f1be163)(dataset);
[H5Pget_fill_value](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga82bbe8c77c7eb9c460bfd1eb26881355)(plist_id, s1_tid, fillbuf);
printf(“filler.a: %d\n”,((s1_t *) fillbuf)->a);
printf(“filler.b: %c\n”,((s1_t *) fillbuf)->b);
printf(“filler.c: %f\n”,((s1_t *) fillbuf)->c);
##  Complex Combinations of Datatypes
Several composite datatype classes define collections of other datatypes, including other composite datatypes. In general, a datatype can be nested to any depth, with any combination of datatypes.
For example, a compound datatype can have members that are other compound datatypes, arrays, VL datatypes. An array can be an array of array, an array of compound, or an array of VL. And a VL datatype can be a variable-length array of compound, array, or VL datatypes.
These complicated combinations of datatypes form a logical tree, with a single root datatype, and leaves which must be atomic datatypes (predefined or user-defined). The figure below shows an example of a logical tree describing a compound datatype constructed from different datatypes.
Recall that the datatype is a description of the layout of storage. The complicated compound datatype is constructed from component datatypes, each of which describes the layout of part of the storage. Any datatype can be used as a component of a compound datatype, with the following restrictions: 
  * 1. No byte can be part of more than one component datatype (in other words, the fields cannot overlap within the compound datatype) 
  * 2. The total size of the components must be less than or equal to the total size of the compound datatype


These restrictions are essentially the rules for C structures and similar record types familiar from programming languages. Multiple typing, such as a C union, is not allowed in HDF5 datatypes.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig18.gif) A compound datatype built with different datatypes  
---  
###  Creating a Complicated Compound Datatype
To construct a complicated compound datatype, each component is constructed, and then added to the enclosing datatype description. The example below shows how to create a compound datatype with four members: 
  * “T1”, a compound datatype with three members 
  * “T2”, a compound datatype with two members 
  * “T3”, a one-dimensional array of integers 
  * “T4”, a string


Below the example code is a figure that shows this datatype as a logical tree. The output of the [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) utility is shown in the example below the figure.
Each datatype is created as a separate datatype object. Figure "The storage layout for the four member datatypes" below shows the storage layout for the four individual datatypes. Then the datatypes are inserted into the outer datatype at an appropriate offset. Figure "The storage layout of the combined four members" below shows the resulting storage layout. The combined record is 89 bytes long.
The Dataset is created using the combined compound datatype. The dataset is declared to be a 4 by 3 array of compound data. Each data element is an instance of the 89-byte compound datatype. Figure "The layout of the dataset" below shows the layout of the dataset, and expands one of the elements to show the relative position of the component data elements.
Each data element is a compound datatype, which can be written or read as a record, or each field may be read or written individually. The first field (“T1”) is itself a compound datatype with three fields (“T1.a”, “T1.b”, and “T1.c”). “T1” can be read or written as a record, or individual fields can be accessed. Similarly, the second filed is a compound datatype with two fields (“T2.f1”, “T2.f2”).
The third field (“T3”) is an array datatype. Thus, “T3” should be accessed as an array of 40 integers. Array data can only be read or written as a single element, so all 40 integers must be read or written to the third field. The fourth field (“T4”) is a single string of length 25.
_Create a compound datatype with four members_
typedef struct s1_t {
int a;
char b;
double c;
} s1_t;
typedef struct s2_t {
float f1;
float f2;
} s2_t;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) s1_tid, s2_tid, s3_tid, s4_tid, s5_tid;
// Create a datatype for s1
s1_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(s1_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “a_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, a), [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “b_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, b), [H5T_NATIVE_CHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga7439560bc4ef6d995a4e35b30262e660));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s1_tid, “c_name”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s1_t, c), [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09));
// Create a datatype for s2.
s2_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sizeof(s2_t));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s2_tid, “f1”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s2_t, f1), [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c));
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s2_tid, “f2”, [HOFFSET](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#af5242159129a7f37ab85d33d85a1ccac)(s2_t, f2), [H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c));
// Create a datatype for an Array of integers
s3_tid = [H5Tarray_create](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga39d41fdef06b3d7972ec2eab16ab5e96)([H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), [RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dim);
// Create a datatype for a String of 25 characters
s4_tid = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d));
[H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)(s4_tid, 25);
// Create a compound datatype composed of one of each of these types.
// The total size is the sum of the size of each.
sz = [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s1_tid) + [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s2_tid) + [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s3_tid) + [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s4_tid);
s5_tid = [H5Tcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaa9afc38e1a7d35e4d0bec24c569b3c65) ([H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6), sz);
// Insert the component types at the appropriate offsets.
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s5_tid, “T1”, 0, s1_tid);
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s5_tid, “T2”, sizeof(s1_t), s2_tid);
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s5_tid, “T3”, sizeof(s1_t) + sizeof(s2_t), s3_tid);
[H5Tinsert](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga487d8f64a76f48b6eeb7f402d3b8b081)(s5_tid, “T4”, (sizeof(s1_t) + sizeof(s2_t) + [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s3_tid)), s4_tid);
// Create the dataset with this datatype.
dataset = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, DATASETNAME, s5_tid, space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5)
#define RANK
**Definition** h5import.h:343
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig19.gif) Logical tree for the compound datatype with four members  
---  
_Output from[h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) for the compound datatype_
DATATYPE [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6) {
[H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6) {
[H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b) “a_name”;
[H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593) “b_name”;
[H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e) “c_name”;
} “T1”;
[H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6) {
[H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea) “f1”;
[H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea) “f2”;
} “T2”;
[H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854) { [10] [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b) } “T3”;
[H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) {
STRSIZE 25;
STRPAD [H5T_STR_NULLTERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a23c685afc240bbac4da23b36d8fd7e13);
CSET [H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663);
CTYPE [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d);
} “T4”;
}
[H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663)
@ H5T_CSET_ASCII
**Definition** H5Tpublic.h:96
[H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa)
@ H5T_STRING
**Definition** H5Tpublic.h:35
[H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854)
@ H5T_ARRAY
**Definition** H5Tpublic.h:42
[H5T_STR_NULLTERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a23c685afc240bbac4da23b36d8fd7e13)
@ H5T_STR_NULLTERM
**Definition** H5Tpublic.h:121
[H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593)
#define H5T_STD_I8LE
**Definition** H5Tpublic.h:451
The storage layout for the four member datatypes a) Compound type ‘s1_t’, size 16 bytes.   
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig20a.gif)  
b) Compound type ‘s2_t’, size 8 bytes.   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig20b.gif)  
c) Array type ‘s3_tid’, 40 integers, total size 40 bytes.   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig20c.gif)  
d) String type ‘s4_tid’, size 25 bytes.   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig20d.gif)  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig21.gif) The storage layout of the combined four members  
---  
  * A 4 x 3 array of Compound Datatype 
  * Element [1,1] expanded  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig22.gif) The layout of the dataset  
---  


###  Analyzing and Navigating a Compound Datatype
A complicated compound datatype can be analyzed piece by piece to discover the exact storage layout. In the example above, the outer datatype is analyzed to discover that it is a compound datatype with four members. Each member is analyzed in turn to construct a complete map of the storage layout.
The example below shows an example of code that partially analyzes a nested compound datatype. The name and overall offset and size of the component datatype is discovered, and then its type is analyzed depending on the datatype class. Through this method, the complete storage layout can be discovered.
_Output from[h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) for the compound datatype_
s1_tid = [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb)(dataset);
if ([H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898)(s1_tid) == [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6)) {
printf(“COMPOUND DATATYPE {\n”);
sz = [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s1_tid);
nmemb = [H5Tget_nmembers](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626)(s1_tid);
printf(“ %d bytes\n”,sz);
printf(“ %d members\n”,nmemb);
for (i =0; i < nmemb; i++) {
s2_tid = [H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb)(s1_tid, i);
if ([H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898)(s2_tid) == [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6)) {
// recursively analyze the nested type.
}
else if ([H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898)(s2_tid) == [H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854)) {
sz2 = [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s2_tid);
printf(“ %s: NESTED ARRAY DATATYPE offset %d size %d
{\n”, [H5Tget_member_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66)(s1_tid, i), [H5Tget_member_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af)(s1_tid, i), sz2);
[H5Tget_array_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473)(s2_tid, dim);
s3_tid = [H5Tget_super](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga331e8f7b388a50af77294018db788de3)(s2_tid);
// Etc., analyze the base type of the array
}
else {
// analyze a simple type
printf(“ %s: type code %d offset %d size %d\n”, [H5Tget_member_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66)(s1_tid, i),
[H5Tget_class](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga364545c053f925fec65880b235e37898)(s2_tid), [H5Tget_member_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af)(s1_tid, i), [H5Tget_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1b971589cd7a86f3e84affdee455564e)(s2_tid));
}
// and so on....
[H5Tget_array_dims](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_r_r_a_y.html#ga2b4fecf95c9c16e4431d8aba60995473)
#define H5Tget_array_dims
**Definition** H5version.h:1537
[H5Tget_nmembers](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga21bdfc706f71ebe298a433e74b5bc626)
int H5Tget_nmembers(hid_t type_id)
Retrieves the number of elements in a compound or enumeration datatype.
[H5Tget_member_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_e_n_u_m.html#ga64d1d807464d2011192f28115580fb66)
char * H5Tget_member_name(hid_t type_id, unsigned membno)
Retrieves the name of a compound or enumeration datatype member.
[H5Tget_member_offset](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#ga46cf2a60b54a08695635749c215af4af)
size_t H5Tget_member_offset(hid_t type_id, unsigned membno)
Retrieves the offset of a field of a compound datatype.
[H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb)
hid_t H5Tget_member_type(hid_t type_id, unsigned membno)
Returns the datatype of the specified member.
[H5Tget_super](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga331e8f7b388a50af77294018db788de3)
hid_t H5Tget_super(hid_t type)
Returns the base datatype from which a datatype is derived.
##  Life Cycle of the Datatype Object
Application programs access HDF5 datatypes through identifiers. Identifiers are obtained by creating a new datatype or by copying or opening an existing datatype. The identifier can be used until it is closed or until the library shuts down. See items a and b in the figure below. By default, a datatype is transient, and it disappears when it is closed.
When a dataset or attribute is created ([H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) or [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24)), its datatype is stored in the HDF5 file as part of the dataset or attribute object. See item c in the figure below. Once an object created, its datatype cannot be changed or deleted. The datatype can be accessed by calling [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb "Returns an identifier for a copy of the datatype for a dataset."), [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44 "Gets an attribute's datatype."), [H5Tget_super](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga331e8f7b388a50af77294018db788de3 "Returns the base datatype from which a datatype is derived."), or [H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb "Returns the datatype of the specified member."). See item d in the figure below. These calls return an identifier to a transient copy of the datatype of the dataset or attribute unless the datatype is a committed datatype. Note that when an object is created, the stored datatype is a copy of the transient datatype. If two objects are created with the same datatype, the information is stored in each object with the same effect as if two different datatypes were created and used.
A transient datatype can be stored using [H5Tcommit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd) in the HDF5 file as an independent, named object, called a committed datatype. Committed datatypes were formerly known as named datatypes. See item e in the figure below. Subsequently, when a committed datatype is opened with [H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209) (item f), or is obtained with [H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb "Returns the datatype of the specified member.") or similar call (item k), the return is an identifier to a transient copy of the stored datatype. The identifier can be used in the same way as other datatype identifiers except that the committed datatype cannot be modified. When a committed datatype is copied with [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e "Copies an existing datatype."), the return is a new, modifiable, transient datatype object (item f).
When an object is created using a committed datatype ([H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf), [H5Acreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24)), the stored datatype is used without copying it to the object. See item j in the figure below. In this case, if multiple objects are created using the same committed datatype, they all share the exact same datatype object. This saves space and makes clear that the datatype is shared. Note that a committed datatype can be shared by objects within the same HDF5 file, but not by objects in other files. For more information on copying committed datatypes to other HDF5 files, see the [Copying Committed Datatypes with H5Ocopy](https://support.hdfgroup.org/documentation/hdf5/latest/copying_committed.html) topic in the “Additional Resources” chapter.
A committed datatype can be deleted from the file by calling [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") which replaces [H5Gunlink](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gacb843cbd5bbb816cfa9c855463d1e51c "Removes the link to an object from a group."). See item i in the figure below. If one or more objects are still using the datatype, the committed datatype cannot be accessed with [H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209), but will not be removed from the file until it is no longer used. [H5Tget_member_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___c_o_m_p_o_u_n_d.html#gaf5de0eabe28246f040342e275b9a63eb "Returns the datatype of the specified member.") and similar calls will return a transient copy of the datatype.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig23.gif) Life cycle of a datatype  
---  
Transient datatypes are initially modifiable. Note that when a datatype is copied or when it is written to the file (when an object is created) or the datatype is used to create a composite datatype, a copy of the current state of the datatype is used. If the datatype is then modified, the changes have no effect on datasets, attributes, or datatypes that have already been created. See the figure below.
A transient datatype can be made read-only ([H5Tlock](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga523642dbf4c60a83127fff87664a965b "Locks a datatype.")). Note that the datatype is still transient, and otherwise does not change. A datatype that is immutable is read-only but cannot be closed except when the entire library is closed. The predefined types such as [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) are immutable transient types.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig24.gif) Transient datatype states: modifiable, read-only, and immutable  
---  
To create two or more datasets that share a common datatype, first commit the datatype, and then use that datatype to create the datasets. See the example below.
_Create a shareable datatype_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) t1 = ...some transient type...;
[H5Tcommit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd) (file, “shared_type”, t1, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset1 = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (file, “dset1”, t1, space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset2 = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (file, “dset2”, t1, space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset1 = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3) (file, “dset1”, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) t2 = [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb) (dset1);
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset3 = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (file, “dset3”, t2, space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dset4 = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf) (file, “dset4”, t2, space, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Tcommit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd)
#define H5Tcommit
**Definition** H5version.h:1515
Datatype APIs Function  | Description   
---|---  
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) [H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) location, const char *name) [H5Topen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga1d14b407603fdcedfbed1f723784c209) #define H5Topen **Definition** H5version.h:1548 | A committed datatype can be opened by calling this function, which returns a datatype identifier. The identifier should eventually be released by calling [H5Tclose()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gafcba4db244f6a4d71e99c6e72b8678f0 "Releases a datatype.") to release resources. The committed datatype returned by this function is read-only or a negative value is returned for failure. The location is either a file or group identifier.   
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) [H5Tcommit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga9a224eb59f0ba807789e3f8ba3a840cd) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) location, const char *name, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)) [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) int herr_t **Definition** H5public.h:252 | A transient datatype (not immutable) can be written to a file and turned into a committed datatype by calling this function. The location is either a file or group identifier and when combined with name refers to a new committed datatype.   
[htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) [H5Tcommitted](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga0eba38d8c49784269e71ac9fa79b0f0a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) type) [htri_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#aa8f6c28736dbd0f18388c67911d38aca) int htri_t **Definition** H5public.h:282 [H5Tcommitted](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#ga0eba38d8c49784269e71ac9fa79b0f0a) htri_t H5Tcommitted(hid_t type_id) Determines whether a datatype is a committed type or a transient type. | A type can be queried to determine if it is a committed type or a transient type. If this function returns a positive value then the type is committed. Datasets which return committed datatypes with [H5Dget_type()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb "Returns an identifier for a copy of the datatype for a dataset.") are able to share the datatype with other datasets in the same file.   
##  Data Transfer: Datatype Conversion and Selection
When data is transferred (write or read), the storage layout of the data elements may be different. For example, an integer might be stored on disk in big-endian byte order and read into memory with little-endian byte order. In this case, each data element will be transformed by the HDF5 Library during the data transfer.
The conversion of data elements is controlled by specifying the datatype of the source and specifying the intended datatype of the destination. The storage format on disk is the datatype specified when the dataset is created. The datatype of memory must be specified in the library call.
In order to be convertible, the datatype of the source and destination must have the same datatype class (with the exception of enumeration type). Thus, integers can be converted to other integers, and floats to other floats, but integers cannot (yet) be converted to floats. For each atomic datatype class, the possible conversions are defined. An enumeration datatype can be converted to an integer or a floating-point number datatype.
Basically, any datatype can be converted to another datatype of the same datatype class. The HDF5 Library automatically converts all properties. If the destination is too small to hold the source value then an overflow or underflow exception occurs. If a handler is defined with the [H5Pset_type_conv_cb](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_x_p_l.html#ga10a80b29444d933da1aa2003f46cf003 "Sets user-defined datatype conversion callback function.") function, it will be called. Otherwise, a default action will be performed. The table below summarizes the default actions.
Default actions for datatype conversion exceptions Datatype Class  | Possible Exceptions  | Default Action   
---|---|---  
Integer  | Size, offset, pad  |   
Float  | Size, offset, pad, ebits  |   
String  | Size  | Truncates, zero terminate if required.   
Enumeration  | No field  | All bits set   
For example, when reading data from a dataset, the source datatype is the datatype set when the dataset was created, and the destination datatype is the description of the storage layout in memory. The destination datatype must be specified in the [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c "Reads raw data from a dataset into a provided buffer.") call. The example below shows an example of reading a dataset of 32-bit integers. The figure below the example shows the data transformation that is performed.
_Specify the destination datatype with H5Dread_
// Stored as H5T_STD_BE32
// Use the native memory order in the destination
mem_type_id = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, buf);
Layout of a datatype conversion ![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig25a.gif)   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig25b.gif)   
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig25c.gif)  
---  
One thing to note in the example above is the use of the predefined native datatype [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2). Recall that in this example, the data was stored as a 4-bytes in big-endian order. The application wants to read this data into an array of integers in memory. Depending on the system, the storage layout of memory might be either big or little-endian, so the data may need to be transformed on some platforms and not on others. The [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) type is set by the HDF5 Library to be the correct type to describe the storage layout of the memory on the system. Thus, the code in the example above will work correctly on any platform, performing a transformation when needed.
There are predefined native types for most atomic datatypes, and these can be combined in composite datatypes. In general, the predefined native datatypes should always be used for data stored in memory. Predefined native datatypes describe the storage properties of memory.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig26.gif) An enum datatype conversion  
---  
_Create an aligned and packed compound datatype_
// Stored as H5T_STD_BE32
// Use the native memory order in the destination
mem_type_id = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, buf);
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig27.gif) Alignment of a compound datatype  
---  
_Transfer some fields of a compound datatype_
// Stored as H5T_STD_BE32
// Use the native memory order in the destination
mem_type_id = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2));
status = [H5Dread](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga8287d5a7be7b8e55ffeff68f7d26811c)(dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, buf);
![](https://support.hdfgroup.org/documentation/hdf5/latest/Dtypes_fig28.gif) Layout when an element is skipped  
---  
##  Text Descriptions of Datatypes: Conversion to and from
HDF5 provides a means for generating a portable and human-readable text description of a datatype and for generating a datatype from such a text description. This capability is particularly useful for creating complex datatypes in a single step, for creating a text description of a datatype for debugging purposes, and for creating a portable datatype definition that can then be used to recreate the datatype on many platforms or in other applications.
These tasks are handled by two functions provided in the HDF5 Lite high-level library: 
  * [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f "Creates an HDF5 datatype given a text description.") Creates an HDF5 datatype in a single step. 
  * [H5LTdtype_to_text](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga155a5e59441ed05ca1b91e6404314862 "Creates a text description of an HDF5 datatype.") Translates an HDF5 datatype into a text description.


Note that this functionality requires that the HDF5 High-Level Library (H5LT) be installed.
While [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f "Creates an HDF5 datatype given a text description.") can be used to generate any sort of datatype, it is particularly useful for complex datatypes.
[H5LTdtype_to_text](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga155a5e59441ed05ca1b91e6404314862 "Creates a text description of an HDF5 datatype.") is most likely to be used in two sorts of situations: when a datatype must be closely examined for debugging purpose or to create a portable text description of the datatype that can then be used to recreate the datatype on other platforms or in other applications.
These two functions work for all valid HDF5 datatypes except time, bitfield, and reference datatypes.
The currently supported text format used by [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f "Creates an HDF5 datatype given a text description.") and [H5LTdtype_to_text](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#ga155a5e59441ed05ca1b91e6404314862 "Creates a text description of an HDF5 datatype.") is the data description language (DDL) and conforms to the [DDL in BNF for HDF5 2.0.0 and above](https://support.hdfgroup.org/documentation/hdf5/latest/_d_d_l_b_n_f200.html). The portion of the [DDL in BNF for HDF5 2.0.0 and above](https://support.hdfgroup.org/documentation/hdf5/latest/_d_d_l_b_n_f200.html) that defines HDF5 datatypes appears below.
_The definition of HDF5 datatypes from the HDF5 DDL_
<datatype> ::= <atomic_type> | <compound_type> | <variable_length_type> | <array_type> |
<complex_type>
<atomic_type> ::= <integer> | <float> | <time> | <string> |
<bitfield> | <opaque> | <reference> | <enum>
<integer> ::= [H5T_STD_I8BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gafc97668c32b74f7d20dec0d0d2f47e4f) | [H5T_STD_I8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaeaa5eb139424ba27d5af8925f1aa9593) |
[H5T_STD_I16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc0ccea703096e20a4c3e51e858836dd) | [H5T_STD_I16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf88985315398de5fc4a716707b5c7694) |
[H5T_STD_I32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga37e8a6be7ee64587c2a282b965019bb8) | [H5T_STD_I32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8db8c9c2bcc457f9f8526c8fcb81218b) |
[H5T_STD_I64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga7a14305593830bbe08a622642eae928c) | [H5T_STD_I64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga591a85b894eab3e3ab1a2ccd9b3be565) |
[H5T_STD_U8BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga10b5774adca6e8432615dac892e13ed9) | [H5T_STD_U8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gae641eb8f98d930fe5ddd3a04adb98383) |
[H5T_STD_U16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga10e46ab72ac0330c51ed6cf709db4697) | [H5T_STD_U16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga4c2494c16f3e9443af1d56a078e0db3c) |
[H5T_STD_U32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga320fc4cfe8e4a3d1ab9997c8e342780b) | [H5T_STD_U32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaa03600cecf2cd9063346084e41eb82a3) |
[H5T_STD_U64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga417489ff74b1cc52d25259c6357dcc6b) | [H5T_STD_U64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga3a38267be678af40576a380617e3fff9) |
[H5T_NATIVE_CHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga7439560bc4ef6d995a4e35b30262e660) | [H5T_NATIVE_UCHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga39cadf50f1701a2af3afa86eba3c7cbd) |
[H5T_NATIVE_SHORT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae0625357fa121eca117f7fa9c4701810) | [H5T_NATIVE_USHORT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gad0a240282e54647b83fe28ef65b40655) |
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) | [H5T_NATIVE_UINT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga904b507c7b8aa4838fbb7c6ce71a37c5) |
[H5T_NATIVE_LONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga290b9655882754ee0ec9f31b42ac04f6) | [H5T_NATIVE_ULONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gacaea6c2c381167bacf67f6d6c43eb718) |
[H5T_NATIVE_LLONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gafb9c5d393d693869d7d21f71753a532c) | [H5T_NATIVE_ULLONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga91ed0d2ce3289ef3707449cf5babe928)
<float> ::= [H5T_IEEE_F16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga2c910c75e941e9e8a885343fa48d84cb) | [H5T_IEEE_F16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#gac1735e1dcc978313a6605b62f935282a) |
[H5T_IEEE_F32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga71d24a7d4c373ed9a003d7a0d8133f1e) | [H5T_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga994ce9ffdd2c77a9352b102a883503ea) |
[H5T_IEEE_F64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#gaf5064f4498d92e5992a5a0564d026d35) | [H5T_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga3f9c2185ec16632fab8c20ec7a63178e) |
[H5T_FLOAT_BFLOAT16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga4dd1e86cccbdcf1c22fddbf6ad88057e) | [H5T_FLOAT_BFLOAT16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga8fb1cbb4415c84c04663040597ffc009) |
[H5T_FLOAT_F8E4M3](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga8cfa202d0101ec59d2f270d0d5645c58) | [H5T_FLOAT_F8E5M2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#gaf2e1617cdd3d211350dd9414af1ff00f) |
[H5T_FLOAT_F6E2M3](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#gaadc2005e8d8a6050cd9d0ffc1531374d) | [H5T_FLOAT_F6E3M2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga1ce111c9e0e8ab22acde631e2b6ee213) |
[H5T_FLOAT_F4E2M1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga79510bbf529a08cf543edf0cdb984213) | [H5T_NATIVE_FLOAT16](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga596ee6c1b8226b318eef1bca8ac7d20f) |
[H5T_NATIVE_FLOAT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae2523b63144b498f555fa4d04f59ee1c) | [H5T_NATIVE_DOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga087f3b793a299e416bd68678c2ef7c09) |
[H5T_NATIVE_LDOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga6b68c680440c463ce03c5efbf8cdf1c0)
<time> ::= [H5T_TIME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a8d3af61b1a73c5682f7f9b131754f6e3): not yet implemented
<string> ::= [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) {
STRSIZE <strsize> ;
STRPAD <strpad> ;
CSET <cset> ;
CTYPE <ctype> ;
}
<strsize> ::= <int_value>
<strpad> ::= [H5T_STR_NULLTERM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a23c685afc240bbac4da23b36d8fd7e13) | [H5T_STR_NULLPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a128d51156e51b7a2c9db0fe8787b4547) | [H5T_STR_SPACEPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a3f73f8dae99444798f5efd7d2d2a5e5c)
<cset> ::= [H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663) | [H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)
<ctype> ::= [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d) | [H5T_FORTRAN_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga62335f6f57c2809fa1b3b1f9472eb2f6)
<bitfield> ::= [H5T_STD_B8BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaaa2c22c363edaa8587e954d88cc67f2a) | [H5T_STD_B8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8ecfe3ba07af971949a94b157f518940) |
[H5T_STD_B16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga1eadeed01d1dfee62689a51acac1a159) | [H5T_STD_B16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga61fa8825fb07bcdaa285967fa3f3bcb9) |
[H5T_STD_B32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga556365dedd52e74e13d198071d47da1d) | [H5T_STD_B32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gad166a8c49ae8af0ce1fe7812bdb90e19) |
[H5T_STD_B64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga56eb9e49fe4ea45dc80ee6bbe0880625) | [H5T_STD_B64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga195cc861daefb50256990b68545444d8)
<opaque> ::= [H5T_OPAQUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aaf11325a64ed5369e88d8d0d600b5cce) {
OPAQUE_TAG <identifier>;
OPAQUE_SIZE <int_value>;opt
}
<reference> ::= [H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd) { <ref_type> }
<ref_type> ::= H5T_STD_REF_OBJECT | [H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e) | [H5T_STD_REF](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga342d286867f160739f9f5359303759cc) | UNDEFINED
<compound_type> ::= [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6) {
<member_type_def>+
}
<member_type_def> ::= <datatype> <field_name>;
<field_name> ::= <identifier>
<variable_length_type> ::= [H5T_VLEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2ad8ff83b6b7ca22575263561221193028) { <datatype> }
<array_type> ::= [H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854) { <dim_sizes> <datatype> }
<dim_sizes> ::= '['<dimsize>']' | '['<dimsize>']'<dim_sizes>
<dimsize> ::= <int_value>
<enum> ::= [H5T_ENUM](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5ee305303f12787367ac271d8f28f2e6) {
<enum_base_type> <enum_def>+
}
<enum_base_type> ::= <integer>
// Currently enums can only hold integer type data, but they may be expanded
// in the future to hold any datatype
<enum_def> ::= <enum_symbol> <enum_val>;
<enum_symbol> ::= <identifier>
<enum_val> ::= <int_value>
<complex_type> ::= [H5T_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2af2675ccd96bbe2c60daf885185774120) { <complex_base_type> <complex_base_type> } |
[H5T_COMPLEX_IEEE_F16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#gad5759434265beea7545d170a17033410) | [H5T_COMPLEX_IEEE_F16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga3a5ac018d52f68b0453a9dbfbd3cb259) |
[H5T_COMPLEX_IEEE_F32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga2e539a7d52c9f14f0e999617c96e3d52) | [H5T_COMPLEX_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga195a2f2f9614f855d21090e4fbff9f04) |
[H5T_COMPLEX_IEEE_F64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga88831575ae080e95865edb0aaa7ee6ad) | [H5T_COMPLEX_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga492b3488326a498233ab8e6ad8d38719) |
[H5T_NATIVE_FLOAT_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga957f6b409daca9447dfcf671c9b18610) | [H5T_NATIVE_DOUBLE_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga00cdc9734afbfead614191bd9736bc37) |
[H5T_NATIVE_LDOUBLE_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga841b6107417a1bb8bae085333ee7a6c8)
<complex_base_type> ::= <float>
// Currently complex number datatypes can only hold homogeneous floating-point
// type data, but they may be expanded in the future to hold heterogeneous
// floating-point type data or even non-floating-point type data
[H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)
@ H5T_CSET_UTF8
**Definition** H5Tpublic.h:97
[H5T_REFERENCE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a5850e0b9353a5e7aeb615fb943d4e9cd)
@ H5T_REFERENCE
**Definition** H5Tpublic.h:39
[H5T_TIME](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a8d3af61b1a73c5682f7f9b131754f6e3)
@ H5T_TIME
**Definition** H5Tpublic.h:34
[H5T_OPAQUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aaf11325a64ed5369e88d8d0d600b5cce)
@ H5T_OPAQUE
**Definition** H5Tpublic.h:37
[H5T_VLEN](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2ad8ff83b6b7ca22575263561221193028)
@ H5T_VLEN
**Definition** H5Tpublic.h:41
[H5T_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2af2675ccd96bbe2c60daf885185774120)
@ H5T_COMPLEX
**Definition** H5Tpublic.h:43
[H5T_STR_NULLPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a128d51156e51b7a2c9db0fe8787b4547)
@ H5T_STR_NULLPAD
**Definition** H5Tpublic.h:122
[H5T_STR_SPACEPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a3f73f8dae99444798f5efd7d2d2a5e5c)
@ H5T_STR_SPACEPAD
**Definition** H5Tpublic.h:123
[H5T_FLOAT_F6E3M2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga1ce111c9e0e8ab22acde631e2b6ee213)
#define H5T_FLOAT_F6E3M2
**Definition** H5Tpublic.h:367
[H5T_FLOAT_BFLOAT16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga4dd1e86cccbdcf1c22fddbf6ad88057e)
#define H5T_FLOAT_BFLOAT16BE
**Definition** H5Tpublic.h:298
[H5T_FLOAT_F4E2M1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga79510bbf529a08cf543edf0cdb984213)
#define H5T_FLOAT_F4E2M1
**Definition** H5Tpublic.h:383
[H5T_FLOAT_F8E4M3](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga8cfa202d0101ec59d2f270d0d5645c58)
#define H5T_FLOAT_F8E4M3
**Definition** H5Tpublic.h:319
[H5T_FLOAT_BFLOAT16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#ga8fb1cbb4415c84c04663040597ffc009)
#define H5T_FLOAT_BFLOAT16LE
**Definition** H5Tpublic.h:303
[H5T_FLOAT_F6E2M3](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#gaadc2005e8d8a6050cd9d0ffc1531374d)
#define H5T_FLOAT_F6E2M3
**Definition** H5Tpublic.h:351
[H5T_FLOAT_F8E5M2](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_a_l_t_f_l_o_a_t.html#gaf2e1617cdd3d211350dd9414af1ff00f)
#define H5T_FLOAT_F8E5M2
**Definition** H5Tpublic.h:335
[H5T_COMPLEX_IEEE_F32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga195a2f2f9614f855d21090e4fbff9f04)
#define H5T_COMPLEX_IEEE_F32LE
**Definition** H5Tpublic.h:418
[H5T_COMPLEX_IEEE_F32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga2e539a7d52c9f14f0e999617c96e3d52)
#define H5T_COMPLEX_IEEE_F32BE
**Definition** H5Tpublic.h:412
[H5T_COMPLEX_IEEE_F16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga3a5ac018d52f68b0453a9dbfbd3cb259)
#define H5T_COMPLEX_IEEE_F16LE
**Definition** H5Tpublic.h:406
[H5T_COMPLEX_IEEE_F64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga492b3488326a498233ab8e6ad8d38719)
#define H5T_COMPLEX_IEEE_F64LE
**Definition** H5Tpublic.h:430
[H5T_COMPLEX_IEEE_F64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#ga88831575ae080e95865edb0aaa7ee6ad)
#define H5T_COMPLEX_IEEE_F64BE
**Definition** H5Tpublic.h:424
[H5T_COMPLEX_IEEE_F16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_c_o_m_p_l_e_x.html#gad5759434265beea7545d170a17033410)
#define H5T_COMPLEX_IEEE_F16BE
**Definition** H5Tpublic.h:400
[H5T_IEEE_F16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga2c910c75e941e9e8a885343fa48d84cb)
#define H5T_IEEE_F16BE
**Definition** H5Tpublic.h:253
[H5T_IEEE_F32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#ga71d24a7d4c373ed9a003d7a0d8133f1e)
#define H5T_IEEE_F32BE
**Definition** H5Tpublic.h:265
[H5T_IEEE_F64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_i_e_e_e.html#gaf5064f4498d92e5992a5a0564d026d35)
#define H5T_IEEE_F64BE
**Definition** H5Tpublic.h:277
[H5T_NATIVE_DOUBLE_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga00cdc9734afbfead614191bd9736bc37)
#define H5T_NATIVE_DOUBLE_COMPLEX
**Definition** H5Tpublic.h:1056
[H5T_NATIVE_LONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga290b9655882754ee0ec9f31b42ac04f6)
#define H5T_NATIVE_LONG
**Definition** H5Tpublic.h:1000
[H5T_NATIVE_UCHAR](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga39cadf50f1701a2af3afa86eba3c7cbd)
#define H5T_NATIVE_UCHAR
**Definition** H5Tpublic.h:970
[H5T_NATIVE_FLOAT16](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga596ee6c1b8226b318eef1bca8ac7d20f)
#define H5T_NATIVE_FLOAT16
**Definition** H5Tpublic.h:1024
[H5T_NATIVE_LDOUBLE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga6b68c680440c463ce03c5efbf8cdf1c0)
#define H5T_NATIVE_LDOUBLE
**Definition** H5Tpublic.h:1042
[H5T_NATIVE_LDOUBLE_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga841b6107417a1bb8bae085333ee7a6c8)
#define H5T_NATIVE_LDOUBLE_COMPLEX
**Definition** H5Tpublic.h:1063
[H5T_NATIVE_ULLONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga91ed0d2ce3289ef3707449cf5babe928)
#define H5T_NATIVE_ULLONG
**Definition** H5Tpublic.h:1018
[H5T_NATIVE_FLOAT_COMPLEX](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga957f6b409daca9447dfcf671c9b18610)
#define H5T_NATIVE_FLOAT_COMPLEX
**Definition** H5Tpublic.h:1049
[H5T_NATIVE_ULONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gacaea6c2c381167bacf67f6d6c43eb718)
#define H5T_NATIVE_ULONG
**Definition** H5Tpublic.h:1006
[H5T_NATIVE_USHORT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gad0a240282e54647b83fe28ef65b40655)
#define H5T_NATIVE_USHORT
**Definition** H5Tpublic.h:982
[H5T_NATIVE_SHORT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gae0625357fa121eca117f7fa9c4701810)
#define H5T_NATIVE_SHORT
**Definition** H5Tpublic.h:976
[H5T_NATIVE_LLONG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#gafb9c5d393d693869d7d21f71753a532c)
#define H5T_NATIVE_LLONG
**Definition** H5Tpublic.h:1012
[H5T_FORTRAN_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga62335f6f57c2809fa1b3b1f9472eb2f6)
#define H5T_FORTRAN_S1
**Definition** H5Tpublic.h:657
[H5T_STD_U8BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga10b5774adca6e8432615dac892e13ed9)
#define H5T_STD_U8BE
**Definition** H5Tpublic.h:486
[H5T_STD_U16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga10e46ab72ac0330c51ed6cf709db4697)
#define H5T_STD_U16BE
**Definition** H5Tpublic.h:496
[H5T_STD_B64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga195cc861daefb50256990b68545444d8)
#define H5T_STD_B64LE
**Definition** H5Tpublic.h:561
[H5T_STD_B16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga1eadeed01d1dfee62689a51acac1a159)
#define H5T_STD_B16BE
**Definition** H5Tpublic.h:536
[H5T_STD_U32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga320fc4cfe8e4a3d1ab9997c8e342780b)
#define H5T_STD_U32BE
**Definition** H5Tpublic.h:506
[H5T_STD_U64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga3a38267be678af40576a380617e3fff9)
#define H5T_STD_U64LE
**Definition** H5Tpublic.h:521
[H5T_STD_U64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga417489ff74b1cc52d25259c6357dcc6b)
#define H5T_STD_U64BE
**Definition** H5Tpublic.h:516
[H5T_STD_U16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga4c2494c16f3e9443af1d56a078e0db3c)
#define H5T_STD_U16LE
**Definition** H5Tpublic.h:501
[H5T_STD_B32BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga556365dedd52e74e13d198071d47da1d)
#define H5T_STD_B32BE
**Definition** H5Tpublic.h:546
[H5T_STD_B64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga56eb9e49fe4ea45dc80ee6bbe0880625)
#define H5T_STD_B64BE
**Definition** H5Tpublic.h:556
[H5T_STD_I64LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga591a85b894eab3e3ab1a2ccd9b3be565)
#define H5T_STD_I64LE
**Definition** H5Tpublic.h:481
[H5T_STD_B16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga61fa8825fb07bcdaa285967fa3f3bcb9)
#define H5T_STD_B16LE
**Definition** H5Tpublic.h:541
[H5T_STD_I64BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga7a14305593830bbe08a622642eae928c)
#define H5T_STD_I64BE
**Definition** H5Tpublic.h:476
[H5T_STD_B8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#ga8ecfe3ba07af971949a94b157f518940)
#define H5T_STD_B8LE
**Definition** H5Tpublic.h:531
[H5T_STD_U32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaa03600cecf2cd9063346084e41eb82a3)
#define H5T_STD_U32LE
**Definition** H5Tpublic.h:511
[H5T_STD_B8BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaaa2c22c363edaa8587e954d88cc67f2a)
#define H5T_STD_B8BE
**Definition** H5Tpublic.h:526
[H5T_STD_B32LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gad166a8c49ae8af0ce1fe7812bdb90e19)
#define H5T_STD_B32LE
**Definition** H5Tpublic.h:551
[H5T_STD_I16BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc0ccea703096e20a4c3e51e858836dd)
#define H5T_STD_I16BE
**Definition** H5Tpublic.h:456
[H5T_STD_U8LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gae641eb8f98d930fe5ddd3a04adb98383)
#define H5T_STD_U8LE
**Definition** H5Tpublic.h:491
[H5T_STD_REF_DSETREG](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf5cb0d5cec3d40d8b3ac27512f86895e)
#define H5T_STD_REF_DSETREG
**Definition** H5Tpublic.h:571
[H5T_STD_I16LE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gaf88985315398de5fc4a716707b5c7694)
#define H5T_STD_I16LE
**Definition** H5Tpublic.h:461
[H5T_STD_I8BE](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gafc97668c32b74f7d20dec0d0d2f47e4f)
#define H5T_STD_I8BE
**Definition** H5Tpublic.h:446
_Old definitions of the opaque and compound datatypes_
<opaque> ::= [H5T_OPAQUE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2aaf11325a64ed5369e88d8d0d600b5cce) { <identifier> }
<compound_type> ::= [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6) { <member_type_def>+ }
<member_type_def> ::= <datatype> <field_name> ;
<field_name> ::= <identifier>
#### Examples
The code sample below illustrates the use of [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f "Creates an HDF5 datatype given a text description.") to generate a variable-length string datatype.
_Creating a variable-length string datatype from a text description_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype;
if((dtype = [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f)(
“H5T_STRING {
STRSIZE [H5T_VARIABLE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a5185e14efde13b48249fe391324331bc);
STRPAD [H5T_STR_NULLPAD](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#ad2ab726f3bd28222a2ffb91c6bbc3514a128d51156e51b7a2c9db0fe8787b4547);
CSET [H5T_CSET_ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa27383e03d1cad9b4c32d8611a145d663);
CTYPE [H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d);
}”, [H5LT_DDL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63a158bbaa59144a89547203f8e95421911))) < 0)
goto out;
[H5LT_DDL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63a158bbaa59144a89547203f8e95421911)
@ H5LT_DDL
**Definition** H5LTpublic.h:27
[H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f)
H5HL_DLL hid_t H5LTtext_to_dtype(const char *text, H5LT_lang_t lang_type)
Creates an HDF5 datatype given a text description.
The code sample below illustrates the use of [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f "Creates an HDF5 datatype given a text description.") to generate a complex array datatype.
_Creating a complex array datatype from a text description_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) dtype;
if((dtype = [H5LTtext_to_dtype](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l_t.html#gad330b1eba5005b582daad787a638660f)(
“H5T_ARRAY { [5][7][13] [H5T_ARRAY](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a547a4451911e912127f300ab15113854)
{ [17][19] [H5T_COMPOUND](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a7a401c61604dc846dbd3f9eb6fcb0fe6)
{
H5T_STD_I8BE \“arr_compound_1\”;
H5T_STD_I32BE \“arr_compound_2\”;
}
}
}”, [H5LT_DDL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_l_tpublic_8h.html#afe4d47e38833b425f23f15bdbd14fd63a158bbaa59144a89547203f8e95421911)))<0)
goto out;
Previous Chapter [HDF5 Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#sec_dataset) - Next Chapter [HDF5 Dataspaces and Partial I/O](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_s__u_g.html#sec_dataspace)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


