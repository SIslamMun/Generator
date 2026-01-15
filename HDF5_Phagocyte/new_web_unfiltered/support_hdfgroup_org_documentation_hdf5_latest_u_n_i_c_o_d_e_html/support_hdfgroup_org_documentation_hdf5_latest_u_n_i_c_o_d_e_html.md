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
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title0 "H1")
  * [ How and Where Is UTF-8 Supported in HDF5?](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title1 "H1")
  * [ Object and Attribute Names](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title2 "H2")
  * [ Character Datatypes in Datasets and Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title3 "H2")
  * [ Caveats, Pitfalls, and Things to Watch For](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title4 "H1")
  * [ Cross-platform Portability](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title5 "H2")
  * [ Filenames](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title6 "H2")
  * [ The Plain Text Illusion](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title7 "H1")
  * [ Storage Size](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title8 "H2")
  * [ System Dependencies](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title9 "H2")
  * [ Common Characters in UTF-8 and ASCII](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title10 "H1")
  * [ See Also](https://support.hdfgroup.org/documentation/hdf5/latest/_u_n_i_c_o_d_e.html#title11 "H1")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Using UTF-8 Encoding in HDF5 Applications
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
#  Introduction
Text and character data are often discussed as though text means ASCII text. We even go so far as to call a file containing only ASCII text a plain text file. This works reasonably well for English (though better for American English than British English), but what if that plain text file is in French, German, Chinese, or any of several hundred other languages? This document introduces the use of UTF-8 encoding (see note 1), enabling the use of a much more extensive and flexible character set that can faithfully represent any of those languages.
This document assumes a working familiarity with UTF-8 and Unicode. Any reader who is unfamiliar with UTF-8 encoding should read the [Wikipedia UTF-8 article](https://en.wikipedia.org/wiki/UTF-8) before proceeding; it provides an excellent primer.
For our context, the most important UTF-8 concepts are: 
  * Multi-byte and variable-size character encodings 
  * Limitations of the ASCII character set 
  * Risks associated with the use of the term plain text 
  * Representation of multiple language alphabets or characters in a single document


More specific technical details will only become important if they affect the specifics of your application design or implementation.
#  How and Where Is UTF-8 Supported in HDF5?
HDF5 uses characters in object names (which are actually link names, but that's a story for a different article), dataset raw data, attribute names, and attribute raw data. Though the mechanisms differ, you can use either ASCII or UTF-8 character sets in all of these situations.
##  Object and Attribute Names
By default, HDF5 creates object and attribute names with ASCII character encoding. An object or attribute creation property list setting is required to create object names with UTF-8 characters. This uses the function [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names."), which sets the character encoding used for object and attribute names.
For example, the following call sequence could be used to create a dataset with its name encoded with the UTF-8 character set:
lcpl_id = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_LINK_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad2c1d10104f0262c826350ccbf7c49f2)) ;
error = [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc)(lcpl_id, [H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)) ;
dataset_id = [H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)(group_id, "datos_ñ", datatype_id, dataspace_id, 
lcpl_id, [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)) ; 
[H5P_LINK_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#ad2c1d10104f0262c826350ccbf7c49f2)
#define H5P_LINK_CREATE
**Definition** H5Ppublic.h:116
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658)
@ H5T_CSET_UTF8
**Definition** H5Tpublic.h:97
[H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc)
herr_t H5Pset_char_encoding(hid_t plist_id, H5T_cset_t encoding)
Sets the character encoding used to encode link and attribute names.
[H5Dcreate2](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#gabf62045119f4e9c512d87d77f2f992df)
hid_t H5Dcreate2(hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)
Creates a new dataset and links it into the file.
[H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)
hid_t H5Pcreate(hid_t cls_id)
Creates a new property list as an instance of a property list class.
If the character encoding of an attribute name is unknown, the combination of an [H5Aget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0f6b545850bd21f128904eff51df226d "Gets an attribute creation property list identifier.") call and an [H5Pget_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d "Retrieves the character encoding used to create a link or attribute name.") call will reveal that information. If the character encoding of an object name is unknown, the information can be accessed through the object's H5L_info_t structure which can be obtained using [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) or [H5Lget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga4db00b8b944eae68233438165c784b67) calls.
##  Character Datatypes in Datasets and Attributes
Like object names, HDF5 character data in datasets and attributes is encoded as ASCII by default. Setting up attribute or dataset character data to be UTF-8-encoded is accomplished while defining the attribute or dataset datatype. This makes use of the function [H5Tset_cset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga4909c0c3d97c3d212fee032cc8dc031a "Sets character set to be used in a string or character datatype."), which sets the character encoding to be used in building a character datatype.
For example, the following commands could be used to create an 8-character, UTF-8 encoded, string datatype for use in either an attribute or dataset:
datatype_id = [H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)([H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d));
error = [H5Tset_cset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga4909c0c3d97c3d212fee032cc8dc031a)(datatype_id, [H5T_CSET_UTF8](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a03755b8370672668ddc7063add28e71aa41685667f69bf81eb7de5dd5f452e658));
error = [H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)(datatype_id, "8");
[H5Tset_cset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga4909c0c3d97c3d212fee032cc8dc031a)
herr_t H5Tset_cset(hid_t type_id, H5T_cset_t cset)
Sets character set to be used in a string or character datatype.
[H5Tset_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gae5f38bfd4a4c557496b3194b5180212c)
herr_t H5Tset_size(hid_t type_id, size_t size)
Sets size for a datatype.
[H5Tcopy](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html#gaec07efbab84f4e5b4ed22f010786be8e)
hid_t H5Tcopy(hid_t type_id)
Copies an existing datatype.
[H5T_C_S1](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s.html#ga7f6b6959fefe56d2e871659110936d2d)
#define H5T_C_S1
**Definition** H5Tpublic.h:646
If a character or string datatype's character encoding is unknown, the combination of an [H5Aget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga0b070b714b2e535df2e1cb3005026a44 "Gets an attribute's datatype.") or [H5Dget_type](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga7cd04b8332e8a0939b9973fbc500cadb "Returns an identifier for a copy of the datatype for a dataset.") call and an [H5Tget_cset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga5bc2f3e8f708f5bcdd0d8667950310c1 "Retrieves the character set type of a string datatype.") call can be used to determine that.
#  Caveats, Pitfalls, and Things to Watch For
Programmers who are accustomed to using ASCII text without accommodating other text encodings will have to be aware of certain common issues as they begin using UTF-8 encodings.
##  Cross-platform Portability
Since the HDF5 Library handles datatypes directly, UTF-8 encoded text in dataset and attribute datatypes in a well-designed HDF5 application and file should work transparently across platforms. The same should be true of handling names of groups, datasets, committed datatypes, and attributes within a file.
Be aware, however, of system or application limitations once data or other information has been extracted from an HDF5 file. The application or system must be designed to accommodate UTF-8 encodings if the information is then used elsewhere in the application or system environment.
Data from a UTF-8 encoded HDF5 datatype, in either a dataset or an attribute, that has been established within an HDF5 application should "just work" within the HDF5 portions of the application.
##  Filenames
Since file access is a system issue, filenames do not fall within the scope of HDF5's UTF-8 capabilities; filenames are encoded at the system level.
Linux and Mac OS systems normally handle UTF-8 encoded filenames correctly while Windows systems generally do not.
#  The _Plain Text_ Illusion
Beware the use of the term _plain text_. _Plain text_ is at best ambiguous, but often misleading. Many will assume that _plain text_ means ASCII, but plain text German or French, for example, cannot be represented in ASCII. Plain text is only unambiguous in the context of English (and even then can be problematic!).
##  Storage Size
Programmers and data users accustomed to working strictly with ASCII data generally make the reasonable assumption that 1 character, be it in an object name or in data, requires 1 byte of storage. This equation does not work when using UTF-8 or any other Unicode encoding. With Unicode encoding, number of characters is not synonymous with number of bytes. One must get used to thinking in terms of number of characters when talking about content, reserving number of bytes for discussions of storage size.
When working with Unicode text, one can no longer assume a 1:1 correspondence between the number of characters and the data storage requirement.
##  System Dependencies
**Linux** , **Unix** , and similar systems generally handle UTF-8 encodings in correct and predictable ways. There is an apparent consensus in the Linux community that "UTF-8 is just the right way to go."
**Mac OS** systems generally handle UTF-8 encodings correctly.
**Windows** systems use a different Unicode encoding, [UCS-2](https://en.wikipedia.org/wiki/UTF-16) (discussed in this UTF-16 article) at the system level. Within an HDF5 file and application on a Windows system, UTF-8 encoding should work correctly and as expected. Problems may arise, however, when that UTF-8 encoding is exposed directly to the Windows system. For example: 
  * File open and close calls on files with UTF-8 encoded names are likely to fail as the HDF5 open and close operations interact directly with the Windows file system interface. 
  * Anytime an HDF5 command-line utility ([The HDF5 h5ls Tool](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html) or [The HDF5 h5dump Tool](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html), for example) emits text output, the Windows system must interpret the character encodings. If that output is UTF-8 encoded, Windows will correctly interpret only those characters in the ASCII subset of UTF-8.


#  Common Characters in UTF-8 and ASCII
One interesting feature of UTF-8 and ASCII is that the ASCII character set is a discrete subset of the UTF-8 character set. And where they overlap, the encodings are identical. This means that a character string consisting entirely of members of the ASCII character set can be encoded in either ASCII or UTF-8, the two encodings will be indistinguishable, and the encodings will require exactly the same storage space.
#  See Also
  * For object and attribute names: [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.") [H5Pget_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d "Retrieves the character encoding used to create a link or attribute name.")
  * For dataset and attribute datatypes: [H5Tset_cset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga4909c0c3d97c3d212fee032cc8dc031a "Sets character set to be used in a string or character datatype.") [H5Tget_cset](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html#ga5bc2f3e8f708f5bcdd0d8667950310c1 "Retrieves the character set type of a string datatype.")
  * [UTF-8 article on Wikipedia](https://en.wikipedia.org/wiki/UTF-8)


### NOTES
  1. UTF-8 is the only Unicode standard encoding supported in HDF5.


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


