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
  * [ History](https://support.hdfgroup.org/documentation/hdf5/latest/_about.html#title0 "H1")
  * [ Documentation about Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_about.html#title1 "H1")
  * [ Including Plain HTML Pages](https://support.hdfgroup.org/documentation/hdf5/latest/_about.html#title2 "H2")
  * [ Creating a New Reference Manual Entry](https://support.hdfgroup.org/documentation/hdf5/latest/_about.html#title3 "H2")
  * [ Creating Custom Commands](https://support.hdfgroup.org/documentation/hdf5/latest/_about.html#title4 "H2")
  * [ Adding a New RFC or Referencing an Existing RFC](https://support.hdfgroup.org/documentation/hdf5/latest/_about.html#title5 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
About
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html)
* * *
#  History
The implementation of this documentation set is based on the fantastic work of the [Eigen project](https://eigen.tuxfamily.org/index.php?title=Main_Page). Please refer to their [GitLab repository](https://gitlab.com/libeigen/eigen) and the online version of their [Doxygen-based documentation](http://eigen.tuxfamily.org/dox/). Not only does Eigen set a standard as a piece of software, but also as an example of _documentation done right_. And kudos to the [Doxygen Awesome project](https://github.com/jothepro/doxygen-awesome-css) for their style sheets.
#  Documentation about Documentation
In this section, we describe common documentation maintenance tasks.
##  Including Plain HTML Pages
The most common use case for this is the inclusion of older documentation. New documentation should, whenever possible, be created using Doxygen markdown!
Use Doxygen's [`htmlinclude`](https://www.doxygen.nl/manual/commands.html#cmdhtmlinclude) special command to include existing plain HTML pages.
##  Creating a New Reference Manual Entry
Please refer to the [Reference Manual (RM) Page Template](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m_t.html) for guidance on how to create a new reference manual entry.
###  Adding and Referencing API Examples
For each HDF5 module, such as `H5F`, there is an examples source file called `H5*_examples.c`. For example, the `H5F` API examples are located in [`H5F_examples.c`](https://github.com/HDFGroup/hdf5/blob/develop/doxygen/examples/H5F_examples.c). Examples are code blocks marked as Doxygen [snippets](https://www.doxygen.nl/manual/commands.html#cmdsnippet). For example, the source code for the [H5Fcreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.") API sample is located between the 
```
//! <!-- [create] -->
...
//! <!-- [create] -->

```

comments in [`H5F_examples.c`](https://github.com/HDFGroup/hdf5/blob/develop/doxygen/examples/H5F_examples.c).
Add a new API example by adding a new code block enclosed between matching snippet tags. The name of the tag is usually the function name stripped of the module prefix.
The inclusion of such a block of code can then be triggered via Doxygen's [`snippet`](https://www.doxygen.nl/manual/commands.html#cmdsnippet) special command. For example, the following markup 
```
* \snippet H5F_examples.c create

```

yields 
{
__label__ fail_fapl, fail_fcpl, fail_file;
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) fcpl, fapl, file;
if ((fcpl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_CREATE](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a206f334f1e6c973e1215a3148b45b977))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_fcpl;
}
else {
// adjust the file creation properties
}
if ((fapl = [H5Pcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#gaf1b11da01d4d45d788c45f8bc5f0cbfa)([H5P_FILE_ACCESS](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#a60ec2d4334addfc0eda89614598ee38e))) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_fapl;
}
else {
// adjust the file access properties
}
unsigned mode = [H5F_ACC_EXCL](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a7a47250dc1435705233dca7297ba3d90);
char name[] = "f1.h5";
if ((file = [H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4)(name, mode, fcpl, fapl)) == [H5I_INVALID_HID](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a01eab13dccc91afd6909d74dccb780ba)) {
ret_val = EXIT_FAILURE;
goto fail_file;
}
// do something useful with FILE
[H5Fclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gac55cd91d80822e4f8c2a7f04ea71b124)(file);
fail_file:
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fapl);
fail_fapl:
[H5Pclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_l_c_r.html#ga5dce61149211d3ef319452aa598887fb)(fcpl);
fail_fcpl:;
}
###  Adding an API Macro
API macros are handled by the `api_vers_2, api_vers_3, api_vers_4` custom commands. The numbers indicate the number of potential API function mappings. For example, [H5Acreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24) has two potential mappings, [H5Acreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaa30f5f6c277d6c46f8aa31e89cdba085 "Creates an attribute attached to a specified object.") and [H5Acreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308 "Creates an attribute attached to a specified object."). To trigger the creation of a reference manual entry for [H5Acreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24) use the following markup: 
```
\api_vers_2{H5Acreate,H5Acreate1,H5Acreate2}

```

This yields:
[H5Acreate()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4a76e4e5ab6eb0fd2aa7990d38d55f24) is a macro that is mapped to either [H5Acreate1()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#gaa30f5f6c277d6c46f8aa31e89cdba085 "Creates an attribute attached to a specified object.") or [H5Acreate2()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_a.html#ga4f4e5248c09f689633079ed8afc0b308 "Creates an attribute attached to a specified object.").  


See also
    [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html)
##  Creating Custom Commands
See Doxygen's [Custom Commands documentation](https://www.doxygen.nl/manual/custcmd.html) as a general reference.
All custom commands for this project are located in the [`aliases`](https://github.com/HDFGroup/hdf5/blob/develop/doxygen/aliases) file in the [`doxygen`](https://github.com/HDFGroup/hdf5/blob/develop/doxygen) subdirectory of the [main HDF5 repo](https://github.com/HDFGroup/hdf5).
The custom commands are grouped in sections. Find a suitable section for your command or ask for help if unsure!
##  Adding a New RFC or Referencing an Existing RFC
For ease of reference, we define custom commands for each RFC in the `RFCs` section of the [`aliases`](https://github.com/HDFGroup/hdf5/blob/develop/doxygen/aliases) file. For example the custom command `ref_rfc20141210` can be used to insert a reference to "RFC: Virtual Object Layer". In other words, the markup 
```
\ref_rfc20141210

```

yields a clickable link:
[HDF5 Virtual Dataset](https://support.hdfgroup.org/releases/hdf5/documentation/rfc/HDF5-VDS-requirements-use-cases-2014-12-10.pdf)
To add a new RFC, add a custom command for the RFC to the [`aliases`](https://github.com/HDFGroup/hdf5/blob/develop/doxygen/aliases) file. The naming convention for the custom command is `ref_rfcYYYYMMDD`, where `YYYYMMDD` is the ID of the RFC. The URL is composed of the prefix 
```
https://\RFCURL/

```

and the name of your RFC file, typically, a PDF file, i.e., the full URL would be 
```
https://\RFCURL/my_great_rfc_name.pdf

```

* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


