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
  * [Detailed Description](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#title0 "H2")
  * [ Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#title1 "H2")
  * [Field Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#title2 "H2")
  * [◆ authenticate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#title3 "H2")
  * [◆ aws_region](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#title4 "H2")
  * [◆ secret_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#title5 "H2")
  * [◆ secret_key](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#title6 "H2")
  * [◆ version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Fields](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#pub-attribs)
H5FD_ros3_fapl_t Struct Reference
`#include <src/H5FDros3.h>`
## Detailed Description
Configuration structure for [H5Pset_fapl_ros3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.") / [H5Pget_fapl_ros3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e "Queries a File Access Property List for H5FD_ROS3 file driver properties."). 
[H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html "Configuration structure for H5Pset_fapl_ros3\(\) / H5Pget_fapl_ros3\(\).") is a public structure that is used to pass configuration data to the [H5FD_ROS3](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#ace004a4771dcfd40a4c0adecc1974570) driver via a File Access Property List. A pointer to an instance of this structure is a parameter to [H5Pset_fapl_ros3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.") and [H5Pget_fapl_ros3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e "Queries a File Access Property List for H5FD_ROS3 file driver properties."). 
##  Data Fields  
---  
bool  | [authenticate](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#ad3dab4b20a1b92a3ddae467c54f381d3)  
char  |  [aws_region](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a7378f9932dad7faf06da15d5d47a664e) [32+1]  
char  |  [secret_id](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a30a5353319d0a23ebc0b347aff48544a) [128+1]  
char  |  [secret_key](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#ab622227877522d00a1284b9e5465d02c) [128+1]  
int32_t  | [version](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a67fae7dd1de9edce3656ed214d20377f)  
## Field Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#ad3dab4b20a1b92a3ddae467c54f381d3)authenticate
bool authenticate  
---  
A Boolean which specifies if security credentials from this structure should be used for accessing a S3 bucket.
If true, ALL credentials must come from the FAPL and no attempt will be made to load credentials from other places. In this case, both `secret_id` and `secret_key` must be non-empty strings. If only one of `secret_id` or `secret_key` are non-empty strings while the other is an empty string, an error will be returned when opening a file. If a session token is to be used in this case, it must be specified with [H5Pset_fapl_ros3_token()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga74d2ae62171700a1a50c9fd76713cf4d "Modifies the specified File Access Property List to use the H5FD_ROS3 driver by adding the specified ...").
If false, the ROS3 VFD will instead attempt to load credentials from several different places, in this order:
  * From the environment, by checking AWS environment variables such as AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN and AWS_ACCOUNT_ID
  * From the AWS profile files, by reading from ~/.aws/config and ~/.aws/credentials, by default. The specific files read from can be overridden with the AWS_CONFIG_FILE and AWS_SHARED_CREDENTIALS_FILE environment variables.
  * From STS, by using AssumeRoleWithWebIdentity
  * From EC2 instance metadata


If the ROS3 VFD cannot source credentials from any of these locations, it will fallback to using anonymous credentials. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a7378f9932dad7faf06da15d5d47a664e)aws_region
char aws_region[32+1]  
---  
A string which specifies the AWS region of the S3 bucket, e.g. "us-east-1". Specifying an AWS region is always required for the ROS3 VFD, though it does not need to be specified here in the FAPL. The ROS3 VFD looks for the AWS region in the following places, in order:
  * The FAPL, if `aws_region` is not an empty string
  * The AWS_REGION environment variable
  * The AWS_DEFAULT_REGION environment variable
  * The AWS configuration file (~/.aws/config by default)
    * The 'default' profile from this file is used, unless a different profile is specified with the AWS_PROFILE environment variable


If the ROS3 VFD cannot determine an AWS region from one of these locations, an error will be returned when opening a file. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a30a5353319d0a23ebc0b347aff48544a)secret_id
char secret_id[128+1]  
---  
A string which specifies the security ID. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#ab622227877522d00a1284b9e5465d02c)secret_key
char secret_key[128+1]  
---  
A string which specifies the security key. 
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html#a67fae7dd1de9edce3656ed214d20377f)version
int32_t version  
---  
Version number of the [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html "Configuration structure for H5Pset_fapl_ros3\(\) / H5Pget_fapl_ros3\(\).") structure. Any instance passed to [H5Pset_fapl_ros3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga9dd7950acea860b716831488dd87ae8f "Modifies the specified File Access Property List to use the H5FD_ROS3 driver.") / [H5Pget_fapl_ros3()](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gaee5e0b8373b4b493b0cd2616938d601e "Queries a File Access Property List for H5FD_ROS3 file driver properties.") must have a recognized version number or an error will be raised. Currently, this field should be set to [H5FD_CURR_ROS3_FAPL_T_VERSION](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html#a0370e0bade82fd8ff6ac012644599fe2). 
* * *
The documentation for this struct was generated from the following file:
  * src/[H5FDros3.h](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_dros3_8h.html)


  * [H5FD_ros3_fapl_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_f_d__ros3__fapl__t.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


