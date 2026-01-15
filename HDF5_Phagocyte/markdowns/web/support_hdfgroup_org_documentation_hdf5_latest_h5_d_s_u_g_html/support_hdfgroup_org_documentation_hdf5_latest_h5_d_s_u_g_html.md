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
  * [ HDF5 Standard for Dimension Scales](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title0 "H1")
  * [ Conceptual model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title1 "H1")
  * [ Definitions](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title2 "H2")
  * [ Entity Relationship Diagrams](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title3 "H2")
  * [ What types of scales should be implemented?](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title4 "H2")
  * [ Limitations of this Specification](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title5 "H2")
  * [ The HDF5 Dimension Scale Specification](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title6 "H1")
  * [ Brief Summary](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title7 "H2")
  * [ Storage Profile](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title8 "H2")
  * [ Dimension Scale Names and Labels](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title9 "H2")
  * [ Shared Dimension Scales](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title10 "H2")
  * [ Example](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title11 "H2")
  * [ Programming Model and API](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title12 "H1")
  * [ Programming Model](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title13 "H2")
  * [ Dimension Scale Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#title14 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 High Level Dimension Scales
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 Standard for Dimension Scales
Dimension scales are stored as datasets, with additional metadata indicating that they are to be treated as dimension scales. Each dimension scale has an optional name. There is no requirement as to where dimension scales should be stored in the file. Dimension Scale names are not required to be unique within a file. (The name of a dimension scale does not have to be the same as the HDF5 path name for the dataset representing the scale.)
Datasets are linked to dimension scales. Each dimension of a Dataset may optionally have one or more associated Dimension Scales, as well as a label for the dimension. A Dimension Scale can be shared by two or more dimensions, including dimensions in the same or different dataset. Relationships between dataset dimensions and their corresponding dimension scales are not be directly maintained or enforced by the HDF5 library. For instance, a dimension scale would not be automatically deleted when all datasets that refer to it are deleted.
Functions for creating and using Dimension Scales are implemented as high level functions, see [HDF5 Dimension Scales APIs (H5DS)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html).
A frequently requested feature is for Dimension Scales to be represented as functions, rather than a stored array of precomputed values. To meet this requirement, it is recommended that the dataset model be expanded in the future to allow datasets to be represented by a generating function.
#  Conceptual model
Our study of dimension scale use cases has revealed an enormous variety of ways that dimension scales can be used. We recognize the importance of having a model that will be easy to understand and use for the vast majority of applications. It is our sense that those applications will need either no scale, a single 1-D array of floats or integers, or a simple function that provides a scale and offset.
At the same time, we want to place as few restrictions as possible on other uses of dimension scales. For instance, we don’t want to require dimension scales to be 1-D arrays, or to allow only one scale per dimension.
So our goal is to provide a model that serves the needs of two communities. We want to keep the dimension scale model conceptually simple for the majority of applications, but also to place as few restrictions as possible how dimension scales are interpreted and used. With this approach, it becomes the responsibility of applications to make sure that dimension scales satisfy the constraints of the model that they are assuming, such as constraints on the size of dimension scales and valid range of indices.
##  Definitions
Dimension Scales are implemented as an extension of these objects. In the HDF5 Abstract Data Model, a Dataset has a Dataspace, which defines a multi dimensional array of elements. Conceptually, a Dataspace has N dimension objects, which define the current and maximum size of the array in that dimension.
It is important to emphasize that the Dataspace of a Dataset has no intrinsic meaning except to define the layout in computer storage. Dimension Scales may be used to store application specific labels to the positions in the stored data array, i.e., to add application specific meaning to the dimensions of the dataspace.
A Dimension Scale is an object associated with one dimension of a Dataspace. The meaning of the association is left to applications. The values of the Dimension Scale are set by the application to reflect semantics of the data, for example, to associate coordinates of a reference system with positions on the dimension.
In general, these associations define a mapping between values of a dimension index and values of the Dimension Scale dataset. A simple case is where the Dimension Scale s is a (one dimensional) sequence of labels for the dimension ix of Dataset d. In this case, Dimension Scale is an array indexed by the same index as in the dimension of the Dataspace. For example, for the Dimension Scale s, associated with dimension ix, the ith position of ix is associated with the value s[i], so s[i] is taken as a label for ix[i].
##  Entity Relationship Diagrams
Figure 1 shows UML to illustrate the relationship between a Dimension and a Dimension Scale object. Conceptually, each Dimension of a Dataspace may have zero or more Dimension Scales associated with it. In turn, a Dimension Scale object may be associated with zero or more Dimensions (in zero or more Dataspaces).
Figure 2 illustrates the abstract model for a Dimension Scale object. A Dimension Scale is represented as a sub-class of a Dataset: a Dimension Scale has all the properties of a Dataset, with some specializations. A Dimension Scale dataset has an attribute “CLASS” with the value “DIMENSION_SCALE”. (This is analogous to the Table, Image, and Palette objects.) The Dimension Scale dataset has other attributes, including an optional NAME and references to any associated Dataset, as discussed below.
When the Dimension Scale is associated with a dimension of a Dataset, the association is represented by attributes of the two datasets. In the Dataset, the DIMENSION_LIST is an array of object references to scales (Dimension Scale Datasets) (Figure 1), and in the Dimension Scale Dataset the REFERENCE_LIST is an array of object references to Datasets (Figure 2). 
![](https://support.hdfgroup.org/documentation/hdf5/latest/H5DS_fig1.png) Figure 1. The relationship between a Dimension and a Dimension Scale.  
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/H5DS_fig2.png) Figure 2. The definition of a Dimension Scale and its attributes.  
---  
##  What types of scales should be implemented?
There seems to be good agreement that the model should accommodate scales that consist of a stored 1-D list of values, certain simple functions, and “no scale.” This specification also includes scales that are higher dimensional arrays, as well.
The four types of scales are: 
  * **No scale.** Frequently no scale is needed, so it should not be required. In some of these cases, an axis label may still be needed, and should be available. In this case, the Datase * t defines a Dimension Scale label for a dimension, with no Dimension Scale dataset. 
  * **1-D array.** Both fixed length and extendable arrays should be available. The size of the Dimension Scale is not required by HDF5 to conform to the size of the corresponding dimension(s), so that the number of scale values could be less than, equal to, or greater than the corresponding dimension size. 
  * **Simple function.** At a minimum, a linear scale of the form A + Bx should be available. This will be discussed in a future proposal. 
  * **Higher dimensional arrays.** This specification allows Dimension Scales to have any number of dimensions.


A number of use cases have been proposed in which more than one scale is needed for a given dimension. This specification places no restrictions on the number of scales that can be associated with a dimension, nor on the number or identities of Dimensions that may share the same Dimension Scale.
There are use cases for storing many types of data in a scale, including, but not limited to integers, floats, and strings. Therefore, this specification places no restrictions on the datatypes of scale values: a Dimension Scale can have any HDF5 Datatype. The interpretation of dimension scale values is left to applications.
##  Limitations of this Specification
One-to-many mapping. When there are fewer values in a dimension scale than in the corresponding dimension, it is useful to have a mapping between the two. For example, mappings are used by HDF-EOS to map geolocation information to dimension, in order that, for example, every second point may have geolocation. On the other hand, the way that mappings are defined can be very idiosyncratic, and it would seem to be challenging to provide a mapping model that satisfied a large number of cases. These mappings are not included in the model specified here.
Visibility and Integrity. Since Dimension Scales are a specialization of a Dataset, it is “visible” and accessible as a regular Dataset through the HDF5 API. This means that an application program could alter the values of or delete a Dimension Scale object or required attributes without regard to any of the semantics defined in this document.
One advantage it that the implementation requires no changes to the base library, which reduces the complexity of the code and the risk of side-effects. The implementation builds on existing functions, which should improve the quality and reliability of the code. Also, this approach leaves most of the semantics of dimension scales to applications and communities, who can use the specification in any way they need.
An important disadvantage is that the core HDF5 library will not manage the semantics of Dimension Scales. In particular, applications or other software must implement: 
  * **Naming** – the HDF5 library will impose no rules on the names of Dimension Scales 
  * **Consistency of references** – e.g., if a Dataset (Dimension Scale) is deleted (e.g., with [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group."), any Dimension Scales (Datasets) that it refers to (refer to it) will not be updated by the HDF5 library. 
  * **Consistency of extents** – the HDF5 library will not assure that a Dimension and associated Dimension Scale have the same extent (number of elements), nor that shared objects are consistent with each other. As in the case of delete, if a Dimension or Dimension Scale is extended (e.g., H5S…), any associated objects will not be automatically extended.


These are briefly summarized here. 
  * **Naming and Name Spaces.** There are many potential schemes for naming dimensions, each suited for different uses. This specification does not impose any specific approach, so it may be used by different applications. However, the lack of restrictions has disadvantages as well.  
For some purposes, it will be important to iterate through all the Dimension Scale objects in a file. The iterate operation is difficult to implement with the design specified here. This will be left to other software. For example, the HDF-EOS library has its own mechanism for managing a set of dimensions, and the netCDF4 library will implement this if it needs to. 
  * **Automatically extending dataset dimensions.** When a dimension of a dataset is extended, should the library automatically extend the corresponding dimension scale, or should this be left to the application? Since a dimension scale can be shared among many datasets, this raises a number of issues that are difficult to address in a general way. For instance, which dimension scale should be extended when only one dataset is extended, and what values are to be added? We have seen no compelling reason to implement an automatic extension of dimension scales when dataset dimensions are extended, so we suggest letting applications be responsible for this operation. 
  * **Automatically deleting dimension scales.** Should a dimension scale be deleted when all datasets that use it have been deleted? This is another case where different applications might have different requirements, so a general policy would be difficult to devise. Furthermore, enforcing a deletion policy, even a simple one, adds complexity to the library, and could also affect performance. Deletion policies seem best left to applications.


Section [Programming Model and API](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html#sec_dim_scales_api) presents an API and programming model that implements some of these features. However, applications may ignore or bypass these APIs, to write or read the attributes directly.
#  The HDF5 Dimension Scale Specification
##  Brief Summary
A Dimension Scale is stored as an HDF5 Dataset. 
  * A Dimension Scale is an object that is associated with a dimension of a Dataset. 
  * A Dimension Scale can have at most one name. 
  * A Dimension Scale may be associated with zero, one, or many different dimensions in any number of Datasets. 
  * Unless otherwise specified, a Dimension Scale inherits the properties of an HDF5 Dataset. 
  * There are no restrictions on the size, shape, or datatype of a Dimension Scale.


A Dimension Scale can be associated with a dimension of an HDF5 dataset 
  * A dimension of a Dataset may have zero, one, or more Dimension Scales associated with it. 
  * Each scale is identified by an index value.


A dimension may have a label without a scale, and may have a scale with no label. 
  * The label need not be the same as the name of any associated Dimension Scales.


The implementation has two parts: 
  * A storage profile 
  * An API and programming model


##  Storage Profile
This section specifies the storage profile for Dimension Scale objects and the association between Dimensions and Dimension Scales.
This profile is compatible with an earlier netcdf prototype and the HDF4 to HDF5 Mapping. This profile is also compatible with the netCDF4 proposal. This profile may be used to augment the HDF-EOS5 profile.
See Appendix 2 for a discussion of how to store converted HDF4 objects. See Appendix 3 for a discussion of netCDF4 issues. See Appendix 4 for a discussion of HDF-EOS5.
###  Dimension Scale Dataset
A Dimension Scale dataset is stored as an HDF5 dataset. Table 1 summarizes the stored data, i.e., the values of the scale. There are no restrictions on the dataspace or datatype, or storage properties of the dataset.
The scale may have any HDF5 datatype, and does not have to be the same as the datatype of the Dataset(s) that use the scale. E.g., an integer dataset might have dimension scales that are string or float values.
The dataspace of the scale can be any rank and shape. A scale is not limited to one dimension, and is not restricted by the size of any dimension(s) associated with it. When a dimension is associated with a one dimensional scale, the scale may be a different size from the dimension. In this case, it is up to the application to interpret or resolve the difference. When a dimension is associated with a scale with a rank higher than 1, the interpretation of the association is up to the application.
The Dimension Scale dataset can use any storage properties (including fill values, filters, and storage layout), not limited by the properties of any datasets that refer to it. When the Dimension Scale is extendible, it must be chunked.
Table 2 defines the required and optional attributes of the Dimension Scale Dataset. The attribute REFERENCE_LIST is a list of (dataset, index) pairs. Each pair represents an association defined by ‘attach_scale’. These pairs are stored as an array of compound data. Table 3 defines this datatype.
The Dimension Scale Dataset has an attribute called SUB_CLASS. This string is intended to be used to document particular specializations of this profile, e.g., a Dimension Scale created by netCDF4.
Table 1. The properties of the Dimension Scale dataset Field  | Datatype  | Dataspace  | Storage Properties  | Notes   
---|---|---|---|---  
<data> | Any  | Any  | Any  | These are the values of the Dimension Scale.   
Table 2. Standard Attributes for a stored Dimension Scale dataset. Attribute Name  | Datatype and Dimensions  | Value  | Required / Optional  | Notes   
---|---|---|---|---  
CLASS  |  [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) length = 16  | “DIMENSION_SCALE”  | Required  | This attribute distinguishes the dataset as a Dimension scale object.  
This is set by [H5DSset_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaca0f37c36c4e1288d161d26daa3a41d3 "Convert dataset dsid to a dimension scale, with optional name, dimname.")  
NAME  |  [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) length = <user defined> | <user defined>   
The name does not have to be the same as the HDF5 path name for the dataset. The name does not have to be related to any labels. Several Dimension Scales may have the same name.  | Optional, (Maximum of 1)  | The user defined label of the Dimension Scale.  
This is set by [H5DSset_label](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga10e953a37cf817f491fe47acb1370d88 "Set label for the dimension idx of did to the value label.")  
REFERENCE_LIST  | Array of Dataset Reference Type (Compound Datatype), variable length.  | [ {dataset1, ind1 }, …] [,…] ….]  | Optional, required when scale is attached  | See Table 3. This is set by [H5DSattach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b "Attach dimension scale dsid to dimension idx of dataset did.").   
SUB_CLASS  |  [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) length = <profile defined> | “HDF4_DIMENSION”,  
“NC4_DIMENSION”,  | Optional, defined by other profiles  | This is used to indicate a specific profile was used.   
<Other attributes> |  |  | Optional  | For example, UNITS.   
Table 3. Dataset Reference Type. This is a pair, <dataset_ref, index>. This is created when the Dimension Scale is attached to a Dataset. Field  | Datatype  | Value  | Notes   
---|---|---|---  
DATASET  | Object Reference.  | Pointer to a Dataset that refers to the scale  | Set by [H5DSattach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b "Attach dimension scale dsid to dimension idx of dataset did.").  
Removed by [H5DSdetach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad4af4f5d5c880c5c93ca950cdf5a01fd "Detach dimension scale dsid from the dimension idx of dataset did.").   
INDEX  |  [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2) | Index of the dimension the dataset pointed to by DATASET  | Set by [H5DSattach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b "Attach dimension scale dsid to dimension idx of dataset did.").  
Removed by [H5DSdetach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad4af4f5d5c880c5c93ca950cdf5a01fd "Detach dimension scale dsid from the dimension idx of dataset did.").   
###  Attributes of a Dataset with a Dimension Scale
A Dataset may have zero or more Dimension Scales associated with its dataspace. When present, these associations are represented by two attributes of the Dataset. Table 4 defines these attributes.
The DIMENSION_LIST is a two dimensional array with one row for each dimension of the Dataset, and a variable number of entries in each row, one for each associated scale. This is stored as a one dimensional array, with the HDF5 Datatype variable length array of object references.
When a dimension has more than one scale, the order of the scales in the DIMENSION_LIST attribute is not defined. A given Dimension Scale should appear in the list only once. (I.E., the DIMENSION_LIST is a “set”.)
When a scale is shared by more than one dimension (of one or more Dataset), the order of the records in REFERENCE_LIST is not defined. The Dataset and Dimension should appear in the list only once.
Table 4. Standard Attributes of a Dataset with associated Dimension Scale. Attribute Name  | Datatype and Dimensions  | Value  | Required / Optional  | Notes   
---|---|---|---|---  
DIMENSION_LIST  | The HDF5 datatype is ARRAY of Variable Length [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1) with rank of the dataspace.  | [[{object__ref1, object__ref2, … object__refn}, …] […] ..]  | Optional, required if scales are attached  | Set by [H5DSattach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b "Attach dimension scale dsid to dimension idx of dataset did.").  
Entries removed by [H5DSdetach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad4af4f5d5c880c5c93ca950cdf5a01fd "Detach dimension scale dsid from the dimension idx of dataset did.").   
DIMENSION_LABELLIST  | The HDF5 datatype is ARRAY of [H5T_STRING](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_tpublic_8h.html#a071841985f647f69516dbe77d93167f2a2de5d7919fe54466b7cf6a6c0b4265fa) with rank of the dataspace.  | [ <Label1>, <Label2>, …, <Label3>]  | Optional, required for scales with a label  | Set by [H5DSset_label](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga10e953a37cf817f491fe47acb1370d88 "Set label for the dimension idx of did to the value label.").   
##  Dimension Scale Names and Labels
Dimension scales are often referred to by name, so Dimension Scales may have names. Since some applications do not wish to apply names to dimension scales, Dimension Scale names be optional. In addition, some applications will have a name but no associated data values for a dimension (i.e., just a label). To support this, each dimension may have a label, which may be but need not be the same as the name of an associated Dimension Scale.
**Dimension Scale Name.** Associated with the Dimension Scale object. A Dimension Scale may have no name, or one name.
**Dimension Label.** A optional label associated with a dimension of a Dataset.
**How is a name represented?** Three options seem reasonable: 
  * Last link in the pathname. The h4toh5 mapping uses this approach [6], but there could be more than one path to a dataset, leading to ambiguities. This could be overcome by enforcing conventions. 
  * Attribute. This exposes this information at the top level, making it accessible to any viewer that can see attributes. It also makes it easy for applications to change the name, which could be dangerous, or valuable. 
  * Header message. This approach makes the name a little less available at the top level, but firmly pushes the concept into the base format and library. Since it also requires applications to change the name through a function call, it leaves open the possibility that the form of the name could be altered later without requiring application codes to change. On the other hand, if we treat names this way, it means that the “name” attribute is being treated differently from the “class” attribute, which could be confusing.


Dimension Scale names are stored in attributes of the Dimension Scale or the Dataset that refers to a Dimension Scale.
**Should dimension scale names be unique among dimension scales within a file?** We have seen a number of cases in which applications need more than one dimension scale with the same name. We have also seen applications where the opposite is true: dimension scale names are assumed to be unique within a file. This specification leaves it to applications to enforce a policy of uniqueness when they need it.
**Can a dimension have a label, without having an associated scale?** Some applications may wish to name dimensions without having an associated scale. Therefore, a dataset may have a label for a dimension without having an associated Dimension Scale dataset.
**Can a dimension have a scale, without having an associated label?** Some applications may wish to assign a dimension scale with no label. Therefore, a dataset may have one or more Dimension Scales for a dimension without having an associated label.
**Anonymous Dimensions.** It is possible to have a Dimension Scale dataset with no name, and associate it with a dimension of a dataset with no label. This case associates an array of data values to the dimension, but no identifier.
**A dimension with a label and a name.** A dimension of a dataset can be associated with a Dimension Scale that has a name, and assigned a label. In this case, the association has two “names”, the label and the dimension scale name. It is up to applications to interpret these names.
Table 5 summarizes the six possible combinations of label and name.
Table 5. Labels and scales of a dimension. | No scale  | Scale with no name  | Scale with name   
---|---|---|---  
No label  | Dimension has no label or scale (default)  | Dimension has an anonymous scale  | Dimension has scale, the scale is called “name”   
Label  | Dimension has label  | Dimension has scale with a label.  | Dimension has scale with both a label and name. A shared dimension has one name, but may have several labels   
##  Shared Dimension Scales
Given the design described above, datasets can share dimension scales. The following additional capabilities would seem to be useful. 
  * When a dimension scale is deleted, remove the reference to the dimension scale in all datasets that refer to it. 
  * Determine how many datasets are attached to a given dimension scale 
  * Determine what datasets are attached to a given dimension scale


These capabilities can be provided in several ways: 
  * Back pointers. If every dimension scale contained a list of back pointers to all datasets that referenced it, then it would be relatively easy to open all of these datasets and remove the references, as well as to answer questions #2 and #3. This would require the library to update the back pointer list every time a link was made. 
  * Alternatively, such lists could be maintained in a separate table. Such a table could contain all information about a number of dimension scales, which might provide a convenient way for applications to gain information about a set of dimension scales. For instance, this table might correspond to the coordinate variable definitions in a netCDF file. 
  * If no such list were available, an HDF5 function could be available to search all datasets to determine which ones referenced a given dimension scale. This would be straightforward, but in some cases could be very time consuming.


This specification defines attributes that maintain back pointers, which enable these kinds of cross referencing. Other software, such as NetCDF4, may well need a global table to track a set of dimensions. Such a table can be done in addition to the attributes defined here.
##  Example
This section presents an example to illustrate the data structures defined above.
Figure 3 shows a Dataset with a four dimensional Dataspace. The file also contains six Dimension Scale datasets. The Dimension Scale datasets are HDF5 objects, with path names such as “/DS1”.
Figure 4 illustrates the use of dimension scales in this example. Each Dimension Scale Dataset has an optional NAME. For example, “/DS3” has been assigned the name “Scale3”.
The dimensions of dataset D have been assigned zero or more scales and labels. Dimension 0 has two scales, Dimension 1 has one scale, and so on. Dimension 2 has no scale associated with it.
Some of the dimensions have labels as well. Note that dimension 2 has a label but no scale, and dimension 3 has scales but no label.
Some of the Dimension Scales are shared. Dimension Scale DS1 is referenced by dimension 0 of D and by another unspecified dataset. Dimension Scale DS3 is referenced by dimension 1 and 3 of Dataset D.
These relationships are represented in the file by attributes of the Dataset D and the Dimension Scale Datasets. Figure 5 shows the values that are stored for the DIMENSION_LIST attribute of Dataset D.
![](https://support.hdfgroup.org/documentation/hdf5/latest/UML_Attribute.jpg)
The UML model for an HDF5 attribute
This attribute is a one-dimensional array with the HDF5 datatype variable length [H5T_STD_REF_OBJ](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_s_t_d.html#gadc17e0960dc7ef08e029bf17201670e1). Each row of the array is zero or more object references for Dimension Scale datasets.
Table 6 shows the DIMENSION_LABELLIST for Dataset D. This is a one dimensional array with some empty values.
Each of the Dimension Scale Datasets has a name and other attributes. The references are represented in the REFERENCE_LIST attributes. Table 7 – Table 10 show the values for these tables. Note that Dimension Scale DS4 and DS6 have no references to them in this diagram.
The tables are stored as attributes of the Dimension Scale Dataset and the Datasets that refer to scales. Essentially, the association between a dimension of a Dataset and a Dimension Scale is represented by “pointers” (i.e., HDF5 Object References) in both of the associated objects. Since there can be multiple associations, there can be multiple pointers stored at each object, representing the endpoints of the associations. These will be stored in tables, i.e., as an attribute with an array of values.
When dimension scales are attached or detached, the tables in the Dataset and the Dimension Scale must be updated. The arrays in the attributes can grow, and items can be deleted.
The associations are identified by the object reference and dimension which is stored in a back pointer and returned from an API. The detach function needs to be careful how it deletes an item from the table, because the entries at both ends of the association must be updated at the same time.
![](https://support.hdfgroup.org/documentation/hdf5/latest/H5DS_fig3.png)
Figure 3. Example dataset and scales.
![](https://support.hdfgroup.org/documentation/hdf5/latest/H5DS_fig4.png)
Figure 4. Example labels, names, and attached scales.
![](https://support.hdfgroup.org/documentation/hdf5/latest/H5DS_fig5.png)
Figure 5. The table of dimension references.
Table 6. The table of dimension labels. Dataset Dimension  | Label   
---|---  
0  | “LX”   
1  | “LZ”   
2  | “LQ”   
3  | “”   
Table 7. The reference list for DS1. Reference  | Dataset Reference Record   
---|---  
0  | {Object reference to Dataset D, 0}   
1  | {Object reference to other Dataset, ?}   
Table 8. Reference list for DS2 Reference  | Dataset Reference Record   
---|---  
0  | {Object reference to Dataset D, 0}   
Table 9. Reference list for DS3 Reference  | Dataset Reference Record   
---|---  
0  | {Object reference to Dataset D, 1}   
1  | {Object reference to Dataset D, 3}   
Table 10. Reference List for DS5 Reference  | Dataset Reference Record   
---|---  
0  | {Object reference to Dataset D, 3}   
#  Programming Model and API
##  Programming Model
Dimension Scales are HDF5 Datasets, so they may be created and accesses through any HDF5 API for datasets [10]. The HDF5 Dimension Scale API implements the specification defined in this document. The operations include: 
  * Convert dataset to scale (D) – convert dataset D to a dimension scale. D may be specified by an id or by a path name. 
  * Attach scale (D, S, i) – attach dimension scale S to the ith dimension of D. D and S may be specified by an id or path name 
  * Detach scale (D, i, scale) – detach scale from the ith dimension of D. 
  * Iterate through scales of (D, i) – get each scale attached. 
  * Get the number of scales of (D, i) 
  * Get the ith scale of D 
  * Set/Get name (S) – set/get the name about dimension scale S.


The API also defines operations for dimension labels: 
  * Set/Get label (D, i) – set/get the label for ith dimension of D.


###  Create new Dimension Scale with Initial Values
  1. Create dataset with for the Dimension Scale with H5Dcreate and other standard HDF5 calls.
  2. Initialize the values of the Dimension Scale with H5Dwrite and other calls.
  3. Convert the dataset to a Dimension Scale with H5DSmake_scale.
  4. Close the Dimension Scale when finished with H5Dclose.


###  Attach Dimension Scale to Dataset
  1. Create or open the Dataset, D, with H5Dopen, etc.
  2. Create or open the Dimension Scale dataset, S, with H5Dopen or as above.
  3. Attach the Dimension Scale S to dimension j of Dataset D with H5DSattach_scale
  4. When finished, close the Dimension Scale and Dataset with H5Dclose.


###  Read Dimension Scale values
  1. Open the Dataset D, with H5Dopen
  2. Get the number of dimensions of D with H5Dget_space.
  3. Iterate through the scales of dimension i, locate the target scale, S (e.g., by its name).
  4. Get the datatype, dataspace, etc. of S with H5Dget_space, H5Dget_type, H5Sget_ndims, etc.
  5. Read the values of S into memory with H5Dread, e.g. into dscalebuff.
  6. When finished, close the S and other objects with H5Dclose etc.
  7. When finished, close the Dataset D with H5Dclose.


###  Write or Update Dimension Scale values
  1. Open the Dimension Scale Dataset S with H5open
  2. Get the datatype, dataspace, etc. of S with H5Dget_space, H5Dget_type, H5Sget_ndims, etc.
  3. If needed, read the values of S into memory with H5Dread. Note, may read selected values using a selection.
  4. Write updated values to S with H5Dwrite. Note, may write selected values using a selection.
  5. When finished, close S and other objects with H5Dclose etc.


###  Create a label for a dimension
  1. Open the Dataset D with H5open
  2. Add write a label for dimension i of D, with H5DSset_label.
  3. When finished, close the Dimension Scale Dataset and other objects with H5Dclose etc.


###  Extending a Dimension with a Dimension Scale attached
When an extendible Dataset has Dimension Scales, it is necessary to coordinate when the dimensions change size.
  1. Open the Dataset to be extended, with H5Dopen.
  2. Extend the dimension(s) with H5Dextend
  3. Iterate through the scales of each extended dimension. For each scale a. Extend the scale to the new size of the dimension with H5Dextend b. Write new values to the extended scale with H5Dwrite, etc. c. Close the Dimension Scale Dataset with H5Dclose if necessary.
  4. When finished, close the Dataset with H5Dclose


###  Detach Dimension Scale from Dataset
The detach operation removes an association between a dimension and a scale. It does not delete the Dimension Scale Dataset.
  1. Open the Dataset, D, with H5Dopen.
  2. Iterate through the scales of dimension i, locate the target scale, S (e.g., by its name).
  3. Detach the Dimension Scale S to dimension j of Dataset D with H5DSdetach_scale
  4. When finished, close the Dimension Scale and Dataset with H5Dclose.


###  Delete a Dimension Scale Dataset
When it is necessary to delete a Dimension Scale Dataset, it is necessary to detach it from all dataset. This section outlines the necessary steps.
  1. Open the Dimension Scale to be deleted.
  2. Read the REFERENCE_LIST attribute into memory with H5Aread etc.
  3. For each entry in the list: a. Dereference the dataset reference b. Detach the scale with H5DSdetach_scale c. Close the dataset reference
  4. Delete the Dimension Scale Dataset


###  Clean up Dimension Scales when deleting a Dataset
When it is necessary to delete a dataset that has scales attached, it is necessary to delete all the scales before deleting the dataset. Here is a sketch of the steps.
  1. Open the Dataset to be deleted, with H5Dopen.
  2. Iterate through the scales of each dimension of D
  3. For each scale, detach the Dimension Scale S from dimension j of Dataset D with H5DSdetach_scale
  4. Delete the Dataset, with H5Gunlink.


##  Dimension Scale Operations
Attach dimension scales to datasets using [H5DSattach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga0560840eeab7366cfe0c07cc6b57fe4b "Attach dimension scale dsid to dimension idx of dataset did.") and detach with [H5DSdetach_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad4af4f5d5c880c5c93ca950cdf5a01fd "Detach dimension scale dsid from the dimension idx of dataset did."). Set and retrieve dimension scale names with [H5DSset_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaca0f37c36c4e1288d161d26daa3a41d3 "Convert dataset dsid to a dimension scale, with optional name, dimname.") and [H5DSget_scale_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga704bbb7eb2c9696f0f2c84e0267e4ffa "Retrieves name of scale did into buffer name."). Query if a dataset is a dimension scale with [H5DSis_scale](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gaf6c043d90b502ecf96783e8f50cadcd5 "Determines whether did is a Dimension Scale.") and check attachments with [H5DSis_attached](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#gad7bbd73a68b031f6ac4c305333b0c8b7 "Report if dimension scale dsid is currently attached to dimension idx of dataset did."). Iterate through scales with [H5DSiterate_scales](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d_s.html#ga7cb78620d05e7dcb81a9f53a8cc496f5 "Iterates the operation visitor through the scales attached to dimension dim.").
Previous Chapter [Direct Chunk Write Function](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_o__u_g.html#sec_hldo_direct_chunk) - Next Chapter [HDF5 Images](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_i_m__u_g.html#sec_hl_images)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


