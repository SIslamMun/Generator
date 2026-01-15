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
  * [ Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title0 "H2")
  * [ Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title1 "H2")
  * [ Functions](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title2 "H2")
  * [Typedef Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title3 "H2")
  * [◆ attr_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title4 "H2")
  * [◆ visit_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title5 "H2")
  * [Function Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title6 "H2")
  * [◆ f_Attribute_setId()](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title7 "H2")
  * [◆ f_DataSet_setId()](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title8 "H2")
  * [◆ f_DataSpace_setId()](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title9 "H2")
  * [◆ f_DataType_setId()](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title10 "H2")
  * [◆ f_PropList_setId()](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title11 "H2")
  * [◆ userAttrOpWrpr()](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title12 "H2")
  * [◆ userVisitOpWrpr()](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#title13 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
[Data Structures](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#nested-classes) | [Typedefs](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#typedef-members) | [Functions](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#func-members)
H5 Namespace Reference
##  Data Structures  
---  
class  | [AbstractDs](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_abstract_ds.html)  
|  [AbstractDs](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_abstract_ds.html "AbstractDs is an abstract base class, inherited by Attribute and DataSet.") is an abstract base class, inherited by [Attribute](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute.html "Class Attribute operates on HDF5 attributes.") and [DataSet](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html "Class DataSet operates on HDF5 datasets."). [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_abstract_ds.html#details)  
  
class  | [ArrayType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_array_type.html)  
| Class [ArrayType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_array_type.html "Class ArrayType inherits from DataType and provides wrappers for the HDF5's Array Datatypes.") inherits from [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") and provides wrappers for the HDF5's Array Datatypes. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_array_type.html#details)  
  
class  | [AtomType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_atom_type.html)  
|  [AtomType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_atom_type.html "AtomType is a base class, inherited by IntType, FloatType, StrType, and PredType.") is a base class, inherited by [IntType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_int_type.html "IntType is a derivative of a DataType and operates on HDF5 integer datatype."), [FloatType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_float_type.html "FloatType is a derivative of a DataType and operates on HDF5 floating point datatype."), [StrType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_str_type.html "StrType is a derivative of a DataType and operates on HDF5 string datatype."), and [PredType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_pred_type.html "Class PredType holds the definition of all the HDF5 predefined datatypes."). [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_atom_type.html#details)  
  
class  | [Attribute](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute.html)  
| Class [Attribute](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute.html "Class Attribute operates on HDF5 attributes.") operates on HDF5 attributes. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute.html#details)  
  
class  | [AttributeIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute_i_exception.html)  
class  | [CommonFG](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_common_f_g.html)  
|  _[CommonFG](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_common_f_g.html "CommonFG is an abstract base class of H5Group.")_ is an abstract base class of H5Group. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_common_f_g.html#details)  
  
class  | [CompType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_comp_type.html)  
|  [CompType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_comp_type.html "CompType is a derivative of a DataType and operates on HDF5 compound datatypes.") is a derivative of a [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") and operates on HDF5 compound datatypes. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_comp_type.html#details)  
  
class  | [DataSet](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html)  
| Class [DataSet](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html "Class DataSet operates on HDF5 datasets.") operates on HDF5 datasets. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html#details)  
  
class  | [DataSetIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set_i_exception.html)  
class  | [DataSpace](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_space.html)  
| Class [DataSpace](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_space.html "Class DataSpace inherits from IdComponent and provides wrappers for the HDF5's dataspaces.") inherits from [IdComponent](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_id_component.html "Class IdComponent provides wrappers of the C functions that operate on an HDF5 identifier.") and provides wrappers for the HDF5's dataspaces. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_space.html#details)  
  
class  | [DataSpaceIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_space_i_exception.html)  
class  | [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html)  
| Class [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") provides generic operations on HDF5 datatypes. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html#details)  
  
class  | [DataTypeIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type_i_exception.html)  
class  | [DSetAccPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_acc_prop_list.html)  
| Class [DSetAccPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_acc_prop_list.html "Class DSetAccPropList inherits from LinkAccPropList and provides wrappers for the HDF5 dataset access...") inherits from [LinkAccPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_link_acc_prop_list.html "Class LinkAccPropList inherits from PropList and provides wrappers for the HDF5 link access property ...") and provides wrappers for the HDF5 dataset access property functions. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_acc_prop_list.html#details)  
  
class  | [DSetCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_creat_prop_list.html)  
| Class [DSetCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_creat_prop_list.html "Class DSetCreatPropList inherits from ObjCreatPropList and provides wrappers for the HDF5 dataset cre...") inherits from [ObjCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_obj_creat_prop_list.html "Class ObjCreatPropList inherits from PropList and provides wrappers for the HDF5 object create proper...") and provides wrappers for the HDF5 dataset creation property functions. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_creat_prop_list.html#details)  
  
class  | [DSetMemXferPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_mem_xfer_prop_list.html)  
| Class [DSetCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_creat_prop_list.html "Class DSetCreatPropList inherits from ObjCreatPropList and provides wrappers for the HDF5 dataset cre...") inherits from [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html "Class PropList inherits from IdComponent and provides wrappers for the HDF5 generic property list.") and provides wrappers for the HDF5 dataset memory and transfer property list. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_d_set_mem_xfer_prop_list.html#details)  
  
class  | [EnumType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_enum_type.html)  
|  [EnumType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_enum_type.html "EnumType is a derivative of a DataType and operates on HDF5 enum datatypes.") is a derivative of a [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") and operates on HDF5 enum datatypes. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_enum_type.html#details)  
  
class  | [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html)  
|  [Exception](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html "Exception provides wrappers of HDF5 error handling functions.") provides wrappers of HDF5 error handling functions. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_exception.html#details)  
  
class  | [FileAccPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_file_acc_prop_list.html)  
| Class [FileAccPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_file_acc_prop_list.html "Class FileAccPropList inherits from PropList and provides wrappers for the HDF5 file access property ...") inherits from [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html "Class PropList inherits from IdComponent and provides wrappers for the HDF5 generic property list.") and provides wrappers for the HDF5 file access property list. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_file_acc_prop_list.html#details)  
  
class  | [FileCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_file_creat_prop_list.html)  
| Class [FileCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_file_creat_prop_list.html "Class FileCreatPropList inherits from PropList and provides wrappers for the HDF5 file create propert...") inherits from [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html "Class PropList inherits from IdComponent and provides wrappers for the HDF5 generic property list.") and provides wrappers for the HDF5 file create property list. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_file_creat_prop_list.html#details)  
  
class  | [FileIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_file_i_exception.html)  
class  | [FloatType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_float_type.html)  
|  [FloatType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_float_type.html "FloatType is a derivative of a DataType and operates on HDF5 floating point datatype.") is a derivative of a [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") and operates on HDF5 floating point datatype. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_float_type.html#details)  
  
class  | [Group](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_group.html)  
| Class [Group](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_group.html "Class Group represents an HDF5 group.") represents an HDF5 group. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_group.html#details)  
  
class  | [GroupIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_group_i_exception.html)  
class  | [H5File](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_file.html)  
| Class [H5File](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_file.html "Class H5File represents an HDF5 file and inherits from class Group as file is a root group.") represents an HDF5 file and inherits from class [Group](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_group.html "Class Group represents an HDF5 group.") as file is a root group. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_file.html#details)  
  
class  | [H5Library](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html)  
| Class [H5Library](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html "Class H5Library operates the HDF5 library globably.") operates the HDF5 library globably. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_library.html#details)  
  
class  | [H5Location](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_location.html)  
|  [H5Location](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_location.html "H5Location is an abstract base class, added in version 1.8.12.") is an abstract base class, added in version 1.8.12. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_location.html#details)  
  
class  | [H5Object](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_object.html)  
| Class [H5Object](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_object.html "Class H5Object is a bridge between H5Location and DataSet, DataType, and Group.") is a bridge between [H5Location](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_location.html "H5Location is an abstract base class, added in version 1.8.12.") and [DataSet](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html "Class DataSet operates on HDF5 datasets."), [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes."), and [Group](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_group.html "Class Group represents an HDF5 group."). [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_object.html#details)  
  
class  | [IdComponent](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_id_component.html)  
| Class [IdComponent](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_id_component.html "Class IdComponent provides wrappers of the C functions that operate on an HDF5 identifier.") provides wrappers of the C functions that operate on an HDF5 identifier. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_id_component.html#details)  
  
class  | [IdComponentException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_id_component_exception.html)  
class  | [IntType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_int_type.html)  
|  [IntType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_int_type.html "IntType is a derivative of a DataType and operates on HDF5 integer datatype.") is a derivative of a [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") and operates on HDF5 integer datatype. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_int_type.html#details)  
  
class  | [LibraryIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_library_i_exception.html)  
class  | [LinkAccPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_link_acc_prop_list.html)  
| Class [LinkAccPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_link_acc_prop_list.html "Class LinkAccPropList inherits from PropList and provides wrappers for the HDF5 link access property ...") inherits from [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html "Class PropList inherits from IdComponent and provides wrappers for the HDF5 generic property list.") and provides wrappers for the HDF5 link access property list. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_link_acc_prop_list.html#details)  
  
class  | [LinkCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_link_creat_prop_list.html)  
| Class [LinkCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_link_creat_prop_list.html "Class LinkCreatPropList inherits from PropList and provides wrappers for the HDF5 link creation prope...") inherits from [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html "Class PropList inherits from IdComponent and provides wrappers for the HDF5 generic property list.") and provides wrappers for the HDF5 link creation property list. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_link_creat_prop_list.html#details)  
  
class  | [LocationException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_location_exception.html)  
class  | [ObjCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_obj_creat_prop_list.html)  
| Class [ObjCreatPropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_obj_creat_prop_list.html "Class ObjCreatPropList inherits from PropList and provides wrappers for the HDF5 object create proper...") inherits from [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html "Class PropList inherits from IdComponent and provides wrappers for the HDF5 generic property list.") and provides wrappers for the HDF5 object create property list. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_obj_creat_prop_list.html#details)  
  
class  | [ObjHeaderIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_obj_header_i_exception.html)  
class  | [PredType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_pred_type.html)  
| Class [PredType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_pred_type.html "Class PredType holds the definition of all the HDF5 predefined datatypes.") holds the definition of all the HDF5 predefined datatypes. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_pred_type.html#details)  
  
class  | [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html)  
| Class [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html "Class PropList inherits from IdComponent and provides wrappers for the HDF5 generic property list.") inherits from [IdComponent](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_id_component.html "Class IdComponent provides wrappers of the C functions that operate on an HDF5 identifier.") and provides wrappers for the HDF5 generic property list. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html#details)  
  
class  | [PropListIException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list_i_exception.html)  
class  | [ReferenceException](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_reference_exception.html)  
class  | [StrType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_str_type.html)  
|  [StrType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_str_type.html "StrType is a derivative of a DataType and operates on HDF5 string datatype.") is a derivative of a [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") and operates on HDF5 string datatype. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_str_type.html#details)  
  
class  | [UserData4Aiterate](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_user_data4_aiterate.html)  
class  | [UserData4Visit](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_user_data4_visit.html)  
class  | [VarLenType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_var_len_type.html)  
|  [VarLenType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_var_len_type.html "VarLenType is a derivative of a DataType and operates on HDF5 Variable-length Datatypes.") is a derivative of a [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html "Class DataType provides generic operations on HDF5 datatypes.") and operates on HDF5 Variable-length Datatypes. [More...](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_var_len_type.html#details)  
  
##  Typedefs  
---  
typedef void(*  |  [attr_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a652a9e72bbf8d3496fad8d0740f14c0a)) ([H5Object](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_object.html) &loc, const std::string attr_name, void *operator_data)  
typedef int(*  |  [visit_operator_t](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a5013def89685133d863a58525e27b606)) ([H5Object](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_object.html) &obj, const std::string attr_name, const [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *oinfo, void *operator_data)  
##  Functions  
---  
void  |  [f_Attribute_setId](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a0c8380480ed8047f97833abe159bd4dc) ([Attribute](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute.html) *attr, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_id)  
void  |  [f_DataSet_setId](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#ac7ff248608dd4f8680042d1aa6962fc6) ([DataSet](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html) *dset, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_id)  
void  |  [f_DataSpace_setId](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a0c67708838c1b3647849a9931ff2cee1) ([DataSpace](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_space.html) *dspace, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_id)  
void  |  [f_DataType_setId](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a65aad2c86b597a4b3c6b04811727455a) ([DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html) *dtype, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_id)  
void  |  [f_PropList_setId](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a6636793e5aaf921a442ced29ebeaf13f) ([PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html) *plist, [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) new_id)  
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [userAttrOpWrpr](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a8a716ba96d2f6eb5edeeda4b3ba48cd3) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) loc_id, const char *attr_name, const [H5A_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a__info__t.html) *ainfo, void *op_data)  
static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) |  [userVisitOpWrpr](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#ab1a01fe914dda17529d53f5ac6bee40a) ([hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) obj_id, const char *attr_name, const [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *obj_info, void *op_data)  
## Typedef Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a652a9e72bbf8d3496fad8d0740f14c0a)attr_operator_t
typedef void(* attr_operator_t) ([H5Object](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_object.html) &loc, const std::string attr_name, void *operator_data)  
---  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a5013def89685133d863a58525e27b606)visit_operator_t
typedef int(* visit_operator_t) ([H5Object](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_h5_object.html) &obj, const std::string attr_name, const [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) *oinfo, void *operator_data)  
---  
## Function Documentation
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a0c8380480ed8047f97833abe159bd4dc)f_Attribute_setId()
void f_Attribute_setId  | ( |  [Attribute](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_attribute.html) * |  _attr_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _new_id_ )  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#ac7ff248608dd4f8680042d1aa6962fc6)f_DataSet_setId()
void f_DataSet_setId  | ( |  [DataSet](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_set.html) * |  _dset_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _new_id_ )  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a0c67708838c1b3647849a9931ff2cee1)f_DataSpace_setId()
void f_DataSpace_setId  | ( |  [DataSpace](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_space.html) * |  _dspace_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _new_id_ )  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a65aad2c86b597a4b3c6b04811727455a)f_DataType_setId()
void f_DataType_setId  | ( |  [DataType](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_data_type.html) * |  _dtype_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _new_id_ )  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a6636793e5aaf921a442ced29ebeaf13f)f_PropList_setId()
void f_PropList_setId  | ( |  [PropList](https://support.hdfgroup.org/documentation/hdf5/latest/class_h5_1_1_prop_list.html) * |  _plist_ ,   
---|---|---|---  
|  | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _new_id_ )  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#a8a716ba96d2f6eb5edeeda4b3ba48cd3)userAttrOpWrpr()
| static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) userAttrOpWrpr  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _loc_id_ ,   
---|---|---|---  
|  | const char * |  _attr_name_ ,   
|  | const [H5A_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_a__info__t.html) * |  _ainfo_ ,   
|  | void * |  _op_data_ )  
static  
##  [◆ ](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html#ab1a01fe914dda17529d53f5ac6bee40a)userVisitOpWrpr()
| static [herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) userVisitOpWrpr  | ( | [hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) |  _obj_id_ ,   
---|---|---|---  
|  | const char * |  _attr_name_ ,   
|  | const [H5O_info2_t](https://support.hdfgroup.org/documentation/hdf5/latest/struct_h5_o__info2__t.html) * |  _obj_info_ ,   
|  | void * |  _op_data_ )  
static  
  * [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


