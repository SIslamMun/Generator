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
  * [ HDF5 Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#title0 "H1")
  * [ Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#title1 "H2")
  * [ Description of the Group Object](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#title2 "H2")
  * [ Using h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#title3 "H2")
  * [ Group Function Summaries](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#title4 "H2")
  * [ Programming Model for Groups](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#title5 "H2")
  * [ Examples of File Structures](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#title6 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
HDF5 Groups
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
* * *
#  HDF5 Groups
##  Introduction
As suggested by the name Hierarchical Data Format, an HDF5 file is hierarchically structured. The HDF5 group and link objects implement this hierarchy.
In the simple and most common case, the file structure is a tree structure; in the general case, the file structure may be a directed graph with a designated entry point. The tree structure is very similar to the file system structures employed on UNIX systems, directories and files, and on Apple and Microsoft Windows systems, folders and files. HDF5 groups are analogous to the directories and folders; HDF5 datasets are analogous to the files.
The one very important difference between the HDF5 file structure and the above-mentioned file system analogs is that HDF5 groups are linked as a directed graph, allowing circular references; the file systems are strictly hierarchical, allowing no circular references. The figures below illustrate the range of possibilities.
In the first figure below, the group structure is strictly hierarchical, identical to the file system analogs.
In the next two figures below, the structure takes advantage of the directed graph's allowance of circular references. In the second figure, GroupA is not only a member of the root group, /, but a member of GroupC. Since Group C is a member of Group B and Group B is a member of Group A, Dataset1 can be accessed by means of the circular reference /Group A/Group B/Group C/Group A/Dataset1. The third figure below illustrates an extreme case in which GroupB is a member of itself, enabling a reference to a member dataset such as /Group A/Group B/Group B/Group B/Dataset2.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig1.gif) A file with a strictly hierarchical group structure  
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig2.gif) A file with a circular reference  
---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig3.gif) A file with one group as a member of itself  
---  
As becomes apparent upon reflection, directed graph structures can become quite complex; caution is advised!
The balance of this chapter discusses the following topics: 
  * The HDF5 group object (or a group) and its structure in more detail 
  * HDF5 link objects (or links) 
  * The programming model for working with groups and links 
  * HDF5 functions provided for working with groups, group members, and links 
  * Retrieving information about objects in a group 
  * Discovery of the structure of an HDF5 file and the contained objects 
  * Examples of file structures


##  Description of the Group Object
###  The Group Object
Abstractly, an HDF5 group contains zero or more objects and every object must be a member of at least one group. The root group, the sole exception, may not belong to any group.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig4.gif) Abstract model of the HDF5 group object  
---  
Group membership is actually implemented via link objects. See the figure above. A link object is owned by a group and points to a named object. Each link has a name, and each link points to exactly one object. Each named object has at least one and possibly many links to it.
There are three classes of named objects: group, dataset, and committed datatype (formerly called named datatype). See the figure below. Each of these objects is the member of at least one group, which means there is at least one link to it.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig5.gif) Classes of named objects  
---  
The primary operations on a group are to add and remove members and to discover member objects. These abstract operations, as listed in the figure below, are implemented in the [Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) APIs. For more information,  

See also
     [Group Function Summaries](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#subsec_group_function).
To add and delete members of a group, links from the group to existing objects in the file are created and deleted with the link and unlink operations. When a new named object is created, the HDF5 Library executes the link operation in the background immediately after creating the object (in other words, a new object is added as a member of the group in which it is created without further user intervention).
Given the name of an object, the get_object_info method retrieves a description of the object, including the number of references to it. The iterate method iterates through the members of the group, returning the name and type of each object.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig6.gif) The group object  
---  
Every HDF5 file has a single root group, with the name /. The root group is identical to any other HDF5 group, except: 
  * The root group is automatically created when the HDF5 file is created ([H5Fcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gae64b51ee9ac0781bc4ccc599d98387f4 "Creates an HDF5 file.")). 
  * The root group has no parent, but by convention has a reference count of 1. 
  * The root group cannot be deleted (in other words, unlinked)!


###  The Hierarchy of Data Objects
An HDF5 file is organized as a rooted, directed graph using HDF5 group objects. The named data objects are the nodes of the graph, and the links are the directed arcs. Each arc of the graph has a name, with the special name / reserved for the root group. New objects are created and then inserted into the graph with a link operation that is automatically executed by the library; existing objects are inserted into the graph with a link operation explicitly called by the user, which creates a named link from a group to the object.
An object can be the target of more than one link.
The names on the links must be unique within each group, but there may be many links with the same name in different groups. These are unambiguous, because some ancestor must have a different name, or else they are the same object. The graph is navigated with path names, analogous to Unix file systems. For more information,  

See also
     [HDF5 Path Names](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#subsubsec_group_descr_path).
An object can be opened with a full path starting at the root group, or with a relative path and a starting point. That starting point is always a group, though it may be the current working group, another specified group, or the root group of the file. Note that all paths are relative to a single HDF5 file. In this sense, an HDF5 file is analogous to a single UNIX file system.
It is important to note that, just like the UNIX file system, HDF5 objects do not have names, the names are associated with paths. An object has an object identifier that is unique within the file, but a single object may have many names because there may be many paths to the same object. An object can be renamed, or moved to another group, by adding and deleting links. In this case, the object itself never moves. For that matter, membership in a group has no implication for the physical location of the stored object.
Deleting a link to an object does not necessarily delete the object. The object remains available as long as there is at least one link to it. After all links to an object are deleted, it can no longer be opened, and the storage may be reclaimed.
It is also important to realize that the linking mechanism can be used to construct very complex graphs of objects. For example, it is possible for an object to be shared between several groups and even to have more than one name in the same group. It is also possible for a group to be a member of itself, or to create other cycles in the graph, such as in the case where a child group is linked to one of its ancestors.
HDF5 also has soft links similar to UNIX soft links. A soft link is an object that has a name and a path name for the target object. The soft link can be followed to open the target of the link just like a regular or hard link. The differences are that the hard link cannot be created if the target object does not exist and it always points to the same object. A soft link can be created with any path name, whether or not the object exists; it may or may not, therefore, be possible to follow a soft link. Furthermore, a soft link's target object may be changed.
###  HDF5 Path Names
The structure of the HDF5 file constitutes the name space for the objects in the file. A path name is a string of components separated by slashes (/). Each component is the name of a hard or soft link which points to an object in the file. The slash not only separates the components, but indicates their hierarchical relationship; the component indicated by the link name following a slash is a always a member of the component indicated by the link name preceding that slash.
The first component in the path name may be any of the following: 
  * The special character dot (., a single period), indicating the current group 
  * The special character slash (/), indicating the root group 
  * Any member of the current group


Component link names may be any string of ASCII characters not containing a slash or a single dot (/ and ., which are reserved as noted above). However, users are advised to avoid the use of punctuation and non-printing characters, as they may create problems for other software. The figure below provides a BNF grammar for HDF5 path names.
_A BNF grammar for HDF5 path names_
PathName ::= AbsolutePathName | RelativePathName
Separator ::= "/" ["/"]*
AbsolutePathName ::= Separator [ RelativePathName ]
RelativePathName ::= Component [ Separator RelativePathName ]*
Component ::= "." | Characters
Characters ::= Character+ - { "." }
Character ::= {c: c Î { { legal ASCII characters } - {'/'} }
An object can always be addressed by either a full or an absolute path name, starting at the root group, or by a relative path name, starting in a known location such as the current working group. As noted elsewhere, a given object may have multiple full and relative path names.
Consider, for example, the file illustrated in the figure below. Dataset1 can be identified by either of these absolute path names: _/GroupA/Dataset1_
_/GroupA/GroupB/GroupC/Dataset1_
Since an HDF5 file is a directed graph structure, and is therefore not limited to a strict tree structure, and since this illustrated file includes the sort of circular reference that a directed graph enables, Dataset1 can also be identified by this absolute path name: _/GroupA/GroupB/GroupC/GroupA/Dataset1_
Alternatively, if the current working location is GroupB, Dataset1 can be identified by either of these relative path names: _GroupC/Dataset1_
_GroupC/GroupA/Dataset1_
Note that relative path names in HDF5 do not employ the ../ notation, the UNIX notation indicating a parent directory, to indicate a parent group.
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig2.gif) A file with a circular reference  
---  
###  Group Implementations in HDF5
The original HDF5 group implementation provided a single indexed structure for link storage. A new group implementation, as of HDF5 Release 1.8.0, enables more efficient compact storage for very small groups, improved link indexing for large groups, and other advanced features. 
  * The original indexed format remains the default. Links are stored in a B-tree in the group's local heap. 
  * Groups created in the new compact-or-indexed format, the implementation introduced with Release 1.8.0, can be tuned for performance, switching between the compact and indexed formats at thresholds set in the user application. 
    * The compact format will conserve file space and processing overhead when working with small groups and is particularly valuable when a group contains no links. Links are stored as a list of messages in the group's header. 
    * The indexed format will yield improved performance when working with large groups. A large group may contain thousands to millions of members. Links are stored in a fractal heap and indexed with an improved B-tree. 
  * The new implementation also enables the use of link names consisting of non-ASCII character sets (see [H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.")) and is required for all link types other than hard or soft links; the link types other than hard or soft links are external links and user-defined links  

See also
     [Links (H5L)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html) APIs. 


The original group structure and the newer structures are not directly interoperable. By default, a group will be created in the original indexed format. An existing group can be changed to a compact-or-indexed format if the need arises; there is no capability to change back. As stated above, once in the compact-or-indexed format, a group can switch between compact and indexed as needed.
Groups will be initially created in the compact-or-indexed format only when one or more of the following conditions is met: 
  * The low version bound value of the library version bounds property has been set to Release 1.8.0 or later in the file access property list (see [H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.")). Currently, that would require an [H5Pset_libver_bounds](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gacbe1724e7f70cd17ed687417a1d2a910 "Controls the range of library release versions used when creating objects in a file.") call with the low parameter set to [H5F_LIBVER_LATEST](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_fpublic_8h.html#a2d963b599894f684571fbd4d5e8a96a2aa1212669916e7389d0a687a3673153b0).
When this property is set for an HDF5 file, all objects in the file will be created using the latest available format; no effort will be made to create a file that can be read by older libraries. 
  * The creation order tracking property, [H5P_CRT_ORDER_TRACKED](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#aa52f444ce2ba8bc5a062612f195e899f), has been set in the group creation property list (see [H5Pset_link_creation_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga24817b5c9553df3872de57c20bf11512 "Sets creation order tracking and indexing for links in a group.")). 


An existing group, currently in the original indexed format, will be converted to the compact-or- indexed format upon the occurrence of any of the following events: 
  * An external or user-defined link is inserted into the group. 
  * A link named with a string composed of non-ASCII characters is inserted into the group. 


The compact-or-indexed format offers performance improvements that will be most notable at the extremes (for example, in groups with zero members and in groups with tens of thousands of members). But measurable differences may sometimes appear at a threshold as low as eight group members. Since these performance thresholds and criteria differ from application to application, tunable settings are provided to govern the switch between the compact and indexed formats (see [H5Pset_link_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gab463ac9355728469eddfd973b4a5964f "Sets the parameters for conversion between compact and dense groups.")). Optimal thresholds will depend on the application and the operating environment.
Future versions of HDF5 will retain the ability to create, read, write, and manipulate all groups stored in either the original indexed format or the compact-or-indexed format.
##  Using h5dump
You can use [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump), the command-line utility distributed with HDF5, to examine a file for purposes either of determining where to create an object within an HDF5 file or to verify that you have created an object in the intended place.
In the case of the new group created later in this chapter, the following [h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) command will display the contents of FileA.h5: 
h5dump FileA.h5 
For more information,  

See also
     [Creating a Group](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#subsubsec_group_program_create).
Assuming that the discussed objects, GroupA and GroupB are the only objects that exist in FileA.h5, the output will look something like the following: 
HDF5 "FileA.h5" {
GROUP "/" {
GROUP GroupA {
GROUP GroupB {
}
}
}
}
[h5dump](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html#sec_cltools_h5dump) is described on the [Command Line Tools for HDF5 Files](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html#sec_cltools) page of the [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html).
The HDF5 DDL grammar is described in the [DDL in BNF for HDF5 2.0.0 and above](https://support.hdfgroup.org/documentation/hdf5/latest/_d_d_l_b_n_f200.html).
##  Group Function Summaries
Functions that can be used with groups ([Groups (H5G)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html) functions) and property list functions that can used with groups ([Property Lists (H5P)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_p.html) functions) are listed below. A number of group functions have been deprecated. Most of these have become link ([Links (H5L)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html)) or object ([Objects (H5O)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html)) functions. These replacement functions are also listed below.
Group functions Function  | Purpose   
---|---  
[H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) | Creates a new empty group and gives it a name. The C function is a macro:  

See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html).   
[H5Gcreate_anon](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gab52641f0736281faaaae4e3039bbb344 "Creates a new empty group without linking it into the file structure.") | Creates a new empty group without linking it into the file structure.   
[H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245) | Opens an existing group for modification and returns a group identifier for that group. The C function is a macro:  

See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html).   
[H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221 "Closes the specified group.") | Closes the specified group.   
[H5Gget_create_plist](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga0b959a53cbffa48f5d68ce33b43b7ed8 "Gets a group creation property list identifier.") | Gets a group creation property list identifier.   
[H5Gget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gad4be126ab7bbf2001435e8e70089f3d3 "Retrieves information about a group.") | Retrieves information about a group. Use instead of H5Gget_num_objs.   
[H5Gget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga985f27ad1a164d99fa1f58c6de60ab00 "Retrieves information about a group, according to the group's position within an index.") | Retrieves information about a group according to the group's position within an index.   
[H5Gget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#gadedd0c73c98f2ada69305f2992c3300e "Retrieves information about a group by its name.") | Retrieves information about a group.   
Link and object functions Function  | Purpose   
---|---  
[H5Lcreate_hard](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2 "Creates a hard link to an object.") | Creates a hard link to an object. Replaces H5Glink and H5Glink2.   
[H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link.") | Creates a soft link to an object. Replaces H5Glink and H5Glink2.   
[H5Lcreate_external](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.") | Creates a soft link to an object in a different file. Replaces H5Glink and H5Glink2.   
[H5Lcreate_ud](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#gadaf9732947c45cd4d2442e7f58873fc2 "Creates a link of a user-defined type.") | Creates a link of a user-defined type.   
[H5Lget_val](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga8eaacc372afc314e44521dfc1f66dcf4 "Returns the value of a link.") | Returns the value of a symbolic link. Replaces H5Gget_linkval.   
[H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) | Iterates through links in a group. Replaces H5Giterate. See also [H5Ovisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) and [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69).   
[H5Literate_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga655b002428e0176c2fa23a0315fbbcc2) | Iterates through links in a group.   
[H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) | Recursively visits all links starting from a specified group.   
[H5Ovisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) | Recursively visits all objects accessible from a specified object.   
[H5Lget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) | Returns information about a link. Replaces H5Gget_objinfo.   
[H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) | Retrieves the metadata for an object specified by an identifier. Replaces H5Gget_objinfo.   
[H5Lget_name_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga453ea40c3bb85ec8120dd17deed2bd90 "Retrieves name of the n-th link in a group, according to the order within a specified field or index.") | Retrieves name of the nth link in a group, according to the order within a specified field or index. Replaces H5Gget_objname_by_idx.   
[H5Oget_info_by_idx](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gafe764884e1530f86079288dd5895a7bd) | Retrieves the metadata for an object, identifying the object by an index position. Replaces H5Gget_objtype_by_idx.   
[H5Oget_info_by_name](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga16d8ac07f9244cfccb42b5f309ca6b3c) | Retrieves the metadata for an object, identifying the object by location and relative name.   
[H5Oset_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga8b5cf8e916204e29616516046121f631 "Sets comment for specified object.") | Sets the comment for specified object. Replaces H5Gset_comment.   
[H5Oget_comment](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaa1511ce5e2fe01ce7ea58f2f851d694b "Retrieves comment for specified object.") | Gets the comment for specified object. Replaces H5Gget_comment.   
[H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") | Removes a link from a group. Replaces H5Gunlink.   
[H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file.") | Renames a link within an HDF5 file. Replaces H5Gmove and H5Gmove2.   
Group creation property list functions (H5P) Function  | Purpose   
---|---  
[H5Pall_filters_avail](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#ga70f5346250698afc950532e9593c3988 "Verifies that all required filters are available.") | Verifies that all required filters are available.   
[H5Pget_filter](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7e070dfec9cb3a3aaf9c188a987e6a15) | Returns information about a filter in a pipeline. The C function is a macro:  

See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html).   
[H5Pget_filter_by_id](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#ac7aa336e7b1b9033cea2448ba623951f) | Returns information about the specified filter. The C function is a macro:  

See also
     [API Compatibility Macros](https://support.hdfgroup.org/documentation/hdf5/latest/api-compat-macros.html).   
[H5Pget_nfilters](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gacbad1ca36a61246b439a25f28e7575fb "Returns the number of filters in the pipeline.") | Returns the number of filters in the pipeline.   
[H5Pmodify_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga12a358b3725a889c1768bbd2b5f541d8 "Modifies a filter in the filter pipeline.") | Modifies a filter in the filter pipeline.   
[H5Premove_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gabffbf6d013c090fa052ac4bafce8e532 "Delete one or more filters in the filter pipeline.") | Deletes one or more filters in the filter pipeline.   
[H5Pset_deflate](https://support.hdfgroup.org/documentation/hdf5/latest/group___d_c_p_l.html#gaf1f569bfc54552bdb9317d2b63318a0d "Sets deflate \(GNU gzip\) compression method and compression level.") | Sets the deflate (GNU gzip) compression method and compression level.   
[H5Pset_filter](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#gae7ac4c110e9ec3fcf38966928b96cda3 "Adds a filter to the filter pipeline.") | Adds a filter to the filter pipeline.   
[H5Pset_fletcher32](https://support.hdfgroup.org/documentation/hdf5/latest/group___o_c_p_l.html#ga8bc81abfbd0393b0a46e121f817a3f81 "Sets up use of the Fletcher32 checksum filter.") | Sets up use of the Fletcher32 checksum filter.   
[H5Pset_local_heap_size_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga870728af2bf3c0b16edafd762a1c44d6 "Specifies the anticipated maximum size of a local heap.")/[H5Pget_local_heap_size_hint](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga49e14718767fa160248e3852c2abdd74 "Retrieves the anticipated size of the local heap for original-style groups.") | Sets/gets the anticipated maximum size of a local heap.   
[H5Pset_link_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gab463ac9355728469eddfd973b4a5964f "Sets the parameters for conversion between compact and dense groups.") | Sets the parameters for conversion between compact and dense groups.   
[H5Pget_link_phase_change](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gacab66461dca6c2beafd624c2e4d9f94d "Queries the settings for conversion between compact and dense groups.") | Queries the settings for conversion between compact and dense groups.   
[H5Pset_est_link_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa8571642d45e73ab5a9ae71cf00501f9 "Sets estimated number of links and length of link names in a group.") | Sets estimated number of links and length of link names in a group.   
[H5Pget_est_link_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga701867215546a345dea7b8e9cf7a1b61 "Returns the estimated link count and average link name length in a group.") | Queries data required to estimate required local heap or object header size.   
[H5Pset_nlinks](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#gaa46c63c196a0cf5cd94dede039c030f4 "Sets maximum number of soft or user-defined link traversals.") | Sets maximum number of soft or user-defined link traversals.   
[H5Pget_nlinks](https://support.hdfgroup.org/documentation/hdf5/latest/group___l_a_p_l.html#ga6bfa33fa9a77011cbdc06d0fbc907177 "Retrieves the maximum number of link traversals.") | Retrieves the maximum number of link traversals.   
[H5Pset_link_creation_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#ga24817b5c9553df3872de57c20bf11512 "Sets creation order tracking and indexing for links in a group.") | Sets creation order tracking and indexing for links in a group.   
[H5Pget_link_creation_order](https://support.hdfgroup.org/documentation/hdf5/latest/group___g_c_p_l.html#gaa2c2f433c7e65f694e0444e7f0ed2d33 "Queries whether link creation order is tracked and/or indexed in a group.") | Queries whether link creation order is tracked and/or indexed in a group.   
[H5Pset_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#gad4fa8e2d17236786f770cf17eef908cc "Sets the character encoding used to encode link and attribute names.") | Sets the character encoding used to encode a string. Use to set ASCII or UTF-8 character encoding for object names.   
[H5Pget_char_encoding](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_c_p_l.html#ga9b35ef9add6463997330e9b4b606603d "Retrieves the character encoding used to create a link or attribute name.") | Retrieves the character encoding used to create a string.   
Other external link functions Function  | Purpose   
---|---  
[H5Pset_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325 "Sets the number of files that can be held open in an external link open file cache.") | Sets the size of the external link open file cache from the specified file access property list.   
[H5Pget_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#ga4c9bcfff90f48bfefa2c25e551485923 "Retrieves the size of the external link open file cache.") | Retrieves the size of the external link open file cache from the specified file access property list.   
[H5Fclear_elink_file_cache](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gafcc153d8606829d4401e93305e5246d7 "Clears the external link open file cache.") | Clears the external link open file cache for a file.   
##  Programming Model for Groups
The programming model for working with groups is as follows: 
  1. Create a new group or open an existing one. 
  2. Perform the desired operations on the group. 
     * Create new objects in the group. 
     * Insert existing objects as group members. 
     * Delete existing members. 
     * Open and close member objects. 
     * Access information regarding member objects. 
     * Iterate across group members. 
     * Manipulate links.
  3. Terminate access to the group (Close the group).


###  Creating a Group
To create a group, use [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8), specifying the location and the path of the new group. The location is the identifier of the file or the group in a file with respect to which the new group is to be identified. The path is a string that provides either an absolute path or a relative path to the new group. For more information,  

See also
     [HDF5 Path Names](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_g__u_g.html#subsubsec_group_descr_path).
A path that begins with a slash (/) is an absolute path indicating that it locates the new group from the root group of the HDF5 file. A path that begins with any other character is a relative path. When the location is a file, a relative path is a path from that file's root group; when the location is a group, a relative path is a path from that group.
The sample code in the example below creates three groups. The group Data is created in the root directory; two groups are then created in /Data, one with absolute path, the other with a relative path.
_Creating three new groups_
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943) file;
file = [H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)(....);
group = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)(file, "/Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
group_new1 = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)(file, "/Data/Data_new1", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
group_new2 = [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)(group, "Data_new2", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[hid_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ipublic_8h.html#a0045db7ff9c22ad35db6ae91662e1943)
int64_t hid_t
**Definition** H5Ipublic.h:60
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f)
#define H5P_DEFAULT
**Definition** H5Ppublic.h:220
[H5Fopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#gaa3f4f877b9bb591f3880423ed2bf44bc)
hid_t H5Fopen(const char *filename, unsigned flags, hid_t fapl_id)
Opens an existing HDF5 file.
[H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8)
#define H5Gcreate
**Definition** H5version.h:1240
The third [H5Gcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga187cee27a9fc4f1a311eb19b0522c7b8) parameter optionally specifies how much file space to reserve to store the names that will appear in this group. If a non-positive value is supplied, a default size is chosen.
###  Opening a Group and Accessing an Object in that Group
Though it is not always necessary, it is often useful to explicitly open a group when working with objects in that group. Using the file created in the example above, the example below illustrates the use of a previously-acquired file identifier and a path relative to that file to open the group Data.
Any object in a group can be also accessed by its absolute or relative path. To open an object using a relative path, an application must first open the group or file on which that relative path is based. To open an object using an absolute path, the application can use any location identifier in the same file as the target object; the file identifier is commonly used, but object identifier for any object in that file will work. Both of these approaches are illustrated in the example below.
Using the file created in the examples above, the example below provides sample code illustrating the use of both relative and absolute paths to access an HDF5 data object. The first sequence (two function calls) uses a previously-acquired file identifier to open the group Data, and then uses the returned group identifier and a relative path to open the dataset CData. The second approach (one function call) uses the same previously-acquired file identifier and an absolute path to open the same dataset.
_Open a dataset with relative and absolute paths_
group = [H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)(file, "Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dataset1 = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(group, "CData", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dataset2 = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file, "/Data/CData", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)
#define H5Dopen
**Definition** H5version.h:1135
[H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)
#define H5Gopen
**Definition** H5version.h:1251
###  Creating a Dataset in a Specific Group
Any dataset must be created in a particular group. As with groups, a dataset may be created in a particular group by specifying its absolute path or a relative path. The example below illustrates both approaches to creating a dataset in the group /Data.
_Create a dataset with absolute and relative paths_
dataspace = [H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)([RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5), dims, NULL);
dataset1 = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(file, "/Data/CData", [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace,
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
group = [H5Gopen](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga3eca6807deff4f9e51fc5fe0befc2245)(file, "Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dataset2 = [H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)(group, "Cdata2", [H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2), dataspace,
[H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Dcreate](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html#ga0647ba4bbd26d5230cc07f3a5685b2cf)
#define H5Dcreate
**Definition** H5version.h:1124
[H5Screate_simple](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_s.html#ga8e35eea5738b4805856eac7d595254ae)
hid_t H5Screate_simple(int rank, const hsize_t dims[], const hsize_t maxdims[])
Creates a new simple dataspace and opens it for access.
[H5T_NATIVE_INT](https://support.hdfgroup.org/documentation/hdf5/latest/group___p_d_t_n_a_t.html#ga3cf93ffc6782be68070ef8e00f219ec2)
#define H5T_NATIVE_INT
**Definition** H5Tpublic.h:988
[RANK](https://support.hdfgroup.org/documentation/hdf5/latest/h5import_8h.html#a4e76a9ce86d91fa75775a7ae3f8d3aa5)
#define RANK
**Definition** h5import.h:343
###  Closing a Group
To ensure the integrity of HDF5 objects and to release system resources, an application should always call the appropriate close function when it is through working with an HDF5 object. In the case of groups, H5Gclose ends access to the group and releases any resources the HDF5 library has maintained in support of that access, including the group identifier.
As illustrated in the example below, all that is required for an H5Gclose call is the group identifier acquired when the group was opened; there are no relative versus absolute path considerations.
_Close a group_
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160) status;
status = [H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221)(group);
[herr_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5public_8h.html#a3b079ecf932a5c599499cf7e298af160)
int herr_t
**Definition** H5public.h:252
[H5Gclose](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_g.html#ga8dbe20b390d2504f0bd3589ed8f4e221)
herr_t H5Gclose(hid_t group_id)
Closes the specified group.
A non-negative return value indicates that the group was successfully closed and the resources released; a negative return value indicates that the attempt to close the group or release resources failed.
###  Creating Links
As previously mentioned, every object is created in a specific group. Once created, an object can be made a member of additional groups by means of links created with one of the H5Lcreate_* functions.
A link is, in effect, a path by which the target object can be accessed; it therefore has a name which functions as a single path component. A link can be removed with an [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") call, effectively removing the target object from the group that contained the link (assuming, of course, that the removed link was the only link to the target object in the group).
#### Hard Links
There are two kinds of links, hard links and symbolic links. Hard links are reference counted; symbolic links are not. When an object is created, a hard link is automatically created. An object can be deleted from the file by removing all the hard links to it.
Working with the file from the previous examples, the code in the example below illustrates the creation of a hard link, named Data_link, in the root group, /, to the group Data. Once that link is created, the dataset Cdata can be accessed via either of two absolute paths, /Data/Cdata or /Data_Link/Cdata.
_Create a hard link_
status = [H5Lcreate_hard](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2)(Data_loc_id, "Data", DataLink_loc_id, "Data_link", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dataset1 = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file, "/Data_link/CData", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dataset2 = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file, "/Data/CData", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Lcreate_hard](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga69d50f7acdfd2f1dc7c4372397e63bd2)
herr_t H5Lcreate_hard(hid_t cur_loc, const char *cur_name, hid_t dst_loc, const char *dst_name, hid_t lcpl_id, hid_t lapl_id)
Creates a hard link to an object.
The example below shows example code to delete a link, deleting the hard link Data from the root group. The group /Data and its members are still in the file, but they can no longer be accessed via a path using the component /Data.
_Delete a link_
status = [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf)(Data_loc_id, "Data", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dataset1 = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file, "/Data_link/CData", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// This call should succeed; all path components still exist
dataset2 = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file, "/Data/CData", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
// This call will fail; the path component '/Data' has been deleted.
[H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf)
herr_t H5Ldelete(hid_t loc_id, const char *name, hid_t lapl_id)
Removes a link from a group.
When the last hard link to an object is deleted, the object is no longer accessible. [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") will not prevent you from deleting the last link to an object. To see if an object has only one link, use the [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) function. If the value of the rc (reference count) field in the is greater than 1, then the link can be deleted without making the object inaccessible.
The example below shows [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) to the group originally called Data.
_Finding the number of links to an object_
status = [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5)(Data_loc_id, object_info);
[H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5)
#define H5Oget_info
**Definition** H5version.h:1351
It is possible to delete the last hard link to an object and not make the object inaccessible. Suppose your application opens a dataset, and then deletes the last hard link to the dataset. While the dataset is open, your application still has a connection to the dataset. If your application creates a hard link to the dataset before it closes the dataset, then the dataset will still be accessible.
#### Symbolic Links
Symbolic links are objects that assign a name in a group to a path. Notably, the target object is determined only when the symbolic link is accessed, and may, in fact, not exist. Symbolic links are not reference counted, so there may be zero, one, or more symbolic links to an object.
The major types of symbolic links are soft links and external links. Soft links are symbolic links within an HDF5 file and are created with the [H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78 "Creates a soft link.") function. Symbolic links to objects located in external files, in other words external links, can be created with the [H5Lcreate_external](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga15dfaeb9b1c0b3136533cb97ee45e683 "Creates an external link, a soft link to an object in a different file.") function. Symbolic links are removed with the [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") function.
The example below shows the creating two soft links to the group /Data.
_Create a soft link_
status = [H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78)(path_to_target, link_loc_id, "Soft2", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
status = [H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78)(path_to_target, link_loc_id, "Soft3", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f), [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
dataset = [H5Dopen](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a7dba2e5b2045f31c0932123ffb54f7a3)(file, "/Soft2/CData", [H5P_DEFAULT](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_ppublic_8h.html#afa85e97bfbf9bf1c58e39263846c568f));
[H5Lcreate_soft](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga894444623b58ce1ac3bd35538245ac78)
herr_t H5Lcreate_soft(const char *link_target, hid_t link_loc_id, const char *link_name, hid_t lcpl_id, hid_t lapl_id)
Creates a soft link.
With the soft links defined in the example above, the dataset CData in the group /Data can now be opened with any of the names /Data/CData, /Soft2/CData, or /Soft3/CData.
In release 1.8.7, a cache was added to hold the names of files accessed via external links. The size of this cache can be changed to help improve performance. For more information, see the entry in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html) for the [H5Pset_elink_file_cache_size](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html#gac21a815e9b133802df625c9f766ef325 "Sets the number of files that can be held open in an external link open file cache.") function call.
#### Note Regarding Hard Links and Soft Links
Note that an object's existence in a file is governed by the presence of at least one hard link to that object. If the last hard link to an object is removed, the object is removed from the file and any remaining soft link becomes a dangling link, a link whose target object does not exist.
#### Moving or Renaming Objects, and a Warning
An object can be renamed by changing the name of a link to it with [H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file."). This has the same effect as creating a new link with the new name and deleting the link with the old name.
Exercise caution in the use of [H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file.") and [H5Ldelete](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga5b4e7f59f5d4bdae94fd8ce6875295cf "Removes a link from a group.") as these functions each include a step that unlinks a pointer to an HDF5 object. If the link that is removed is on the only path leading to an HDF5 object, that object will become permanently inaccessible in the file.
##### Scenario 1: Removing the Last Link
To avoid removing the last link to an object or otherwise making an object inaccessible, use the [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) function. Make sure that the value of the reference count field (rc) is greater than 1.
##### Scenario 2: Moving a Link that Isolates an Object
Consider the following example: assume that the group group2 can only be accessed via the following path, where top_group is a member of the file's root group: _/top_group/group1/group2/_
Using [H5Lmove](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga0bbc7f9bf25c8aca9dd8433a325c8acb "Moves a link within an HDF5 file."), top_group is renamed to be a member ofgroup2. At this point, since top_group was the only route from the root group to group1, there is no longer a path by which one can access group1, group2, or any member datasets. And since top_group is now a member of group2, top_group itself and any member datasets have thereby also become inaccessible.
#### Mounting a File
An external link is a permanent connection between two files. A temporary connection can be set up with the [H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file.") function. For more information,  

See also
     [The HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#sec_file). For more information, see the [H5Fmount](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_f.html#ga9f5ca4c8695597e83568f9f4717fc083 "Mounts an HDF5 file.") function in the [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html).
###  Discovering Information about Objects
There is often a need to retrieve information about a particular object. The [H5Lget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_l.html#ga97279697f3010a6ad31dd7f4341eb698) and [H5Oget_info](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#gaf4f302a33faba9e1c2b5f64c62ca4ed5) functions fill this niche by returning a description of the object or link in an [H5L_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#af542682cfe46de7b0759e52a1608d1e4) or [H5O_info_t](https://support.hdfgroup.org/documentation/hdf5/latest/_h5version_8h.html#a5f76b0cdd6d68d61f11e46d4f06e50d4) structure.
###  Discovering Objects in a Group
To examine all the objects or links in a group, use the [H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) or [H5Ovisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) functions to examine the objects, and use the [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) function to examine the links. [H5Literate](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#ga55406698106930db68242987c11ba051) is useful both with a single group and in an iterative process that examines an entire file or section of a file (such as the contents of a group or the contents of all the groups that are members of that group) and acts on objects as they are encountered. [H5Ovisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_o.html#ga5ce86255fcc34ceaf84a62551cd24233) recursively visits all objects accessible from a specified object. [H5Lvisit](https://support.hdfgroup.org/documentation/hdf5/latest/group___t_r_a_v.html#gac0558936502924d9e898d8b6e041ed69) recursively visits all the links starting from a specified group.
###  Discovering All of the Objects in the File
The structure of an HDF5 file is self-describing, meaning that an application can navigate an HDF5 file to discover and understand all the objects it contains. This is an iterative process wherein the structure is traversed as a graph, starting at one node and recursively visiting linked nodes. To explore the entire file, the traversal should start at the root group.
##  Examples of File Structures
This section presents several samples of HDF5 file structures.
Figure 9 shows examples of the structure of a file with three groups and one dataset. The file in part a contains three groups: the root group and two member groups. In part b, the dataset dset1 has been created in /group1. In part c, a link named dset2 from /group2 to the dataset has been added. Note that there is only one copy of the dataset; there are two links to it and it can be accessed either as /group1/dset1 or as /group2/dset2.
Part d illustrates that one of the two links to the dataset can be deleted. In this case, the link from _/group1_ has been removed. The dataset itself has not been deleted; it is still in the file but can only be accessed as _/group2/dset2_
Figure 9 - Some file structures ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig9_a.gif) a) The file contains three groups: the root group, /group1, and /group2. |  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig9_b.gif) b) The dataset dset1 (or /group1/dset1) is created in /group1.  
---|---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig9_aa.gif) c) A link named dset2 to the same dataset is created in /group2. |  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig9_bb.gif) d) The link from /group1 to dset1 is removed. The dataset is still in the file, but can be accessed only as /group2/dset2.   
Figure 10 illustrates loops in an HDF5 file structure. The file in part a contains three groups and a dataset; group2 is a member of the root group and of the root group's other member group, group1. group2 thus can be accessed by either of two paths: /group2 or /group1/GXX. Similarly, the dataset can be accessed either as /group2/dset1 or as /group1/GXX/dset1.
Part b illustrates a different case: the dataset is a member of a single group but with two links, or names, in that group. In this case, the dataset again has two names, /group1/dset1 and /group1/dset2.
In part c, the dataset dset1 is a member of two groups, one of which can be accessed by either of two names. The dataset thus has three path names: /group1/dset1, /group2/dset2, and /group1/GXX/dset2.
And in part d, two of the groups are members of each other and the dataset is a member of both groups. In this case, there are an infinite number of paths to the dataset because GXX and GYY can be traversed any number of times on the way from the root group, /, to the dataset. This can yield a path name such as /group1/GXX/GYY/GXX/GYY/GXX/dset2.
Figure 10 - More sample file structures ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig10_a.gif) a) dset1 has two names: /group2/dset1 and /group1/GXX/dset1. |  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig10_b.gif) b) dset1 again has two names: /group1/dset1 and /group1/dset2.  
---|---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig10_c.gif) c) dset1 has 3 names: /group1/dset1, /group2/dset2, and /group1/GXX/dset2. |  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig10_d.gif) d) dset1 has an infinite number of available path names.  
Figure 11 takes us into the realm of soft links. The original file, in part a, contains only three hard links. In part b, a soft link named dset2 from group2 to /group1/dset1 has been created, making this dataset accessible as /group2/dset2.
In part c, another soft link has been created in group2. But this time the soft link, dset3, points to a target object that does not yet exist. That target object, dset, has been added in part d and is now accessible as either /group2/dset or /group2/dset3.
It could be said that HDF5 extends the organizing concepts of a file system to the internal structure of a single file.
Figure 11 - Hard and soft links ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig11_a.gif) a) The file contains only hard links. |  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig11_b.gif) b) A soft link is added from group2 to /group1/dset1.  
---|---  
![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig11_c.gif) c) A soft link named dset3 is added with a target that does not yet exist. |  ![](https://support.hdfgroup.org/documentation/hdf5/latest/Groups_fig11_d.gif) d) The target of the soft link is created or linked.  
Previous Chapter [The HDF5 File](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f__u_g.html#sec_file) - Next Chapter [HDF5 Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html#sec_dataset)
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 User Guide](https://support.hdfgroup.org/documentation/hdf5/latest/_u_g.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


