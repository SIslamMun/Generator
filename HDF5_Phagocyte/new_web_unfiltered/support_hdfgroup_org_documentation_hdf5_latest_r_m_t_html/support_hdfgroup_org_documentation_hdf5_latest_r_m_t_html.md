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


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Reference Manual (RM) Page Template
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html)
* * *
We treat documentation like code and use [Doxygen](https://www.doxygen.nl/index.html) to [markup comments in the code](https://github.com/HDFGroup/hdf5/blob/develop/src/H5Fpublic.h) or create [stand-alone pages](https://github.com/HDFGroup/hdf5/blob/develop/doxygen/dox/Overview.dox).
Every RM entry consists of a subset of the elements listed below. Not every RM entry warrants the full set. More is better, and we can, perhaps, distinguish minimal, typical, and great RM entries.
A minimal RM entry must include elements 1-3, 8, 11, and 7 if applicable.
A _typical_ RM entry is a minimal RM entry that in addition has elements 9, 10, and 12.
A **great** RM entry is a typical RM entry plus everything else.
The current RM is a mixed bag. Take what's there with a pinch of salt and apply the [Scout Rule](https://www.oreilly.com/library/view/97-things-every/9780596809515/ch08.html)! 

RM entry elements

  1. Module indication
     * Indicate the HDF5 module in which the function will appear. ```
     * \ingroup H5XYZ

```

  2. Synopsis
     * A phrase or sentence that summarizes the function's purpose ```
     * \brief Simplifies your life

```

  3. Prototype (parameters and return value)
     * A description of the function parameters and return value ```
     * \param[in]     name1 Description of IN parameter \p name1
     * \param[out]    name2 Description of OUT parameter \p name2
     * \param[in,out] name3 Description of INOUT parameter \p name3
     * \return Returns what you always wanted

```

     * Clearly indicate the parameter direction as `in`, `out`, or `in,out`
     * Make reference to other parameters using `\p`
  4. Preconditions
     * A set of preconditions that must be met. ```
     * \pre The argument supplied in parameter \p name2 must be even.

```

  5. Invariants
     * A set of invariants. ```
     * \invariant The mouse pointer will always be visible.

```

  6. Postconditions
     * What will be true when the function returns. ```
     * \post On error, the output parameters will be unmodified.

```

  7. Deprecation note
     * If a function was deprecated, list the version in which the function was deprecated (below), why it was deprecated, and which function(s) succeed it. ```
     * \deprecated Deprecated in favor of another great function.

```

  8. Details
     * A detailed description of the function's behavior ```
     * \details This is the heart of the matter. Try to be helpful!

```

  9. Example
     * The function in context and action, usually a (Doxygen) snippet. ```
     * \par Example
     * \snippet H5F_examples.c minimal

```

  10. Instruction (attention, note, warning)
     * Behaviors, features, side-effects, etc. the user should be aware of ```
     * \note  Dear reader, ...
     *
     * \attention Colorless green ideas sleep furiously.
     *
     * \warning Don't do this at home!

```

  11. Since
     * The HDF5 library version in which the function was introduced ```
     * \since 1.MAJOR.MINOR

```

  12. Version
     * Use this element to record a deprecation version, a change in parameter types, changes in behavior, etc. ```
     * \version 1.MAJOR.MINOR Function was deprecated in this release

```



* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [HDF5 Reference Manual](https://support.hdfgroup.org/documentation/hdf5/latest/_r_m.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


