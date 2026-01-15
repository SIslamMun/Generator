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
  * [ Building HDF5 with CMake](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#title0 "H1")
  * [ CMake Presets Use Case: Static Library and Tools](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#title1 "H2")
  * [ CMake Presets Use Case: S3 on linux](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#title2 "H2")
  * [ CMake Presets Use Case: Parallel Library](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#title3 "H2")
  * [ CMake Presets Use Case: Java Bindings](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#title4 "H2")
  * [ CMake Presets](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#title5 "H1")
  * [ CMakePresets.json](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#title6 "H2")
  * [ Default CMakePresets.json Settings](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#title7 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Building with CMake Presets
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
* * *
#  Building HDF5 with CMake 

Problem
    You want to build HDF5 with CMake, but there are so many options to consider. 

Solution
    CMake introduced presets in version 3.19. HDF Group provides a [`CMakePresets.json`](https://github.com/HDFGroup/hdf5/blob/develop/CMakePresets.json), requiring CMake 3.26, that will build HDF5 with the options for building a typical shared library with the common languages for a platform. The features include building the tools, examples, plugins, and the shared and static libraries. 

Discussion
    The [`CMakePresets.json`](https://github.com/HDFGroup/hdf5/blob/develop/CMakePresets.json) file is located in the root directory of the HDF5 source. It is from here you will execute the cmake command to build HDF5. The following example shows how to build HDF5 with the [`CMakePresets.json`](https://github.com/HDFGroup/hdf5/blob/develop/CMakePresets.json) file: 
  * change directory to the [hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html) source folder 
  * execute `cmake –workflow –preset ci-StdShar-<compiler-type> –fresh`  
where "<compiler-type>" is GNUC or MSVC or Clang

The above example will create a "build" folder in the source parent directory, which will contain the results of the build, including installation package files.
The CMakePresets.json files created by HDF Group are intended to be used with the Ninja build system, which may need to be installed separately on some platforms. 

See Also
    See CMake documentation for details on presets: 
  * <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>


##  CMake Presets Use Case: Static Library and Tools 

Problem
    I want to build the HDF5 library statically and include the tools. 

Solution
    Static libraries can be built with zlib and libaec - but cannot be built with plugins. Java JNI bindings require shared libraries.
Create a CMakeUserPresets.json file with the following content: 
{
"version": 6,
"cmakeMinimumRequired": {
"major": 3,
"minor": 25,
"patch": 0
},
"configurePresets": [
{
"name": "my-Static-Tools",
"hidden": true,
"description": "Build HDF5 with static library and tools",
"inherits": ["ci-base", "ci-CompressionVars", "ci-StdExamples"],
"cacheVariables": {
"HDF_PACKAGE_NAMESPACE": {"type": "STRING", "value": "hdf5::"},
"HDF5_BUILD_SHARED_LIBS": "OFF",
"HDF5_BUILD_TOOLS": "ON",
"HDF5_ENABLE_ZLIB_SUPPORT": "ON",
"HDF5_ENABLE_LIBAEC": "ON",
"HDF5_ENABLE_PLUGIN_SUPPORT": "OFF"
}
},
{
"name": "my-Static-Tools-MSVC",
"description": "MSVC Standard Config for x64 (Release)",
"inherits": [
"ci-x64-Release-MSVC",
"ci-CPP",
"my-Static-Tools"
]
},
{
"name": "my-Static-Tools-Clang",
"description": "Clang Standard Config for x64 (Release)",
"inherits": [
"ci-x64-Release-Clang",
"ci-CPP",
"ci-Fortran",
"my-Static-Tools"
]
},
{
"name": "my-Static-Tools-GNUC",
"description": "GNUC Standard Config for x64 (Release)",
"inherits": [
"ci-x64-Release-GNUC",
"ci-CPP",
"ci-Fortran",
"my-Static-Tools"
]
},
{
"name": "my-Static-Tools-macos-GNUC",
"description": "GNUC Standard Config for macos (Release)",
"inherits": [
"ci-macos-Release-GNUC",
"ci-CPP",
"my-Static-Tools"
]
}
],
"buildPresets": [
{
"name": "my-Static-Tools-MSVC",
"description": "MSVC Standard Build for x64 (Release)",
"configurePreset": "my-Static-Tools-MSVC",
"inherits": [
"ci-x64-Release-MSVC"
]
},
{
"name": "my-Static-Tools-Clang",
"description": "Clang Standard Build for x64 (Release)",
"configurePreset": "my-Static-Tools-Clang",
"inherits": [
"ci-x64-Release-Clang"
]
},
{
"name": "my-Static-Tools-GNUC",
"description": "GNUC Standard Build for x64 (Release)",
"configurePreset": "my-Static-Tools-GNUC",
"inherits": [
"ci-x64-Release-GNUC"
]
},
{
"name": "my-Static-Tools-macos-GNUC",
"description": "GNUC Standard Build for macos (Release)",
"configurePreset": "my-Static-Tools-macos-GNUC",
"verbose": true,
"inherits": [
"ci-macos-Release-GNUC"
]
}
],
"testPresets": [
{
"name": "my-Static-Tools-MSVC",
"configurePreset": "my-Static-Tools-MSVC",
"inherits": [
"ci-x64-Release-MSVC"
]
},
{
"name": "my-Static-Tools-Clang",
"configurePreset": "my-Static-Tools-Clang",
"inherits": [
"ci-x64-Release-Clang"
]
},
{
"name": "my-Static-Tools-GNUC",
"configurePreset": "my-Static-Tools-GNUC",
"inherits": [
"ci-x64-Release-GNUC"
]
},
{
"name": "my-Static-Tools-macos-GNUC",
"configurePreset": "my-Static-Tools-macos-GNUC",
"inherits": [
"ci-macos-Release-GNUC"
]
}
],
"packagePresets": [
{
"name": "my-Static-Tools-MSVC",
"configurePreset": "my-Static-Tools-MSVC",
"inherits": "ci-x64-Release-MSVC"
},
{
"name": "my-Static-Tools-Clang",
"configurePreset": "my-Static-Tools-Clang",
"inherits": "ci-x64-Release-Clang"
},
{
"name": "my-Static-Tools-GNUC",
"configurePreset": "my-Static-Tools-GNUC",
"inherits": "ci-x64-Release-GNUC"
},
{
"name": "my-Static-Tools-macos-GNUC",
"configurePreset": "my-Static-Tools-macos-GNUC",
"inherits": "ci-macos-Release-GNUC"
}
],
"workflowPresets": [
{
"name": "my-Static-Tools-MSVC",
"steps": [
{"type": "configure", "name": "my-Static-Tools-MSVC"},
{"type": "build", "name": "my-Static-Tools-MSVC"},
{"type": "test", "name": "my-Static-Tools-MSVC"},
{"type": "package", "name": "my-Static-Tools-MSVC"}
]
},
{
"name": "my-StdShar-Clang",
"steps": [
{"type": "configure", "name": "my-Static-Tools-Clang"},
{"type": "build", "name": "my-Static-Tools-Clang"},
{"type": "test", "name": "my-Static-Tools-Clang"},
{"type": "package", "name": "my-Static-Tools-Clang"}
]
},
{
"name": "my-StdShar-GNUC",
"steps": [
{"type": "configure", "name": "my-Static-Tools-GNUC"},
{"type": "build", "name": "my-Static-Tools-GNUC"},
{"type": "test", "name": "my-Static-Tools-GNUC"},
{"type": "package", "name": "my-Static-Tools-GNUC"}
]
},
{
"name": "my-Static-Tools-macos-GNUC",
"steps": [
{"type": "configure", "name": "my-Static-Tools-macos-GNUC"},
{"type": "build", "name": "my-Static-Tools-macos-GNUC"},
{"type": "test", "name": "my-Static-Tools-macos-GNUC"},
{"type": "package", "name": "my-Static-Tools-macos-GNUC"}
]
}
]
} 

Discussion
    The above example shows how to build HDF5 with a static library and tools. The CMakeUserPresets.json file is located in the root directory of the HDF5 source. It is from here you execute: 
  * `cmake –workflow –preset my-Static-Tools-<compiler-type> –fresh`  
where "<compiler-type>" is GNUC or MSVC or Clang or macos-GNUC

If you don't need cross-compiler or cross-platform support, you can remove the compiler-specific presets and use the "my-Static-Tools" preset directly. 

See Also
     [Default CMakePresets.json Settings](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#subsec_cmake_presets_files_json_details) for details on the default settings for the StdShar presets.
##  CMake Presets Use Case: S3 on linux 

Problem
    I want to build the HDF5 library with the ros3 vfd. 

Solution
    Create a CMakeUserPresets.json file with the following content: 
{
"version": 6,
"cmakeMinimumRequired": {
"major": 3,
"minor": 25,
"patch": 0
},
"configurePresets": [
{
"name": "my-linux-S3",
"hidden": true,
"description": "Build HDF5 with library and tools for S3",
"inherits": ["ci-base", "ci-S3", "ci-CompressionVars", "ci-StdExamples", "ci-StdPlugins"],
"cacheVariables": {
"HDF_PACKAGE_NAMESPACE": {"type": "STRING", "value": "hdf5::"},
"HDF5_BUILD_SHARED_LIBS": "ON",
"HDF5_BUILD_TOOLS": "ON",
"HDF5_ENABLE_ZLIB_SUPPORT": "ON",
"HDF5_ENABLE_LIBAEC": "ON",
"HDF5_ENABLE_PLUGIN_SUPPORT": "ON"
}
},
{
"name": "my-linux-S3-Clang",
"description": "Clang Standard Config for x64 (Release)",
"inherits": [
"ci-x64-Release-Clang",
"ci-CPP",
"ci-Fortran",
"my-linux-S3"
]
},
{
"name": "my-linux-S3-GNUC",
"description": "GNUC Standard Config for x64 (Release)",
"inherits": [
"ci-x64-Release-GNUC",
"ci-CPP",
"ci-Fortran",
"my-linux-S3"
]
}
],
"buildPresets": [
{
"name": "my-linux-S3-Clang",
"description": "Clang Standard Build for x64 (Release)",
"configurePreset": "my-linux-S3-Clang",
"inherits": [
"ci-x64-Release-Clang"
]
},
{
"name": "my-linux-S3-GNUC",
"description": "GNUC Standard Build for x64 (Release)",
"configurePreset": "my-linux-S3-GNUC",
"inherits": [
"ci-x64-Release-GNUC"
]
}
],
"testPresets": [
{
"name": "my-linux-S3-Clang",
"configurePreset": "my-linux-S3-Clang",
"inherits": [
"ci-x64-Release-Clang"
]
},
{
"name": "my-linux-S3-GNUC",
"configurePreset": "my-linux-S3-GNUC",
"inherits": [
"ci-x64-Release-GNUC"
]
}
],
"packagePresets": [
{
"name": "my-linux-S3-Clang",
"configurePreset": "my-linux-S3-Clang",
"inherits": "ci-x64-Release-Clang"
},
{
"name": "my-linux-S3-GNUC",
"configurePreset": "my-linux-S3-GNUC",
"inherits": "ci-x64-Release-GNUC"
}
],
"workflowPresets": [
{
"name": "my-linux-S3-Clang",
"steps": [
{"type": "configure", "name": "my-linux-S3-Clang"},
{"type": "build", "name": "my-linux-S3-Clang"},
{"type": "test", "name": "my-linux-S3-Clang"},
{"type": "package", "name": "my-linux-S3-Clang"}
]
},
{
"name": "my-linux-S3-GNUC",
"steps": [
{"type": "configure", "name": "my-linux-S3-GNUC"},
{"type": "build", "name": "my-linux-S3-GNUC"},
{"type": "test", "name": "my-linux-S3-GNUC"},
{"type": "package", "name": "my-linux-S3-GNUC"}
]
}
]
} 

Discussion
    The above example shows how to build HDF5 with a static library and tools. The CMakeUserPresets.json file is located in the root directory of the HDF5 source. It is from here you execute: 
  * `cmake –workflow –preset my-S3-<compiler-type> –fresh`  
where "<compiler-type>" is GNUC or Clang or macos-GNUC

If you don't need cross-compiler support, you can remove the compiler-specific presets and use the "my-linux-S3" preset directly. 

See Also
     [Default CMakePresets.json Settings](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#subsec_cmake_presets_files_json_details) for details on the default settings for the StdShar presets.
##  CMake Presets Use Case: Parallel Library 

Problem
    I want to build the HDF5 library with a parallel compiler. 

Solution
    Create a CMakeUserPresets.json file with the following content: 
{
"version": 6,
"cmakeMinimumRequired": {
"major": 3,
"minor": 25,
"patch": 0
},
"configurePresets": [
{
"name": "my-linux-par",
"hidden": true,
"description": "Build HDF5 with library and tools using parallel compiler",
"inherits": ["ci-base", "ci-CompressionVars", "ci-StdExamples"],
"environment": {
"CC": "mpicc"
},
"cacheVariables": {
"HDF_PACKAGE_NAMESPACE": {"type": "STRING", "value": "hdf5::"},
"HDF5_BUILD_SHARED_LIBS": "ON",
"HDF5_BUILD_TOOLS": "ON",
"MPIEXEC_NUMPROC_FLAG": "-n",
"MPIEXEC_MAX_NUMPROCS": "2",
"HDF5_ENABLE_PARALLEL": "ON",
"HDF5_ENABLE_SUBFILING_VFD": "OFF",
"HDF5_ENABLE_ZLIB_SUPPORT": "ON",
"HDF5_ENABLE_LIBAEC": "ON",
"HDF5_ENABLE_PLUGIN_SUPPORT": "OFF"
}
},
{
"name": "my-linux-par-mpich",
"description": "mpich Standard Config for x64 (Release)",
"inherits": [
"ci-x64-Release-GNUC",
"my-linux-par"
]
}
],
"buildPresets": [
{
"name": "my-linux-par-mpich",
"description": "mpich Standard Build for x64 (Release)",
"configurePreset": "my-linux-par-mpich",
"inherits": [
"ci-x64-Release-GNUC"
]
}
],
"testPresets": [
{
"name": "my-linux-serial-mpich",
"configurePreset": "my-linux-par-mpich",
"inherits": [
"ci-x64-Release-GNUC"
],
"filter": {
"exclude": {
"name": "MPI_TEST"
}
}
},
{
"name": "my-linux-par-mpich",
"configurePreset": "my-linux-par-mpich",
"inherits": [
"ci-x64-Release-GNUC"
],
"filter": {
"include": {
"name": "MPI_TEST"
}
}
}
],
"packagePresets": [
{
"name": "my-linux-par-mpich",
"configurePreset": "my-linux-par-mpich",
"inherits": "ci-x64-Release-GNUC"
}
],
"workflowPresets": [
{
"name": "my-linux-par-mpich",
"steps": [
{"type": "configure", "name": "my-linux-par-mpich"},
{"type": "build", "name": "my-linux-par-mpich"},
{"type": "test", "name": "my-linux-serial-mpich"},
{"type": "test", "name": "my-linux-par-mpich"},
{"type": "package", "name": "my-linux-par-mpich"}
]
}
]
} 

Discussion
    The above example shows how to build HDF5 with a parallel compiler. The CMakeUserPresets.json file is located in the root directory of the HDF5 source. It is from here you execute: 
  * `cmake –workflow –preset my-linux-par-GNUC –fresh`



See Also
     [Default CMakePresets.json Settings](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#subsec_cmake_presets_files_json_details) for details on the default settings for the StdShar presets.
##  CMake Presets Use Case: Java Bindings 

Problem
    I want to build HDF5 with Java bindings for use in Java applications. 

Solution
    HDF5 provides two Java implementation options with the same API: 
  * **JNI (Java Native Interface)** - Default, works with Java 8+, production-stable 
  * **FFM (Foreign Function & Memory)** - Optional, requires Java 25+, modern native access

_Note: FFM will become the default in the next major HDF5 release as Java 25+ adoption increases._ 

JNI Build (Default - Java 8+)
    The standard presets automatically include JNI bindings when Java is detected: 
cmake --workflow --preset ci-StdShar-GNUC --fresh
This builds shared libraries with C++, Fortran, and Java JNI bindings. Java JNI requires shared libraries and cannot be built with static libraries. 

FFM Build (Java 25+)
    For Java 25+ with FFM bindings, you need: 
  * Java 25 or later (tested with Java 25 - Oracle distribution recommended) 
  * jextract tool installed (download from <https://jdk.java.net/jextract/>) 
  * Set JEXTRACT_HOME environment variable: `export JEXTRACT_HOME=$HOME/tools/jextract-25`


cmake --workflow --preset ci-StdShar-GNUC-FFM --fresh
FFM bindings are generated automatically at CMake configure time and placed in the `build/java/jsrc/` directory. They are never committed to the repository and are always fresh for each build. 

JNI vs FFM Comparison
     Feature | JNI (Default) | FFM (Future Default)   
---|---|---  
Java Version | Java 8+ | Java 25+   
Build Requirements | Java JDK only | Java JDK + jextract tool   
Library Type | Shared only | Shared   
Generation | Pre-compiled native code | Auto-generated at configure time   
Performance | Mature, optimized | Modern, Project Panama   
API Compatibility | Identical - same [hdf.hdf5lib](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf_1_1hdf5lib.html).* package  

Maven Artifacts
    To build platform-specific Maven artifacts for deployment:   
# JNI artifacts (default) - minimal build for artifacts only
cmake --workflow --preset ci-MinShar-GNUC-Maven --fresh
# FFM artifacts (Java 25+) - minimal build for artifacts only
cmake --workflow --preset ci-MinShar-GNUC-Maven-FFM --fresh
# Snapshot versions for development (adds -SNAPSHOT suffix)
cmake --workflow --preset ci-MinShar-GNUC-Maven-Snapshot --fresh
Maven artifacts are deployed to: 
  * GitHub Packages: <https://maven.pkg.github.com/HDFGroup/hdf5>
  * Maven Central (coming soon)



Platform Support
    Both JNI and FFM implementations support: 
  * Linux x86_64 
  * Windows x86_64 
  * macOS x86_64 
  * macOS aarch64 (Apple Silicon)

Each platform requires a separate build with platform-specific classifiers in Maven artifacts. 

Testing Java Bindings
    After building, test the Java bindings: 
# All Java tests (JNI or FFM depending on build)
ctest -R "Java" -V
# FFM-specific tests (if FFM build)
ctest -R "JUnitFFM" -V 

See Also
    
  * [Maven Artifact Selection](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#subsec_maven_artifacts) for using pre-built Maven artifacts 
  * release_docs/INSTALL_CMake.txt Section XIII for Java FFM testing details 
  * [release_docs/README.md](https://support.hdfgroup.org/documentation/hdf5/latest/release__docs_2_r_e_a_d_m_e_8md.html) for comprehensive Java documentation 
  * HDF5Examples/JAVA for Java example programs


#  CMake Presets
Presets are a way to store a set of CMake cache variables in a file. This allows you to save and load a set of variables that you use frequently. Presets can be used to save the configuration of a project for different build configurations, such as Debug and Release.
CMake supports two main files, **CMakePresets.json** and **CMakeUserPresets.json** , that allow users to specify common configure options and share them with others. CMake also supports files included with the include field.
##  CMakePresets.json
HDF Group provides a [`CMakePresets.json`](https://github.com/HDFGroup/hdf5/blob/develop/CMakePresets.json) file, used by the github workflows, that includes the following StdShar presets: 
  * ci-StdShar-MSVC - use MSVC compiler 
  * ci-StdShar-Clang - use Clang compiler 
  * ci-StdShar-macos-Clang -Apple Clang compiler 
  * ci-StdShar-GNUC - use GNU compiler 
  * ci-StdShar-macos-GNUC - use Apple GNU compiler 
  * ci-StdShar-GNUC-S3 - use GNU compiler with S3 support 
  * ci-StdShar-Intel - use Intel OneAPI compiler 
  * ci-StdShar-win-Intel - use Intel OneAPI compiler on Windows


The StdShar presets are used by the GitHub workflows to build and test the HDF5 library on different platforms and compilers. The [`CMakePresets.json`](https://github.com/HDFGroup/hdf5/blob/develop/CMakePresets.json) file is located in the root directory of the HDF5 repository. The StdShar presets inherit the following settings: 
  * ci-CPP - enables C++ support 
  * ci-Fortran - enables Fortran support, except on macOS 
  * ci-Java - enables Java support 
  * ci-StdShar - described below


The ci-StdShar preset inherits the following settings: 
  * ci-StdCompression 
  * ci-StdExamples 
  * ci-StdPlugins


See the [`CMakePresets.json`](https://github.com/HDFGroup/hdf5/blob/develop/CMakePresets.json) file for the complete list of presets and settings. The [Default CMakePresets.json Settings](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#subsec_cmake_presets_files_json_details) section provides an overview of the default settings for the StdShar presets.
##  Default CMakePresets.json Settings
The ci-StdShar preset sets the following cache variables: 
  * "HDF_PACKAGE_NAMESPACE": {"type": "STRING", "value": "hdf5::"} 
  * "HDF5_INSTALL_MOD_FORTRAN": "NO" 
  * "HDF5_ENABLE_ALL_WARNINGS": "ON" 
  * "HDF5_MINGW_STATIC_GCC_LIBS": "ON" 
  * "HDF_TEST_EXPRESS": "2"


The ci-StdExamples preset inherits the following settings: 
  * ci-base 
  * ci-base-tgz


The ci-StdExamples preset sets the following cache variables: 
  * "HDF5_PACK_EXAMPLES": "ON" 
  * "EXAMPLES_DOWNLOAD": "ON"


The ci-StdPlugins preset inherits the following settings: 
  * ci-base-plugins 
  * ci-PluginsVars 
  * ci-base-tgz


The ci-StdPlugins preset sets the following cache variables: 
  * "PLUGIN_USE_LOCALCONTENT": "OFF"


The ci-PluginsVars preset sets the following cache variables (see the <https://github.com/HDFGroup/hdf5/blob/develop/config/CacheURLs.cmake> file for the latest versions): 
  * "HDF5_ENABLE_PLUGIN_SUPPORT": "ON" 
  * "H5PL_ALLOW_EXTERNAL_SUPPORT": {"type": "STRING", "value": "TGZ"} 
  * "PLUGIN_PACKAGE_NAME": {"type": "STRING", "value": "pl"} 
  * "PLUGIN_TGZ_ORIGPATH": {"type": "STRING", "value": "hdf5_plugins download"} 
  * "PLUGIN_TGZ_NAME": {"type": "STRING", "value": "hdf5_plugins-master.tar.gz"}


The ci-base-plugins preset sets the following cache variables (see the <https://github.com/HDFGroup/hdf5/blob/develop/config/CacheURLs.cmake> file for the latest versions): 
  * "BITGROOM_PACKAGE_NAME": {"type": "STRING", "value": "bitgroom"}, 
  * "BITROUND_PACKAGE_NAME": {"type": "STRING", "value": "bitround"}, 
  * "BSHUF_TGZ_NAME": {"type": "STRING", "value": "bitshuffle-x.y.z.tar.gz"} 
  * "BSHUF_PACKAGE_NAME": {"type": "STRING", "value": "bshuf"} 
  * "BLOSC_TGZ_NAME": {"type": "STRING", "value": "c-blosc-x.y.z.tar.gz"} 
  * "BLOSC_PACKAGE_NAME": {"type": "STRING", "value": "blosc"} 
  * "BLOSC_ZLIB_TGZ_NAME": {"type": "STRING", "value": "zlib-x.y.z.tar.gz"} 
  * "BLOSC_ZLIB_PACKAGE_NAME": {"type": "STRING", "value": "zlib"} 
  * "BLOSC2_TGZ_NAME": {"type": "STRING", "value": "c-blosc2-x.y.z.tar.gz"} 
  * "BLOSC2_PACKAGE_NAME": {"type": "STRING", "value": "blosc2"} 
  * "BLOSC2_ZLIB_TGZ_NAME": {"type": "STRING", "value": "zlib-x.y.z.tar.gz"} 
  * "BLOSC2_ZLIB_PACKAGE_NAME": {"type": "STRING", "value": "zlib"} 
  * "BZ2_TGZ_NAME": {"type": "STRING", "value": "bzip2-bzip2-x.y.z.tar.gz"} 
  * "BZ2_PACKAGE_NAME": {"type": "STRING", "value": "bz2"} 
  * "FPZIP_TGZ_NAME": {"type": "STRING", "value": "fpzip-x.y.z.tar.gz"} 
  * "FPZIP_PACKAGE_NAME": {"type": "STRING", "value": "fpzip"} 
  * "JPEG_TGZ_NAME": {"type": "STRING", "value": "jpegsrc.v9e.tar.gz"} 
  * "JPEG_PACKAGE_NAME": {"type": "STRING", "value": "jpeg"} 
  * "BUILD_LZ4_LIBRARY_SOURCE": "ON" 
  * "LZ4_TGZ_NAME": {"type": "STRING", "value": "lz4-x.y.z.tar.gz"} 
  * "LZ4_PACKAGE_NAME": {"type": "STRING", "value": "lz4"} 
  * "LZF_TGZ_NAME": {"type": "STRING", "value": "liblzf-x.y.z.tar.gz"} 
  * "LZF_PACKAGE_NAME": {"type": "STRING", "value": "lzf"} 
  * "SZ_TGZ_NAME": {"type": "STRING", "value": "SZ-x.y.z.tar.gz"} 
  * "SZ_PACKAGE_NAME": {"type": "STRING", "value": "SZ"} 
  * "ZFP_TGZ_NAME": {"type": "STRING", "value": "zfp-x.y.z.tar.gz"} 
  * "ZFP_PACKAGE_NAME": {"type": "STRING", "value": "zfp"} 
  * "ZSTD_TGZ_NAME": {"type": "STRING", "value": "zstd-x.y.z.tar.gz"} 
  * "ZSTD_PACKAGE_NAME": {"type": "STRING", "value": "zstd"}


The ci-StdCompression preset inherits the following settings: 
  * ci-base-tgz 
  * ci-CompressionVars


The ci-StdCompression preset sets the following cache variables: 
  * "HDF5_PACKAGE_EXTLIBS": "ON" 
  * "HDF5_USE_ZLIB_NG": "OFF" 
  * "ZLIB_USE_LOCALCONTENT": "OFF" 
  * "LIBAEC_USE_LOCALCONTENT": "OFF" 
  * "HDF5_USE_ZLIB_STATIC": "ON" 
  * "HDF5_USE_LIBAEC_STATIC": "ON"


The ci-CompressionVars preset sets the following cache variables (see the <https://github.com/HDFGroup/hdf5/blob/develop/config/CacheURLs.cmake> file for the latest versions): 
  * "ZLIB_PACKAGE_NAME": {"type": "STRING", "value": "zlib"} 
  * "ZLIB_TGZ_ORIGPATH": {"type": "STRING", "value": "zlib download"} 
  * "ZLIB_TGZ_NAME": {"type": "STRING", "value": "zlib-x.y.z.tar.gz"} 
  * "ZLIBNG_PACKAGE_NAME": {"type": "STRING", "value": "zlib-ng"} 
  * "ZLIBNG_TGZ_ORIGPATH": {"type": "STRING", "value": "zlib-ng download"} 
  * "ZLIBNG_TGZ_NAME": {"type": "STRING", "value": "x.y.z.tar.gz"} 
  * "LIBAEC_PACKAGE_NAME": {"type": "STRING", "value": "libaec"} 
  * "LIBAEC_TGZ_ORIGPATH": {"type": "STRING", "value": "libaec download"} 
  * "LIBAEC_TGZ_NAME": {"type": "STRING", "value": "libaec-x.y.z.tar.gz"}


The ci-base-tgz preset inherits the following settings: 
  * ci-base


The ci-base-tgz preset sets the following cache variables: 
  * "HDF5_ALLOW_EXTERNAL_SUPPORT": {"type": "STRING", "value": "TGZ"} 
  * "TGZPATH": {"type": "PATH", "value": "${sourceParentDir}/temp"}


The ci-base preset sets the following CMake variables: 
  * "name": "ci-base" 
  * "displayName": "Basic Config" 
  * "description": "Basic build using Ninja generator" 
  * "generator": "Ninja" 
  * "binaryDir": "${sourceParentDir}/build/${presetName}" 
  * "installDir": "${sourceParentDir}/install/${presetName}"


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


