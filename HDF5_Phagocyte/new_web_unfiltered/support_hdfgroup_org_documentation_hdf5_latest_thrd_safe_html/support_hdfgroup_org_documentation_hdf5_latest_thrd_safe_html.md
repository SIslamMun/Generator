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
HDF5 Threadsafety Warning
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
* * *
Any application that creates threads that use the HDF5 library must join those threads before either process exit or library close through [H5close()](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory."). If all HDF5-using threads aren't joined, the threads may exhibit undefined behavior.
#  Discussion for Developers on Potential Improvements
It would in principle be possible to make it safe to have threads continue using HDF5 resources after a call to [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") by keeping a count of threads within the library. (There is probably no solution to an early process exit producing undefined behavior within threads.) This method would only be able to count (and presumably, only _need_ to count) threads that directly interact with the library. Because each thread would need to be counted exactly once, this would most likely be done by use of a thread-local key with e.g. a boolean value used to track whether the a global atomic thread counter has already counted this thread. Then, if [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") is invoked while this thread counter is above one (because one thread must be doing the closing), the library would not close, and instead keep its resources valid to hopefully avoid bad behavior with the threads.
The issues with this approach are as follows: 
  1. The process of checking for the existence/value of the thread-local key is slow, or at least slow enough that it's probably not worth adding this to almost every single API call to prevent this particular edge case. 
  2. Even with this approach, bad behavior would still be possible if the application does something like expose HDF5 resources to threads indirectly via a global variable. 
  3. How to allow [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") to fail is nonobvious. [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") could be allowed to return an error indicating a failure to close, but the number of applications which could usefully respond to such an error by joining threads is small. If an application were able/willing to join its created threads, presumably it would have done so before calling [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory."). Alternatively, [H5close](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5.html#ga8a9fe81dcf66972ed75ea481e7750574 "Flushes all data to disk, closes all open objects, and releases memory.") could succeed but silently leave the library open. This creates the potential for confusing, unexpected behavior when the user thinks they are closing and re-opening the library, e.g. if environment variables are modified between close and re-open, or if resources such as default property lists are modified. 
  4. Applications should join threads before closing libraries that those threads are using, so all of this work would constitute an above-and-beyond effort to maintain safe and defined behavior in the face of an unsafe application. 


Despite these issues, if a more performant method was found to perform threadcounting like this, it might still constitute a worthwhile change.
* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Technical Notes](https://support.hdfgroup.org/documentation/hdf5/latest/_t_n.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


