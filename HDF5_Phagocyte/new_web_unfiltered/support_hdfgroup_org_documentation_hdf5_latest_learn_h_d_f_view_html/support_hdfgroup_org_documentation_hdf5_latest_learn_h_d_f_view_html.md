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
  * [ HDFView Installation](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title0 "H1")
  * [ Begin Tutorial](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title1 "H1")
  * [ Topics Covered](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title2 "H2")
  * [ Topics](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title3 "H1")
  * [ Creating a New HDF5 File with a Contiguous Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title4 "H2")
  * [ Displaying a Dataset as an Image](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title5 "H2")
  * [ Creating Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title6 "H2")
  * [ Creating a Compressed and Chunked Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title7 "H2")
  * [ Creating an Image and a Subset](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title8 "H2")
  * [ Creating a Table (Compound Dataset)](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#title9 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Learning HDF5 with HDFView
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
* * *
This tutorial enables you to get a feel for HDF5 by using the HDFView browser. It does NOT require any programming experience.
#  HDFView Installation
  * Download and install HDFView. It can be downloaded from the [Download HDFView](https://support.hdfgroup.org/releases/hdfview/v3_3/v3_3_2/downloads/index.html) page. 
  * Obtain the [storm1.txt](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/storm1.txt) text file, used in the tutorial.


#  Begin Tutorial
Once you have HDFView installed, bring it up and you are ready to begin the tutorial.
Unable to complete tutorial because fields are greyed out?  This tutorial requires that the default HDFView File Access Mode be Read / Write. If fields are greyed out so that you cannot select them, then the File Access Mode is Read Only. To change the File Access Mode follow these steps: 
  * Bring up HDFView 
  * Left-mouse click on the Tools pull-down menu and select User [Options](https://support.hdfgroup.org/documentation/hdf5/latest/struct_options.html). 
  * A Preferences window pops up with the General Settings tab selected. About half-way down you will see Default File Access Mode. Select Read / Write. 
  * Click on Apply and Close at the bottom of the window. 
  * Close down HDFView. 
  * Bring HDFView back up and try the tutorial again. PLEASE BE AWARE that selecting a File Access Mode of Read / Write can result in changes to the timestamp of HDF files that are viewed with HDFView. In general, a File Access Mode of Read Only should be used to ensure that this does not occur. 

  
---  
##  Topics Covered
Following are the topics covered in the tutorial. The first topic creates the file that is used in the subsequent topics. 
  * [Creating a New HDF5 File with a Contiguous Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#subsec_learn_hv_topics_file)
  * [Displaying a Dataset as an Image](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#subsec_learn_hv_topics_image)
  * [Creating Attributes](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#subsec_learn_hv_topics_attr)
  * [Creating a Compressed and Chunked Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#subsec_learn_hv_topics_compress)
  * [Creating an Image and a Subset](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#subsec_learn_hv_topics_subset)
  * [Creating a Table (Compound Dataset)](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_h_d_f_view.html#subsec_learn_hv_topics_table)


#  Topics
##  Creating a New HDF5 File with a Contiguous Dataset
The steps below describe how to create a file (storm.h5), group (/Data), and a contiguous dataset (/Data/Storm) using HDFView. A group is an HDF5 object that allows objects to be collected together. A dataset is an array of data values. A contiguous dataset is one that is stored as a single block in the HDF5 file. 
  * Select the _File_ pull-down menu at the top left, and then select _New - > HDF5_. 
  * Specify a location and type in _storm.h5_ for the name of your file, and click on the _Save_ button. You will see the _storm.h5_ file in the TableView:  ![](https://support.hdfgroup.org/documentation/hdf5/latest/storm.png)  
---  
  * Right click on _storm.h5_ , and select _New - > Group_. 
  * Enter _Data_ for the name of the group and then click the _Ok_ button. You will see the group _Data_ in the TableView.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/DataGroup.png)  
---  
  * Right click on the group _Data_ and select _New - > Dataset_. 
  * A window pops up on the right. Fill in the information as follows, and then click _Ok_ (leave the Datatype information as is):  Dataset Name  |  _Storm_  
---|---  
Under Dataspace, Current size  | 57x57   
Layout  |  _Contiguous_ (default)   
  * Click to expand the _Data_ group in the tree view to see the _Storm_ dataset:  ![](https://support.hdfgroup.org/documentation/hdf5/latest/StormDataset.png)  
---  
  * Double left click on the _Storm_ dataset in the tree view. A window with an empty spreadsheet pops open. 
  * Copy the data from the [storm1.txt](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/storm1.txt) file into the dataset.
If you downloaded [storm1.txt](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/storm1.txt), then click on the _Import/Export Data_ menu and select _Import Data from - > Text File_. Specify a location, select [storm1.txt](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/storm1.txt) and click on the _Open_ button. Answer _Yes_ in the dialog box that pops up (which asks if you wish to paste the selected data).
Alternately, you can copy/paste directly. Select and copy the data in a separate window. Position your cursor at (0,0) in your table, and select _Paste_ from the _Table_ menu.
The values will be entered into the spreadsheet. 
![](https://support.hdfgroup.org/documentation/hdf5/latest/datasetwdata.png)  
---  
  * _Table - > Close_ the dataset, and save the data. 


##  Displaying a Dataset as an Image
Any dataset can be viewed as an image in HDFView. Below are the steps that demonstrate this. 
  * Right click on _Storm_ in the tree view, and select _Open As_. 
  * Select the _Image_ button under _Display As_ (near the top) in the Dataset Selection window that pops up. Then click _OK_ at the bottom of the window to display the image.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/showasimage.png)  
---  
  * The rainbow icon brings you to the Image Palette window. Click on that to play with the palette (GrayWave probably is the best choice). Close. 


##  Creating Attributes
Additional information to describe an object can be stored in attributes. An attribute can be added to a group or dataset with HDFView.
The following illustrates how to add an attribute to the group _/Data_ : 
  * Click on the _/Data_ folder in the tree view. You will see two tabs, _Object Attribute Info_ and _General Object Info_ , in the pane on the right site of the HDFView window.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/noattrs.png)  
---  
  * With the left mouse button, select the _Add Attribute_ button. 
  * Select the _Add Attribute_ button to add an attribute with these values:  Name  |  _BatchID_  
---|---  
Type  | INTEGER   
Size (bits)  | 32   
  * Select the _Ok_ button. The attribute will show up under the _Object Attribute Info_ tab. 
  * Double-click the BatchID attribute line to open the data table for BatchID. 
  * Click in the first cell and enter _3343_ followed by the enter key. 
  * _Table - > Close_, answer _Yes_ in the dialog box that pops up (which asks if you wish to paste the selected data). 


Adding an attribute to a dataset is very similar to adding an attribute to a group. For example, the following adds an attribute to the _/Storm_ dataset: 
  * Left mouse click on the _/Storm_ dataset in the tree view. You will see the _Object Attribute Info_ and _General Object Info_ tabs on the right 
  * In the _Object Attribute Info_ pane select the _Add Attribute_ button and enter an attribute with these values. (Be sure to add a _String Length_ or the string will be truncated to one character!):  Name  |  _Units_  
---|---  
Type  | STRING   
String Length  | 3   
  * Select the _Ok_ button. The attribute will show up under the _Object Attribute Info_ tab. 
  * Double-click the Units attribute line to open the data table for Units. 
  * Click in the first cell and enter _m/s_ followed by the enter key. 
  * _Table - > Close_, answer _Yes_ in the dialog box that pops up (which asks if you wish to paste the selected data).  ![](https://support.hdfgroup.org/documentation/hdf5/latest/scarletletter.png)  
---  


##  Creating a Compressed and Chunked Dataset
A chunked and compressed dataset can be created using HDFView. A compressed dataset is a dataset whose size has been compressed to take up less space. In order to compress an HDF5 dataset, the dataset must be stored with a chunked dataset layout (as multiple _chunks_ that are stored separately in the file).
Please note that the chunk sizes used in this topic are for demonstration purposes only. For information on chunking and specifying an appropriate chunk size, see the [Chunking in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/hdf5_chunking.html) documentation.
Also see the HDF5 Tutorial topic on [Creating a Compressed Dataset](https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_com_dset.html#secLBComDsetCreate). 
  * Right click on storm.h5. Select _New - > Group_. 
  * Enter _Image_ for the name of the group, and click the _OK_ button to create the group.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/newgroupimage.png)  
---  
  * Right click on the _Image_ group, and select _New - > Dataset_. 
  * Enter the following information for the dataset. Leave the _Datatype_ as is (INTEGER):  Dataset name  |  _Another Storm_  
---|---  
Under Dataspace, Current size  | 57x57   
Storage Layout  | Chunked   
Chunk Size  | 20x20   
Compression  | gzip   
Compression Level  | 9   
You will see the _Another Storm_ dataset in the _Image_ group:  ![](https://support.hdfgroup.org/documentation/hdf5/latest/hdfview-anthrstrm.png)  
---  
  * Double left-mouse click on the _Another Storm_ dataset to display the spreadsheet:  ![](https://support.hdfgroup.org/documentation/hdf5/latest/hdfview-anthrstrm-sprdsht.png)  
---  
  * Copy the data from the [storm1.txt](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/storm1.txt) file into the dataset. (See the previous topic for copying [storm1.txt](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/storm1.txt) into a dataset.) 
  * _Table - > Close_, and save the data. 
  * Right click on _Another Storm_ , and select _Open As_. 
  * Select the _Image_ button in the Dataset Selection window that pops up. Click the _Ok_ button at the bottom of the window to view the dataset as an image.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/hdfview-anthrstrm-img.png)  
---  


##  Creating an Image and a Subset
A previous topic demonstrated how to view any dataset as an image in HDFView. With HDFView you can also create an image to begin with, as is shown below. 
  * Right click on the _Data_ group and select _New - > Image_. 
  * A window pops up on the right. Enter the following and then click _Ok_ : 
Image name  |  _Storm Image_  
---|---  
Height  | 57   
Width  | 57   
  * Close the dataset. 
  * Expand the _Data_ group to see its contents. You will see the _Storm Image_ dataset.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/hdfview-imgicon.png)  
---  
  * Add data to the _Storm Image_ dataset as was shown previously: 
    * Right click on _Storm Image_ , and select _Open As_ to open the Dataset Selection window. 
    * Click on the _Spreadsheet_ button at the top left of the Dataset Selection window to view the image as a spreadsheet. 
    * Copy the data from the [storm1.txt](https://support.hdfgroup.org/archive/support/ftp/HDF5/examples/files/tutorial/storm1.txt) file into the dataset. 
    * Close the dataset and save the data. 
  * Left double click on _Storm Image_ to see the image. Close the dataset. 
  * Right click on _Storm Image_ and select _Open As_ to bring up the Data Selection window. 
  * Select a subset by clicking the left mouse on the image in the window and dragging the mouse. Notice that the Height and Width values change. Select to display it as an image. Click _Ok_.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/hdfview-imgsubset.png)  
---  
  * Position the cursor in the middle of the image. Press Shift+Left Mouse button and hold, and then drag the mouse to select another subset. 
  * Select _Image- >Write Selection to Image_. Enter _Subset_ for the new image name. Click _Ok_. The _Subset_ image will appear in the tree view on the left. 
  * Left double click on the image _Subset_ to bring it up on the right.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/hdfview-newimgsubset.png)  
---  
  * Close the _Subset_ image. 


##  Creating a Table (Compound Dataset)
A dataset with a compound datatype contains data elements that consist of multiple fields. If the dataspace for the compound dataset is one-dimensional, then the dataset can be viewed as a table in HDFView, as is shown below. 
  * Right button click on the group _Data_. Select _New - > Compound DS_. 
  * A window pops up. Only fill in the following fields: 
Dataset name  | Table   
---|---  
Dataspace (Current size only)  | 4   
Compound Datatype Properties:   
Number of Members  | 3   
Compound Datatype Properties:   
_Name_ / Datatype / Size  |  _Description_ / string / 4   
_Temperature_ / float / 1   
_Pressure_ / double / 1   
![](https://support.hdfgroup.org/documentation/hdf5/latest/hdfview-newcmpd.png)  
---  
  * Click Ok at the bottom. 
  * Open the Data group (if it is not open) and double left click on the Table object.  ![](https://support.hdfgroup.org/documentation/hdf5/latest/hdfview-table.png)  
---  
  * Close the dataset. 


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Getting Started with HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


