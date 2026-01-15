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
  * [ Using HDF5 Java Bindings with Maven](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title0 "H1")
  * [ Maven Artifact Selection](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title1 "H2")
  * [ Using JNI Implementation (Java 8+)](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title2 "H2")
  * [ Using FFM Implementation (Java 25+)](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title3 "H2")
  * [ Using Java Examples](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title4 "H2")
  * [ Maven Repositories](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title5 "H2")
  * [ Troubleshooting](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title6 "H2")
  * [ Migrating Between JNI and FFM](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title7 "H2")
  * [ Testing Maven Artifacts](https://support.hdfgroup.org/documentation/hdf5/latest/_c_b__maven_artifacts.html#title8 "H2")


[•All](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))[](javascript:void\(0\))
Loading...
Searching...
No Matches
Using HDF5 Maven Artifacts
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
* * *
#  Using HDF5 Java Bindings with Maven 

Problem
    I want to use HDF5 Java bindings in my Maven project without building from source. 

Solution
    HDF5 Java bindings are available as Maven artifacts with platform-specific JARs for both JNI and FFM implementations. Use the appropriate artifact and classifier for your platform and Java version.
##  Maven Artifact Selection
HDF5 provides three main Maven artifacts:
Artifact ID | Java Version | Description | Status   
---|---|---|---  
**hdf5-java-jni** | Java 8+  | JNI implementation (default)  | Production-ready   
**hdf5-java-ffm** | Java 25+  | FFM implementation (future default)  | Production-ready   
**hdf5-java-examples** | Java 8+  | example programs  | Platform-independent   
_Note: FFM will become the default implementation in the next major HDF5 release._
##  Using JNI Implementation (Java 8+) 

Basic Usage
    Add the following dependency to your `pom.xml`:
<dependency>
<groupId>org.hdfgroup</groupId>
<artifactId>hdf5-java-jni</artifactId>
<version>2.0.0</version>
<classifier>linux-x86_64</classifier>
</dependency>
Replace the classifier with your platform: 
  * `linux-x86_64` - Linux 64-bit 
  * `windows-x86_64` - Windows 64-bit 
  * `macos-x86_64` - macOS Intel 
  * `macos-aarch64` - macOS Apple Silicon



Java Code
    
import hdf.hdf5lib.H5;
import hdf.hdf5lib.HDF5Constants;
public class MyHDF5App {
public static void main(String[] args) throws Exception {
// Initialize HDF5 library
[H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html).H5open();
// Your HDF5 operations here
long file_id = [H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html).H5Fcreate("myfile.h5",
HDF5Constants.H5F_ACC_TRUNC,
HDF5Constants.H5P_DEFAULT,
HDF5Constants.H5P_DEFAULT);
// Clean up
[H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html).H5Fclose(file_id);
[H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html).H5close();
}
}
[H5](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html)
**Definition** H5AbstractDs.cpp:33
##  Using FFM Implementation (Java 25+) 

Basic Usage
    Add the following dependency to your `pom.xml`:
<dependency>
<groupId>org.hdfgroup</groupId>
<artifactId>hdf5-java-ffm</artifactId>
<version>2.0.0</version>
<classifier>linux-x86_64</classifier>
</dependency> 

Java Runtime Configuration
    FFM requires enabling native access: 
java --enable-native-access=ALL-UNNAMED -cp yourapp.jar com.example.Main
Or configure in `pom.xml`: 
<plugin>
<groupId>org.apache.maven.plugins</groupId>
<artifactId>maven-surefire-plugin</artifactId>
<configuration>
<argLine>--enable-native-access=ALL-UNNAMED</argLine>
</configuration>
</plugin> 

Java Code
    The API is identical to JNI - same code works with both implementations: 
import hdf.hdf5lib.H5;
import hdf.hdf5lib.HDF5Constants;
// Same code as JNI example above
##  Using Java Examples 

Adding Examples Artifact
    
<dependency>
<groupId>org.hdfgroup</groupId>
<artifactId>hdf5-java-examples</artifactId>
<version>2.0.0</version>
</dependency>
The examples artifact contains 62 educational examples covering: 
  * File operations and dataset creation 
  * Compound datatypes and variable-length data 
  * Groups, attributes, and links 
  * Chunking, compression, and filters 
  * Parallel I/O examples



Running Examples
    
# List available examples
java -cp [hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)-java-examples-2.0.0.jar examples.intro.H5_CreateFile
# Run a specific example
java -cp "hdf5-java-jni-2.0.0-linux-x86_64.jar:hdf5-java-examples-2.0.0.jar" \
examples.intro.H5_CreateFile
[hdf5](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)
**Definition** HDF5.F90:26
See `HDF5Examples/JAVA/README-MAVEN.md[](https://support.hdfgroup.org/documentation/hdf5/latest/_r_e_a_d_m_e-_m_a_v_e_n_8md.html)` for complete details.
##  Maven Repositories 

GitHub Packages (Current)
    HDF5 artifacts are published to GitHub Packages. Add this repository to your `pom.xml`:
<repositories>
<repository>
<id>github-hdfgroup</id>
<name>HDF Group GitHub Packages</name>
<url>https://maven.pkg.github.com/HDFGroup/hdf5</url>
</repository>
</repositories>
**Note:** GitHub Packages requires authentication even for public packages. Add credentials to your `~/.m2/settings.xml`:
<servers>
<server>
<id>github-hdfgroup</id>
<username>YOUR_GITHUB_USERNAME</username>
<password>YOUR_GITHUB_TOKEN</password>
</server>
</servers>
Generate a GitHub personal access token with `read:packages` permission. 

Maven Central (Coming Soon)
    Future releases will be available on Maven Central, which does not require authentication:
<!-- Maven Central (future releases) -->
<dependency>
<groupId>org.hdfgroup</groupId>
<artifactId>hdf5-java-jni</artifactId>
<version>2.1.0</version>
<classifier>linux-x86_64</classifier>
</dependency>
##  Troubleshooting 

UnsatisfiedLinkError
    If you see `java.lang.UnsatisfiedLinkError`: 
  * Verify you're using the correct platform classifier 
  * Ensure only ONE HDF5 artifact (JNI or FFM) is on classpath 
  * Check Java version matches implementation (Java 25+ for FFM) 
  * Install HDF5 native libraries: `sudo apt-get install libhdf5-dev` (Linux) or `brew install hdf5[](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf5.html)` (macOS)



Native Access Error (FFM only)
    If you see `IllegalCallerException` with FFM: 
  * Add `–enable-native-access=ALL-UNNAMED` to Java command 
  * Or configure in build tool (Maven, Gradle)



Version Conflicts
    If you see dependency conflicts: 
  * Use `<dependencyManagement>` to control versions 
  * Exclude transitive HDF5 dependencies if needed 
  * Ensure all HDF5 artifacts use the same version


##  Migrating Between JNI and FFM 

API Compatibility
    Both implementations use identical APIs: 
  * Same package names: `hdf.hdf5lib[](https://support.hdfgroup.org/documentation/hdf5/latest/namespacehdf_1_1hdf5lib.html).*`
  * Same class names: `H5[](https://support.hdfgroup.org/documentation/hdf5/latest/namespace_h5.html), HDF5Constants`
  * Same method signatures 
  * Same exception handling



Migration Steps
    
  1. Update artifact ID in `pom.xml`: 
     * Change `hdf5-java-jni` to `hdf5-java-ffm`
     * Or vice versa
  2. For FFM: Add native access flag 
     * `–enable-native-access=ALL-UNNAMED`
  3. Rebuild and test 
     * No code changes required!


##  Testing Maven Artifacts 

Local Testing Scripts
    HDF5 provides test scripts to verify Maven artifacts work correctly:
cd HDF5Examples/JAVA
# Test JNI implementation
./test-maven-jni.sh 2.0.0
# Test FFM implementation
./test-maven-ffm.sh 2.0.0
# Test against custom repository
./test-maven-ffm.sh 2.0.0 https://maven.pkg.github.com/YOUR_FORK/hdf5
These scripts automatically: 
  * Configure GitHub authentication 
  * Download and verify JARs 
  * Compile and run example programs 
  * Report detailed test results


See `HDF5Examples/JAVA/README-MAVEN.md[](https://support.hdfgroup.org/documentation/hdf5/latest/_r_e_a_d_m_e-_m_a_v_e_n_8md.html)` for complete testing documentation. 

See Also
    
  * [CMake Presets Use Case: Java Bindings](https://support.hdfgroup.org/documentation/hdf5/latest/cmake-presets.html#subsec_cmake_presets_build_java) for building from source 
  * [release_docs/README.md](https://support.hdfgroup.org/documentation/hdf5/latest/release__docs_2_r_e_a_d_m_e_8md.html) for complete Maven documentation 
  * [HDF5Examples/JAVA/README-MAVEN.md](https://support.hdfgroup.org/documentation/hdf5/latest/_r_e_a_d_m_e-_m_a_v_e_n_8md.html) for testing and examples with Maven 
  * HDF5Examples/JAVA/test-maven-jni.sh for JNI artifact testing 
  * HDF5Examples/JAVA/test-maven-ffm.sh for FFM artifact testing 
  * <https://github.com/HDFGroup/hdf5-examples> for more examples


* * *
Navigate back: [Main](https://support.hdfgroup.org/documentation/hdf5/latest/index.html) / [Cookbook](https://support.hdfgroup.org/documentation/hdf5/latest/_cookbook.html)
  * Generated by [![doxygen](https://support.hdfgroup.org/documentation/hdf5/latest/doxygen.svg)](https://www.doxygen.org/index.html) 1.13.2 


