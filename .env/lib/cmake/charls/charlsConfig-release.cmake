#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "charls" for configuration "Release"
set_property(TARGET charls APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(charls PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcharls.2.2.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libcharls.2.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS charls )
list(APPEND _IMPORT_CHECK_FILES_FOR_charls "${_IMPORT_PREFIX}/lib/libcharls.2.2.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
