# ROL Trilinos Snapshot Instructions

## Overview

This document describes how to create Trilinos snapshots of the standalone ROL repository while properly handling external dependencies and build system differences.

## Standalone ROL Dependencies

The standalone ROL build uses several external dependencies that are **NOT** included in Trilinos snapshots:

### External Dependencies (Excluded from snapshots)
- **pugixml**: XML parsing library, fetched via CMake FetchContent
- **Eigen3**: Linear algebra library, found via system installation
- **BLAS/LAPACK**: Linear algebra libraries, found via system installation

### Build System Differences
- **Standalone**: Uses native CMake with FetchContent for dependencies
- **Trilinos**: Uses TriBITS build system with Trilinos-provided dependencies

## Snapshot Process

### 1. Files to Include in Trilinos Snapshots
```
packages/rol/
├── src/                    # All ROL source code
├── test/                   # All ROL tests  
├── example/               # All ROL examples
├── tutorial/              # All ROL tutorials
├── cmake/                 # ROL CMake modules (including ROLUtils.cmake)
├── CMakeLists.txt         # Main build file with dual TriBITS/standalone support
├── README.md              # Documentation
└── SNAPSHOT.md            # This file
```

### 2. Files to Exclude from Trilinos Snapshots
```
packages/rol/
├── _deps/                 # CMake FetchContent downloads (auto-generated)
├── build*/                # Build directories
├── .git*/                 # Git metadata (if using git submodules)
├── external/              # Future: git submodules location
└── .*                     # Hidden files (.gitignore, etc.)
```

### 3. Recommended Snapshot Script

Create a script that excludes FetchContent artifacts:

```bash
#!/bin/bash
# snapshot-rol.sh - Create Trilinos snapshot of standalone ROL

STANDALONE_ROL_DIR="path/to/standalone/rol"
TRILINOS_ROL_DIR="path/to/trilinos/packages/rol"

# Use rsync to copy while excluding build artifacts
rsync -av --delete \
  --exclude='_deps/' \
  --exclude='build*/' \
  --exclude='.git*' \
  --exclude='external/' \
  --exclude='.*' \
  "${STANDALONE_ROL_DIR}/" \
  "${TRILINOS_ROL_DIR}/"

echo "ROL snapshot complete"
```

### 4. Build System Compatibility

The `CMakeLists.txt` file automatically detects the build environment:

```cmake
if( COMMAND TRIBITS_PACKAGE )
  set( STANDALONE_ROL FALSE )
else()
  set( STANDALONE_ROL TRUE )
endif()
```

- **TriBITS environment**: Uses `cmake/TrilinosROL.cmake` 
- **Standalone environment**: Uses FetchContent and system dependencies

### 5. Test System Compatibility

The `cmake/ROLUtils.cmake` provides compatibility functions:

- `ROL_ADD_EXECUTABLE_AND_TEST()`: Forwards to `TRIBITS_ADD_EXECUTABLE_AND_TEST()` in TriBITS
- `ROL_ADD_EXECUTABLE()`: Forwards to `TRIBITS_ADD_EXECUTABLE()` in TriBITS  
- `ROL_COPY_FILES_TO_BINARY_DIR()`: Forwards to `TRIBITS_COPY_FILES_TO_BINARY_DIR()` in TriBITS

## Migration Strategy

### Phase 1: Current State
- ROL source code lives in both standalone repo and Trilinos
- Manual snapshots synchronize changes
- Dual build system support via feature detection

### Phase 2: Future Submodule Approach (Optional)
If Trilinos policies change to allow submodules:

1. Add ROL as git submodule in Trilinos
2. Use `.gitignore` in Trilinos to exclude `external/` from ROL submodule  
3. Automated snapshot via git submodule updates

```bash
# In Trilinos repo
git submodule add https://github.com/trilinos/ROL.git packages/rol
echo "packages/rol/external/" >> .gitignore
echo "packages/rol/_deps/" >> .gitignore
```

## Testing Snapshots

After creating a snapshot, verify compatibility:

### In Trilinos Build
```bash
cd trilinos-build
cmake -DROL_ENABLE_TESTS=ON ..
make -j8
ctest -R ROL
```

### In Standalone Build  
```bash
cd rol-build
cmake -DENABLE_TESTS=ON ../path/to/standalone/rol
make -j8
ctest
```

## Important Notes

1. **FetchContent artifacts** (`_deps/`) are **never** included in snapshots
2. **All source code** changes must be compatible with both build systems
3. **Test all changes** in both standalone and TriBITS environments before snapshot
4. **CMake compatibility functions** in `ROLUtils.cmake` handle build system differences
5. **XML support** uses pugixml in standalone, existing Trilinos XML in TriBITS

## Contact

For questions about the snapshot process or build system compatibility, contact the ROL development team.