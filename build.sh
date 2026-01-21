#!/bin/bash

source ../../setup-env.sh
export ROCM_VERSION=6.4.1

scorep-libwrap-init --name="rocblas" \
    -x c++ \
    --cppflags="-I/opt/rocm-$ROCM_VERSION/include -D__HIP_PLATFORM_AMD__" \
    --ldflags="-L/opt/rocm-$ROCM_VERSION/lib" \
    --libs="-lrocblas" \
    --update \
    .
printf "#ifndef LIBWRAP_H\n#define LIBWRAP_H\n#include <rocblas/rocblas.h>\n#endif /* LIBWRAP_H */\n" > libwrap.h

# cat <<EOF > rocblas.filter
# SCOREP_REGION_NAMES_BEGIN
#   EXCLUDE *
#   INCLUDE rocblas_dgemm*
#   INCLUDE rocblas_dtrsm*
# SCOREP_REGION_NAMES_END
# EOF

cat <<EOF > rocblas.filter
SCOREP_REGION_NAMES_BEGIN
  EXCLUDE *
  INCLUDE rocblas_*
SCOREP_REGION_NAMES_END
EOF

cat <<EOF > main.cc
#include <rocblas/rocblas.h>
int main() {
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    rocblas_destroy_handle(handle);
    return 0;
}
EOF

make scorep_libwrap_rocblas.cc
if [ ! -f scorep_libwrap_rocblas.cc ]; then
    echo "Error: scorep_libwrap_rocblas.cc was not generated."
    exit 1
fi
perl -i -pe 's|.*SCOREP_Libwrap_Plugins.h.*|$&\n\n#include <hip/hip_runtime.h>\n#ifdef SCOREP_LIBWRAP_EXIT_WRAPPED_REGION\n#undef SCOREP_LIBWRAP_EXIT_WRAPPED_REGION\n#endif\n#define SCOREP_LIBWRAP_EXIT_WRAPPED_REGION() do { hipDeviceSynchronize(); SCOREP_LIBWRAP_API( exit_wrapped_region )( scorep_libwrap_var_previous ); } while ( 0 )\n|' scorep_libwrap_rocblas.cc
make
make check           # execute tests
make install         # install wrapper
make installcheck    # execute more tests