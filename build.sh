#!/bin/bash

source ../../setup-env.sh
export ROCM_VERSION=6.4.1

scorep-libwrap-init --name="rocblas" \
    -x c++ \
    --cppflags="-I/opt/rocm-$ROCM_VERSION/include -D__HIP_PLATFORM_AMD__ -DSCOREP_USER_ENABLE" \
    --ldflags="-L/opt/rocm-$ROCM_VERSION/lib" \
    --libs="-lrocblas -lamdhip64" \
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

  INCLUDE rocblas_*gemm*
  INCLUDE rocblas_*trsm*
  INCLUDE rocblas_*trmm*
  INCLUDE rocblas_*symm*
  INCLUDE rocblas_*syrk*
  INCLUDE rocblas_*syr2k*
  INCLUDE rocblas_*trtri*
  INCLUDE rocblas_*geam*

  INCLUDE rocblas_*gemv*
  INCLUDE rocblas_*trsv*
  INCLUDE rocblas_*ger*
  INCLUDE rocblas_*syr*
  INCLUDE rocblas_*trmv*

  EXCLUDE rocblas_*create*
  EXCLUDE rocblas_*destroy*
  EXCLUDE rocblas_*set*
  EXCLUDE rocblas_*get*
  EXCLUDE rocblas_*query*
  EXCLUDE rocblas_*log*
  EXCLUDE rocblas_*status*
  EXCLUDE rocblas_*version*
  
  EXCLUDE rocblas_*set_vector*
  EXCLUDE rocblas_*get_vector*
  EXCLUDE rocblas_*set_matrix*
  EXCLUDE rocblas_*get_matrix*
SCOREP_REGION_NAMES_END
EOF

cat <<EOF > main.cc
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Inicializar dimensiones y escalares (C = alpha*A*B + beta*C)
    rocblas_int m = 2, n = 2, k = 2;
    double alpha = 1.0, beta = 0.0;
    rocblas_int lda = m, ldb = k, ldc = m;

    // 2. Crear datos en el Host (CPU)
    size_t size_a = lda * k * sizeof(double);
    size_t size_b = ldb * n * sizeof(double);
    size_t size_c = ldc * n * sizeof(double);

    std::vector<double> hA(lda * k, 1.0); // Matriz A llena de 1.0
    std::vector<double> hB(ldb * n, 2.0); // Matriz B llena de 2.0
    std::vector<double> hC(ldc * n, 0.0); // Matriz C inicializada en 0.0

    // 3. Asignar memoria en el Device (GPU)
    double *dA, *dB, *dC;
    hipMalloc(&dA, size_a);
    hipMalloc(&dB, size_b);
    hipMalloc(&dC, size_c);

    // 4. Copiar datos del Host al Device
    hipMemcpy(dA, hA.data(), size_a, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), size_b, hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), size_c, hipMemcpyHostToDevice);

    // 5. Inicializar rocBLAS
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // 6. Ejecutar dgemm en la GPU
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);

    // Esperar a que la GPU termine (opcional si hipMemcpyDeviceToHost va justo después, ya que es bloqueante)
    hipDeviceSynchronize();

    // 7. Traer el resultado de vuelta al Host
    hipMemcpy(hC.data(), dC, size_c, hipMemcpyDeviceToHost);

    // 8. Limpiar memoria y handle
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    rocblas_destroy_handle(handle);

    std::cout << "Ejecución de dgemm completada con éxito." << std::endl;

    return 0;
}
EOF

make scorep_libwrap_rocblas.cc
if [ ! -f scorep_libwrap_rocblas.cc ]; then
    echo "Error: scorep_libwrap_rocblas.cc was not generated."
    exit 1
fi
perl -i -pe 's|.*SCOREP_Libwrap_Plugins.h.*|$&\n\n#include <hip/hip_runtime.h>\n#ifdef SCOREP_LIBWRAP_EXIT_WRAPPED_REGION\n#undef SCOREP_LIBWRAP_EXIT_WRAPPED_REGION\n#endif\n#define SCOREP_LIBWRAP_EXIT_WRAPPED_REGION() do { hipDeviceSynchronize(); SCOREP_LIBWRAP_API( exit_wrapped_region )( scorep_libwrap_var_previous ); } while ( 0 )\n|' scorep_libwrap_rocblas.cc

# Replace `rocblas_status_ return_value = SCOREP_LIBWRAP_ORIGINAL( rocblas_dgemm )( handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );` with:
# `SCOREP_USER_PARAMETER_INT64("rocblas_dgemm::m",m);
# SCOREP_USER_PARAMETER_INT64("rocblas_dgemm::n",n);
# SCOREP_USER_PARAMETER_INT64("rocblas_dgemm::k",k);
# rocblas_status_ return_value = SCOREP_LIBWRAP_ORIGINAL( rocblas_dgemm )( handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );`
sed -i 's/rocblas_status_ return_value = SCOREP_LIBWRAP_ORIGINAL( rocblas_dgemm )( handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );/SCOREP_USER_PARAMETER_INT64("m",m);\nSCOREP_USER_PARAMETER_INT64("n",n);\nSCOREP_USER_PARAMETER_INT64("k",k);\n&/' scorep_libwrap_rocblas.cc

# Add #include <scorep/SCOREP_User.h> to the top
echo "#include <scorep/SCOREP_User.h>" | cat - scorep_libwrap_rocblas.cc > temp && mv temp scorep_libwrap_rocblas.cc

make
make check           # execute tests
make install         # install wrapper
make installcheck    # execute more tests