project(
    'libBICOS', 'cpp', 
    version: '1.0.0',
    license: 'LGPL-3.0'
)

impl = get_option('implementation')
srcs = files(
    'src/lib.cpp'
)

if get_option('buildtype') in ['debug', 'debugoptimized']
    add_project_arguments('-DBICOS_DEBUG', language: 'cpp')
endif

# complaints about opencv inlines
add_project_arguments('-Wno-deprecated-enum-enum-conversion', language: 'cpp')

if impl == 'cpu'
    add_project_arguments('-DBICOS_CPU', language: 'cpp')
    srcs += files('src/impl/cpu.cpp')
elif impl == 'cuda'
    add_languages('cuda')
    add_project_arguments('-DBICOS_CUDA', language: 'cpp')
    add_project_arguments('-DBICOS_CUDA', language: 'cuda')
    srcs += files('src/impl/cuda.cu')
else
    error('unsupported implementation')
endif

both_libraries(
    'BICOS',
    sources: srcs,
    include_directories: include_directories('include'),
    dependencies: [
        dependency('opencv4')
    ],
    install: true,
    override_options: ['cpp_std=c++20'],
)

install_headers(
    'include/config.hpp', 
    'include/match.hpp', 
    subdir: 'BICOS'
)
