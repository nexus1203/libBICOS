project(
    'libBICOS', 'cpp', 
    version: '1.0.0',
    license: 'LGPL-3.0-or-later'
)

impl = get_option('implementation')
srcs = files('src/lib.cpp')
incls = include_directories('include')
deps = [dependency('opencv4')]

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

libs = both_libraries(
    'BICOS',
    sources: srcs,
    include_directories: incls,
    dependencies: deps,
    install: true,
    override_options: ['cpp_std=c++20'],
)

executable(
    'bicos-cli',
    sources: 'src/cli.cpp',
    include_directories: incls,
    dependencies: deps,
    link_with: libs,
    install: false,
    override_options: ['cpp_std=c++20'],
)

install_headers(
    'include/config.hpp', 
    'include/match.hpp', 
    subdir: 'BICOS'
)
