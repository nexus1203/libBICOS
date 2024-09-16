project('libBICOS', 'cpp', version: '1.0.0', license: 'LGPL-3.0-or-later')

impl = get_option('implementation')
srcs = files('src/lib.cpp')
incls = include_directories('include')
deps = [dependency('opencv4')]

if get_option('buildtype') in ['debug', 'debugoptimized']
    add_project_arguments('-DBICOS_DEBUG', language: ['cpp', 'cuda'])
endif

# complaints about opencv inlines
add_project_arguments('-Wno-deprecated-enum-enum-conversion', language: 'cpp')
add_project_arguments('-Xcompiler=-Wno-deprecated-enum-enum-conversion', language: 'cuda')

if impl == 'cpu'
    add_project_arguments('-DBICOS_CPU', language: 'cpp')
    srcs += files('src/impl/cpu.cpp')
elif impl == 'cuda'
    add_languages('cuda')
    add_project_arguments('-DBICOS_CUDA', language: ['cpp', 'cuda'])
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
    sources: ['src/cli.cpp', 'src/fileutils.cpp'],
    include_directories: incls,
    dependencies: deps,
    link_with: libs,
    install: false,
    override_options: ['cpp_std=c++20'],
)

if impl == 'cuda'
    datdir = join_paths(meson.project_source_root(), 'data')
    srcs = files('src/impl/cuda.cu', 'src/impl/cpu.cpp', 'src/fileutils.cpp')
    incls = [incls, include_directories('test/include')]
    
    test(
        'descriptor_transform_8_32',
        executable(
            'test_descriptor_transform_8_32',
            sources: srcs + 'test/descriptor_transform.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: ['-DBICOS_CUDA', '-DINPUT_TYPE=uint8_t', '-DDESCRIPTOR_TYPE=uint32_t'],
            cpp_args: '-DBICOS_CPU',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        ),
        args: [datdir]
    )
    test(
        'descriptor_transform_8_64',
        executable(
            'test_descriptor_transform_8_64',
            sources: srcs + 'test/descriptor_transform.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: ['-DBICOS_CUDA', '-DINPUT_TYPE=uint8_t', '-DDESCRIPTOR_TYPE=uint64_t'],
            cpp_args: '-DBICOS_CPU',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        ),
        args: [datdir]
    )
    test(
        'descriptor_transform_8_128',
        executable(
            'test_descriptor_transform_8_128',
            sources: srcs + 'test/descriptor_transform.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: ['-DBICOS_CUDA', '-DINPUT_TYPE=uint8_t', '-DDESCRIPTOR_TYPE=uint128_t'],
            cpp_args: '-DBICOS_CPU',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        ),
        args: [datdir]
    )
    test(
        'descriptor_transform_16_32',
        executable(
            'test_descriptor_transform_16_32',
            sources: srcs + 'test/descriptor_transform.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: ['-DBICOS_CUDA', '-DINPUT_TYPE=uint16_t', '-DDESCRIPTOR_TYPE=uint32_t'],
            cpp_args: '-DBICOS_CPU',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        ),
        args: [datdir]
    )
    test(
        'descriptor_transform_16_64',
        executable(
            'test_descriptor_transform_16_64',
            sources: srcs + 'test/descriptor_transform.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: ['-DBICOS_CUDA', '-DINPUT_TYPE=uint16_t', '-DDESCRIPTOR_TYPE=uint64_t'],
            cpp_args: '-DBICOS_CPU',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        ),
        args: [datdir]
    )
    test(
        'descriptor_transform_16_128',
        executable(
            'test_descriptor_transform_16_128',
            sources: srcs + 'test/descriptor_transform.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: ['-DBICOS_CUDA', '-DINPUT_TYPE=uint16_t', '-DDESCRIPTOR_TYPE=uint128_t'],
            cpp_args: '-DBICOS_CPU',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        ),
        args: [datdir]
    )
endif

install_headers('include/config.hpp', 'include/match.hpp', subdir: 'BICOS')