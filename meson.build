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

    foreach input_depth : [8, 16]
        foreach descriptor_depth : [32, 64, 128]
            testname = f'descriptor_transform_@input_depth@_@descriptor_depth@'
            test(testname,
                executable('test_' + testname,
                    sources: srcs + 'test/descriptor_transform.cu',
                    dependencies: deps,
                    include_directories: incls,
                    cuda_args: ['-DBICOS_CUDA', f'-DINPUT_TYPE=uint@input_depth@_t', f'-DDESCRIPTOR_TYPE=uint@descriptor_depth@_t'],
                    cpp_args: '-DBICOS_CPU',
                    override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
                )
            )
        endforeach
        foreach agree_cfg : [0, 1]
            if agree_cfg == 1
                testname = f'agree_subpixel_@input_depth@'
            else
                testname = f'agree_@input_depth@'
            endif
            test(testname,
                executable('test_' + testname,
                    sources: srcs + f'test/agree.cu',
                    dependencies: deps,
                    include_directories: incls,
                    cuda_args: ['-DBICOS_CUDA', f'-DINPUT_TYPE=uint@input_depth@_t', f'-DTEST_SUBPIXEL=@agree_cfg@'],
                    cpp_args: '-DBICOS_CPU',
                    override_options: ['cpp_std=c++20', 'cuda_std=c++20']
                )
            )
        endforeach
    endforeach

    foreach descriptor_depth : [32, 64, 128]
        testname = f'bicos_@descriptor_depth@'
        test(testname,
            executable('test_' + testname,
                sources: srcs + 'test/bicos.cu',
                dependencies: deps,
                include_directories: incls,
                cuda_args: ['-DBICOS_CUDA', f'-DDESCRIPTOR_TYPE=uint@descriptor_depth@_t'],
                cpp_args: '-DBICOS_CPU',
                override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
            )
        )
    endforeach

    test('integration',
        executable('test_integration',
            sources: srcs + 'test/integration.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: '-DBICOS_CUDA',
            cpp_args: '-DBICOS_CPU',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        ),
        args: [datdir],
        timeout: 150
    )

    test('integration_raw',
        executable('test_integration_raw',
            sources: srcs + 'test/integration_raw.cu',
            dependencies: deps,
            include_directories: incls,
            cuda_args: '-DBICOS_CUDA',
            cpp_args: '-DBICOS_CPU',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        ),
        args: [datdir],
        timeout: 150
    )
endif

install_headers('include/config.hpp', 'include/match.hpp', subdir: 'BICOS')