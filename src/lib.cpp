#include "config.hpp"
#include "match.hpp"

#if defined( BICOS_CPU )
#   include "cpu.hpp"
#elif defined( BICOS_CUDA )
#   include "cuda.hpp"
#endif

namespace bicos {

using namespace impl;

void match(
    const std::vector<InputImage>& stack0,
    const std::vector<InputImage>& stack1,
    OutputImage& disparity,
    Config cfg
) {
#if defined( BICOS_CPU )
    match_cpu(stack0, stack1, disparity, cfg);
#elif defined( BICOS_CUDA )
    match_gpu(stack0, stack1, disparity, cfg);
#else
#   error "undefined implementation"
#endif
}

} // namespace bicos
