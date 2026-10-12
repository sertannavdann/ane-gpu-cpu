// Definitions for statics declared in upstream include/ane_lm/common.h. Upstream defines them in a
// model translation unit we deliberately do not build (it pulls tokenizer/Rust dependencies).
#include <ane_lm/common.h>
namespace ane_lm {
mach_timebase_info_data_t Timer::tb = {0, 0};
bool Timer::tb_init = false;
}
