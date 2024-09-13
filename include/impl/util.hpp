#pragma once

#define STACKALLOC(n, type) (type*)alloca(n * sizeof(type))
