#include <stdint.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static inline void cpu_cpuid(int index, int regs[4]) {
#ifdef _MSC_VER
	__cpuidex(regs, index, 0);
#elif defined(__GNUC__)
	__cpuid_count(index, 0, regs[0], regs[1], regs[2], regs[3]);
#else
#error "Unknown compiler"
#endif
}

static inline uint64_t cpu_xgetbv(int index) {
#if defined(_MSC_VER)
	return _xgetbv(index);
#elif defined(__GNUC__)
	uint32_t eax, edx;
	__asm("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
	return (((uint64_t)edx) << 32) | eax;
#else
#error "Unknown compiler"
#endif
}

static inline int cpu_supports_avx2() {
	int regs[4] = {0};
	cpu_cpuid(1, regs);
	if ((regs[2] & (1 << 27)) && (regs[2] & (1 << 28))) {
		uint64_t xedxeax = cpu_xgetbv(0);
		if ((xedxeax & 0x06) != 0x06)
			return 0; // no support for avx

		int regs[4] = {0};
		cpu_cpuid(7, regs);
		return (regs[1] & (1 << 5)) != 0;
	}
	return 1;
}
