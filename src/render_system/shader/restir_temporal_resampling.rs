vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    vulkan_version: "1.2",
    spirv_version: "1.3",
    src: r"
#version 460
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

// struct Sample {
//     // visible point and surface normal
//     pub x_v: Vec<Subbuffer<[f32]>>, // vec3
//     pub n_v: Vec<Subbuffer<[f32]>>, // vec3
//     // sample point and surface normal
//     pub x_s: Vec<Subbuffer<[f32]>>, // vec3
//     pub n_s: Vec<Subbuffer<[f32]>>, // vec3
//     // outgoing radiance at sample point
//     pub l_o_hat: Vec<Subbuffer<[f32]>>, // vec3
//     // random seed used for sample
//     pub seed: Vec<Subbuffer<[u32]>>, // uint
// }

layout(set = 0, binding = 0, scalar) readonly restrict buffer InputInitialSampleXV {
    vec3 is_x_v[];
};

layout(set = 0, binding = 1, scalar) readonly restrict buffer InputInitialSampleNV {
    vec3 is_n_v[];
};

layout(set = 0, binding = 2, scalar) readonly restrict buffer InputInitialSampleXS {
    vec3 is_x_s[];
};

layout(set = 0, binding = 3, scalar) readonly restrict buffer InputInitialSampleNS {
    vec3 is_n_s[];
};

layout(set = 0, binding = 4, scalar) readonly restrict buffer InputInitialSampleLOHat {
    vec3 is_l_o_hat[];
};

layout(set = 0, binding = 5, scalar) readonly restrict buffer InputInitialSampleSeed {
    uint is_seed[];
};


// struct Reservoir {
//     // the current sample in the reservoir
//     pub z: Sample,
//     // weight of the current sample
//     pub w: Vec<Subbuffer<[f32]>>, // float
//     // number of samples seen so far
//     pub m: Vec<Subbuffer<[u32]>>, // uint
//     // the sum of the weights of all samples
//     pub w_sum: Vec<Subbuffer<[f32]>>, // float
// }

layout(set = 0, binding = 6, scalar) restrict buffer TemporalReservoirZXV {
    vec3 tr_z_x_v[];
};

layout(set = 0, binding = 7, scalar) restrict buffer TemporalReservoirZNV {
    vec3 tr_z_n_v[];
};

layout(set = 0, binding = 8, scalar) restrict buffer TemporalReservoirZXS {
    vec3 tr_z_x_s[];
};

layout(set = 0, binding = 9, scalar) restrict buffer TemporalReservoirZNS {
    vec3 tr_z_n_s[];
};

layout(set = 0, binding = 10, scalar) restrict buffer TemporalReservoirZLOHat {
    vec3 tr_z_l_o_hat[];
};

layout(set = 0, binding = 11, scalar) restrict buffer TemporalReservoirZSeed {
    uint tr_z_seed[];
};

layout(set = 0, binding = 12, scalar) restrict buffer TemporalReservoirUCW {
    float tr_ucw[];
};

layout(set = 0, binding = 13, scalar) restrict buffer TemporalReservoirM {
    uint tr_m[];
};

layout(set = 0, binding = 14, scalar) restrict buffer TemporalReservoirWSum {
    float tr_w_sum[];
};

layout(push_constant, scalar) uniform PushConstants {
    // always zero is kept at zero, but prevents the compiler from optimizing out the buffer
    uint always_zero;
    // the seed for this invocation of the shader
    uint invocation_seed;
    uint xsize;
    uint ysize;
};

// source: https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// accepts a seed, h, and a 32 bit integer, k, and returns a 32 bit integer
// corresponds to the loop in the murmur3 hash algorithm
// the output should be passed to murmur3_finalize before being used
uint murmur3_combine(uint h, uint k) {
    // murmur3_32_scrambleBlBvhNodeBuffer
    k *= 0x1b873593;

    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
    return h;
}

// accepts a seed, h and returns a random 32 bit integer
// corresponds to the last part of the murmur3 hash algorithm
uint murmur3_finalize(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

uint murmur3_combinef(uint h, float k) {
    return murmur3_combine(h, floatBitsToUint(k));
}

float murmur3_finalizef(uint h) {
    return floatConstruct(murmur3_finalize(h));
}

float dummyUse() {
    if(always_zero == 0) {
        return 0;
    }
    return is_x_v[0].x
         + is_n_v[0].x
         + is_x_s[0].x
         + is_n_s[0].x
         + is_l_o_hat[0].x
         + float(is_seed[0])
         + tr_z_x_v[0].x
         + tr_z_n_v[0].x 
         + tr_z_x_s[0].x 
         + tr_z_n_s[0].x 
         + tr_z_l_o_hat[0].x 
         + float(tr_z_seed[0]) 
         + tr_ucw[0] 
         + tr_m[0] 
         + tr_w_sum[0];

}


void zeroReservoir(uint id) {
    tr_w_sum[id] = 0.0;
    tr_m[id] = 0;
}

void updateReservoir(
    // id
    uint id,
    uint r,
    // sample
    vec3 x_v,
    vec3 n_v,
    vec3 x_s,
    vec3 n_s,
    vec3 l_o_hat,
    uint seed,
    // weight of sample
    float w
) {
    tr_w_sum[id] += w;
    tr_m[id] += 1;
    if(floatConstruct(r) < w / tr_w_sum[id]) {
        tr_z_x_v[id] = x_v;
        tr_z_n_v[id] = n_v;
        tr_z_x_s[id] = x_s;
        tr_z_n_s[id] = n_s;
        tr_z_l_o_hat[id] = l_o_hat;
        tr_z_seed[id] = seed;
    }
}

void main() {
    dummyUse();

    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    uint id = gl_GlobalInvocationID.y * xsize + gl_GlobalInvocationID.x;
    
    // for now, we're not doing any temporal resampling

    zeroReservoir(id);
    
    updateReservoir(
        id,
        invocation_seed,
        is_x_v[id],
        is_n_v[id],
        is_x_s[id],
        is_n_s[id],
        is_l_o_hat[id],
        is_seed[id],
        1.0
    );
}
"
}