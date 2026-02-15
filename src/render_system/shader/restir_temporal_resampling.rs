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


struct Sample {
    vec3 x_v;
    vec3 n_v;
    vec3 x_s;
    vec3 n_s;
    vec3 l_o_hat;
    float p_omega;
    uint seed;
};


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

layout(set = 0, binding = 5, scalar) readonly restrict buffer InputInitialSamplePOmega {
    float is_p_omega[];
};

layout(set = 0, binding = 6, scalar) readonly restrict buffer InputInitialSampleSeed {
    uint is_seed[];
};

// struct Reservoir {
//     // the current sample in the reservoir
//     pub z: Sample,
//     // weight of the current sample
//     pub ucw: Vec<Subbuffer<[f32]>>, // float
//     // number of samples seen so far
//     pub m: Vec<Subbuffer<[u32]>>, // uint
//     // the sum of the weights of all samples
//     pub w_sum: Vec<Subbuffer<[f32]>>, // float
// }

struct Reservoir {
    Sample z;
    float ucw;
    uint m;
    float w_sum;
};

layout(set = 0, binding = 7, scalar) restrict buffer TemporalReservoirZXV {
    vec3 tr_z_x_v[];
};

layout(set = 0, binding = 8, scalar) restrict buffer TemporalReservoirZNV {
    vec3 tr_z_n_v[];
};

layout(set = 0, binding = 9, scalar) restrict buffer TemporalReservoirZXS {
    vec3 tr_z_x_s[];
};

layout(set = 0, binding = 10, scalar) restrict buffer TemporalReservoirZNS {
    vec3 tr_z_n_s[];
};

layout(set = 0, binding = 11, scalar) restrict buffer TemporalReservoirZLOHat {
    vec3 tr_z_l_o_hat[];
};

layout(set = 0, binding = 12, scalar) restrict buffer TemporalReservoirZPOmega {
    float tr_z_p_omega[];
};

layout(set = 0, binding = 13, scalar) restrict buffer TemporalReservoirZSeed {
    uint tr_z_seed[];
};

layout(set = 0, binding = 14, scalar) restrict buffer TemporalReservoirUCW {
    float tr_ucw[];
};

layout(set = 0, binding = 15, scalar) restrict buffer TemporalReservoirM {
    uint tr_m[];
};

layout(set = 0, binding = 16, scalar) restrict buffer TemporalReservoirWSum {
    float tr_w_sum[];
};

layout(set = 0, binding = 17, scalar) restrict buffer OutputDebugInfo {
    vec3 debug_info[];
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
         + is_p_omega[0]
         + float(is_seed[0])
         + tr_z_x_v[0].x
         + tr_z_n_v[0].x 
         + tr_z_x_s[0].x 
         + tr_z_n_s[0].x 
         + tr_z_l_o_hat[0].x 
         + float(tr_z_seed[0]) 
         + tr_ucw[0] 
         + tr_m[0] 
         + tr_w_sum[0]
         + debug_info[0].x;
}

Sample loadInitialSample(uint id) {
    return Sample(
        is_x_v[id],
        is_n_v[id],
        is_x_s[id],
        is_n_s[id],
        is_l_o_hat[id],
        is_p_omega[id],
        is_seed[id]
    );
}

Reservoir loadTemporalReservoir(uint id) {
    return Reservoir(
        Sample(
            tr_z_x_v[id],
            tr_z_n_v[id],
            tr_z_x_s[id],
            tr_z_n_s[id],
            tr_z_l_o_hat[id],
            tr_z_p_omega[id],
            tr_z_seed[id]
        ),
        tr_ucw[id],
        tr_m[id],
        tr_w_sum[id]
    );
}

void storeTemporalReservoir(uint id, Reservoir r) {
    tr_z_x_v[id] = r.z.x_v;
    tr_z_n_v[id] = r.z.n_v;
    tr_z_x_s[id] = r.z.x_s;
    tr_z_n_s[id] = r.z.n_s;
    tr_z_l_o_hat[id] = r.z.l_o_hat;
    tr_z_p_omega[id] = r.z.p_omega;
    tr_z_seed[id] = r.z.seed;
    tr_ucw[id] = r.ucw;
    tr_m[id] = r.m;
    tr_w_sum[id] = r.w_sum;
}

void updateReservoir(
    uint seed,
    inout Reservoir r,
    Sample z,
    float w_new
) {
    r.w_sum += w_new;
    r.m += 1;
    if(floatConstruct(seed) < w_new / r.w_sum) {
        r.z = z;
    }
}

float luminance(vec3 v) {
    return 0.2126 * v.r + 0.7152 * v.g + 0.0722 * v.b;
}

float p_hat_q(Sample S) {
    return luminance(S.l_o_hat);
}

void main() {
    dummyUse();
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    uint id = gl_GlobalInvocationID.y * xsize + gl_GlobalInvocationID.x;
    
    uint pixel_seed = murmur3_combine(invocation_seed, id);

    Sample S = loadInitialSample(id);
    Reservoir R = loadTemporalReservoir(id);
    // for now, we're not doing any temporal resampling, so we re-initialize the reservoir
    if(murmur3_finalizef(murmur3_combine(pixel_seed, 0)) < 0.4) {
        R.w_sum = 0.0;
        R.m = 0;
    }
    
    // compute the RIS weight to insert S with
    // this is defined as p_hat_q(S) / p_q(S)
    // * p_hat_q(S) is the target function, and is set to L_o(x_s, -\omega_i). 
    //   e.g, the outgoing radiance at the sample point
    //   This is given in the sample as l_o_hat
    // * p_q(S) is the actual sampling PDF at the visible point 
    //   This is given in the sample as p_omega 
    
    const float p_q = S.p_omega;
    const float w = p_hat_q(S) / p_q;

    // update the reservoir
    updateReservoir(
        murmur3_finalize(murmur3_combine(pixel_seed, 1)),
        R,
        S,
        w
    );

    // now compute the unbiased contribution weight for the sample
    float p_hat_R_z = p_hat_q(R.z);
    if(p_hat_R_z > 0) {
       R.ucw = R.w_sum / (R.m * p_hat_R_z);
    } else {
        R.ucw = 0.0;
    }
    storeTemporalReservoir(id, R);
}
"
}
