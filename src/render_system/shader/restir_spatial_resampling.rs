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

struct Reservoir {
    Sample z;
    float ucw;
    uint m;
    float w_sum;
};

// TEMPORAL RESERVOIR

layout(set = 0, binding = 0, scalar) readonly restrict buffer InputTemporalReservoirZXV {
    vec3 tr_z_x_v[];
};

layout(set = 0, binding = 1, scalar) readonly restrict buffer InputTemporalReservoirZNV {
    vec3 tr_z_n_v[];
};

layout(set = 0, binding = 2, scalar) readonly restrict buffer InputTemporalReservoirZXS {
    vec3 tr_z_x_s[];
};

layout(set = 0, binding = 3, scalar) readonly restrict buffer InputTemporalReservoirZNS {
    vec3 tr_z_n_s[];
};

layout(set = 0, binding = 4, scalar) readonly restrict buffer InputTemporalReservoirZLOHat {
    vec3 tr_z_l_o_hat[];
};

layout(set = 0, binding = 5, scalar) readonly restrict buffer InputTemporalReservoirZPOmega {
    float tr_z_p_omega[];
};

layout(set = 0, binding = 6, scalar) readonly restrict buffer InputTemporalReservoirZSeed {
    uint tr_z_seed[];
};

layout(set = 0, binding = 7, scalar) readonly restrict buffer InputTemporalReservoirUCW {
    float tr_ucw[];
};

layout(set = 0, binding = 8, scalar) readonly restrict buffer InputTemporalReservoirM {
    uint tr_m[];
};

layout(set = 0, binding = 9, scalar) readonly restrict buffer InputTemporalReservoirWSum {
    float tr_w_sum[];
};

// SPATIAL RESERVOIR

layout(set = 0, binding = 10, scalar) restrict buffer SpatialReservoirZXV {
    vec3 sr_z_x_v[];
};

layout(set = 0, binding = 11, scalar) restrict buffer SpatialReservoirZNV {
    vec3 sr_z_n_v[];
};

layout(set = 0, binding = 12, scalar) restrict buffer SpatialReservoirZXS {
    vec3 sr_z_x_s[];
};

layout(set = 0, binding = 13, scalar) restrict buffer SpatialReservoirZNS {
    vec3 sr_z_n_s[];
};

layout(set = 0, binding = 14, scalar) restrict buffer SpatialReservoirZLOHat {
    vec3 sr_z_l_o_hat[];
};

layout(set = 0, binding = 15, scalar) restrict buffer SpatialReservoirZPOmega {
    float sr_z_p_omega[];
};

layout(set = 0, binding = 16, scalar) restrict buffer SpatialReservoirZSeed {
    uint sr_z_seed[];
};

layout(set = 0, binding = 17, scalar) restrict buffer SpatialReservoirUCW {
    float sr_ucw[];
};

layout(set = 0, binding = 18, scalar) restrict buffer SpatialReservoirM {
    uint sr_m[];
};

layout(set = 0, binding = 19, scalar) restrict buffer SpatialReservoirWSum {
    float sr_w_sum[];
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
    return tr_z_x_v[0].x
         + tr_z_n_v[0].x 
         + tr_z_x_s[0].x 
         + tr_z_n_s[0].x 
         + tr_z_l_o_hat[0].x 
         + tr_z_p_omega[0]
         + float(tr_z_seed[0]) 
         + tr_ucw[0] 
         + tr_m[0] 
         + tr_w_sum[0]
         + sr_z_x_v[0].x
         + sr_z_n_v[0].x 
         + sr_z_x_s[0].x 
         + sr_z_n_s[0].x 
         + sr_z_l_o_hat[0].x 
         + sr_z_p_omega[0]
         + float(sr_z_seed[0]) 
         + sr_ucw[0] 
         + sr_m[0] 
         + sr_w_sum[0];

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


Reservoir loadSpatialReservoir(uint id) {
    return Reservoir(
        Sample(
            sr_z_x_v[id],
            sr_z_n_v[id],
            sr_z_x_s[id],
            sr_z_n_s[id],
            sr_z_l_o_hat[id],
            sr_z_p_omega[id],
            sr_z_seed[id]
        ),
        sr_ucw[id],
        sr_m[id],
        sr_w_sum[id]
    );
}

void storeSpatialReservoir(uint id, Reservoir r) {
    sr_z_x_v[id] = r.z.x_v;
    sr_z_n_v[id] = r.z.n_v;
    sr_z_x_s[id] = r.z.x_s;
    sr_z_n_s[id] = r.z.n_s;
    sr_z_l_o_hat[id] = r.z.l_o_hat;
    sr_z_p_omega[id] = r.z.p_omega;
    sr_z_seed[id] = r.z.seed;
    sr_ucw[id] = r.ucw;
    sr_m[id] = r.m;
    sr_w_sum[id] = r.w_sum;
}

bool similarEnough(uvec2 a, uvec2 b) {
    return true;
}

const uint maxIterations = 3;
const float spatialSearchRadius = 10.0;

void main() {
    dummyUse();

    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    // the current pixel
    uvec2 q = gl_GlobalInvocationID.xy; 


    // set of pixels that we are going to merge
    uint nQ = 0;
    uvec2 Q[maxIterations+1];

    // add current pixel to the set
    Q[0] = q;
    nQ++;

    // now attempt to add more pixels
    for(uint s = 0; s < maxIterations; s++) {
        uint iter_seed = murmur3_combine(invocation_seed, s);

        // choose a random neighbor pixel
        vec2 jitter = spatialSearchRadius*vec2(
            murmur3_finalizef(murmur3_combine(iter_seed, 0))-0.5,
            murmur3_finalizef(murmur3_combine(iter_seed, 1))-0.5
        );

        uvec2 q_n = uvec2(
            clamp(int(Q[s].x + jitter.x), 0, int(xsize-1)),
            clamp(int(Q[s].y + jitter.y), 0, int(ysize-1))
        );

        // test geometric similarity
        if(!similarEnough(q, q_n)) {
            continue;
        }

    }
}
",
}
