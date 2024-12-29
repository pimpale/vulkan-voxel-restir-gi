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

layout(set = 0, binding = 20, scalar) restrict buffer DebugInfo {
    vec4 debug_info[];
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
         + sr_w_sum[0]
         + debug_info[0].x;

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

void mergeReservoir(
    uint seed,
    inout Reservoir r,
    Reservoir r_new,
    float p_hat
) {
    uint m0 = r.m;
    updateReservoir(seed, r, r_new.z, p_hat*r_new.ucw*r_new.m);
    r.m = m0 + r_new.m;
}

bool geometricallySimilar(Sample a, Sample b) {
    // verify that both the visible points are non-null
    if(length(a.n_v) + length(b.n_v) < 0.1) {
        return false;
    }
    // verify that the normals don't differ by more than 25 degrees
    if(dot(a.n_v, b.n_v) < cos(radians(25.0))) {
        return false;
    }
    // verify that the depths don't differ by more than 0.05
    return true;
}

float luminance(vec3 v) {
    return 0.2126 * v.r + 0.7152 * v.g + 0.0722 * v.b;
}

float p_hat_q(Sample z) {
    return luminance(z.l_o_hat);
}

float computeJacobian() {
    return 1.0;
}

const uint maxIterations = 9;
const float spatialSearchRadius = 20.0;

void main() {
    dummyUse();

    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    // the current pixel
    uvec2 q = gl_GlobalInvocationID.xy;
    uint id = q.x + q.y*xsize;
    uint pixel_seed = murmur3_combine(invocation_seed, id);

    // set of pixels that we are going to merge
    uint nQ = 0;
    uvec2 Q[maxIterations+1];

    // add current pixel to the set
    Q[0] = q;
    nQ++;
    
    // spatial reservoir at the current pixel
    Reservoir R_s = Reservoir(
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

    float old_w_sum = R_s.w_sum;

    // now attempt to add more pixels
    for(uint s = 0; s < maxIterations; s++) {
        uint iter_seed = murmur3_combine(pixel_seed, s);

        // choose a random neighbor pixel
        vec2 jitter = spatialSearchRadius*vec2(
            murmur3_finalizef(murmur3_combine(iter_seed, 0))-0.5,
            murmur3_finalizef(murmur3_combine(iter_seed, 1))-0.5
        );

        uvec2 q_n = uvec2(
            clamp(int(q.x + 0.5 + jitter.x), 0, int(xsize-1)),
            clamp(int(q.y + 0.5 + jitter.y), 0, int(ysize-1))
        );

        // load temporal reservoir at the neighbor pixel
        Reservoir R_n = loadTemporalReservoir(q_n.x + q_n.y*xsize);

        // test geometric similarity
        if(!geometricallySimilar(R_s.z, R_n.z)) {
            continue;
        }

        float jacobian = computeJacobian();

        // compute the target function weight to merge R_n with.
        // this is defined as p_hat_q(R_n.z)/jacobian
        // p_hat_q(R_n.z) is the outgoing radiance at the sample point
        float p_hat_q_adj = p_hat_q(R_n.z) / jacobian;

        // merge the reservoirs
        mergeReservoir(
            murmur3_finalize(murmur3_combine(iter_seed, 2)),
            R_s,
            R_n,
            p_hat_q_adj
        );

        // insert into the set
        Q[nQ] = q_n;
        nQ++;
    }

    uint Z = 0;
    for(uint i = 0; i < nQ; i++) {
        uvec2 q_n = Q[i];
        Reservoir R_n = loadTemporalReservoir(q_n.x + q_n.y*xsize);
        Z += R_n.m;
    }

    R_s.ucw = R_s.w_sum / (Z * p_hat_q(R_s.z));
    storeSpatialReservoir(id, R_s);

    // debug_info[id] = vec4(
    //     vec3(old_w_sum, R_s.w_sum, float(isnan(R_s.w_sum))*10)/10,
    //     1.0
    // );
    debug_info[id] = vec4(
        vec3(R_s.ucw, 0.0, 0.0),
        1.0
    );
}
",
}
