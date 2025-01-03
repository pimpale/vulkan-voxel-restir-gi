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

#define M_PI 3.1415926535897932384626433832795

layout(set = 0, binding = 0, scalar) readonly restrict buffer InputOrigin {
    vec3 input_origin[];
};

layout(set = 0, binding = 1, scalar) readonly restrict buffer InputDirection {
    vec3 input_direction[];
};

layout(set = 0, binding = 2, scalar) readonly restrict buffer InputEmissivity {
    vec3 input_emissivity[];
};

layout(set = 0, binding = 3, scalar) readonly restrict buffer InputReflectivity {
    vec3 input_reflectivity[];
};

layout(set = 0, binding = 4, scalar) readonly restrict buffer InputNeeMisWeight {
    float input_nee_mis_weight[];
};

layout(set = 0, binding = 5, scalar) readonly restrict buffer InputBsdfPdf {
    float input_bsdf_pdf[];
};

layout(set = 0, binding = 6, scalar) readonly restrict buffer InputNeePdf {
    float input_nee_pdf[];
};

layout(set = 0, binding = 7, scalar) restrict buffer SpatialReservoirZXV {
    vec3 sr_z_x_v[];
};

layout(set = 0, binding = 8, scalar) restrict buffer SpatialReservoirZNV {
    vec3 sr_z_n_v[];
};

layout(set = 0, binding = 9, scalar) restrict buffer SpatialReservoirZXS {
    vec3 sr_z_x_s[];
};

layout(set = 0, binding = 10, scalar) restrict buffer SpatialReservoirZNS {
    vec3 sr_z_n_s[];
};

layout(set = 0, binding = 11, scalar) restrict buffer SpatialReservoirZLOHat {
    vec3 sr_z_l_o_hat[];
};

layout(set = 0, binding = 12, scalar) restrict buffer SpatialReservoirZPOmega {
    float sr_z_p_omega[];
};

layout(set = 0, binding = 13, scalar) restrict buffer SpatialReservoirZSeed {
    uint sr_z_seed[];
};

layout(set = 0, binding = 14, scalar) restrict buffer SpatialReservoirUCW {
    float sr_ucw[];
};

layout(set = 0, binding = 15, scalar) restrict buffer SpatialReservoirM {
    uint sr_m[];
};

layout(set = 0, binding = 16, scalar) restrict buffer SpatialReservoirWSum {
    float sr_w_sum[];
};

layout(set = 0, binding = 17, scalar) writeonly restrict buffer OutputOutgoingRadiance {
    vec3 output_outgoing_radiance[];
};

layout(set = 0, binding = 18, scalar) restrict buffer DebugInfo {
    vec3 debug_info[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint always_zero;
    uint xsize;
    uint ysize;
};


float dummyUse(uint always_zero) {
    if(always_zero == 0) {
        return 0.0;
    }
    return input_origin[0].x
        + input_direction[0].x
        + input_emissivity[0].x
        + input_reflectivity[0].x
        + input_nee_mis_weight[0]
        + input_bsdf_pdf[0]
        + input_nee_pdf[0]
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

void main() {
    dummyUse(always_zero);
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    const uint id = y * xsize + x;
    const uint bid = 1 * xsize * ysize + id;

    vec3 dir = normalize(sr_z_x_s[id] - input_origin[bid]);
    vec3 normal = sr_z_n_v[id];

    float bsdf_pdf = dot(dir, normal) / M_PI;

    vec3 outgoing_radiance = input_emissivity[id] + input_reflectivity[id] * sr_z_l_o_hat[id] * sr_ucw[id] * bsdf_pdf;
    output_outgoing_radiance[id] = outgoing_radiance;
    debug_info[id] = vec3(sr_ucw[id]/10);
}
",
}
