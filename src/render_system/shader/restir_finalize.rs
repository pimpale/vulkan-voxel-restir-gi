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

layout(set = 0, binding = 7, scalar) readonly restrict buffer InputRISWeight {
    float input_ris_weight[];
};

layout(set = 0, binding = 8, scalar) readonly restrict buffer InputSampleL_o_hat {
    vec3 input_sample_l_o_hat[];
};

layout(set = 0, binding = 9, scalar) writeonly restrict buffer OutputOutgoingRadiance {
    vec3 output_outgoing_radiance[];
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
        + input_ris_weight[0]
        + input_sample_l_o_hat[0].x;
}

void main() {
    dummyUse(always_zero);
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    const uint id = y * xsize + x;

    vec3 outgoing_radiance = input_emissivity[id] + input_reflectivity[id] * input_sample_l_o_hat[id] * input_ris_weight[id];
    output_outgoing_radiance[id] = outgoing_radiance;
}
",
}
