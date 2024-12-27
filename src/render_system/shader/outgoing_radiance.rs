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

layout(set = 0, binding = 0, scalar) readonly buffer InputOrigin {
    vec3 input_origin[];
};

layout(set = 0, binding = 1, scalar) readonly buffer InputDirection {
    vec3 input_direction[];
};

layout(set = 0, binding = 2, scalar) readonly buffer InputEmissivity {
    vec3 input_emissivity[];
};

layout(set = 0, binding = 3, scalar) readonly buffer InputReflectivity {
    vec3 input_reflectivity[];
};

layout(set = 0, binding = 4, scalar) readonly buffer InputNeeMisWeight {
    float input_nee_mis_weight[];
};

layout(set = 0, binding = 5, scalar) readonly buffer InputBsdfPdf {
    float input_bsdf_pdf[];
};

layout(set = 0, binding = 6, scalar) readonly buffer InputNeePdf {
    float input_nee_pdf[];
};

layout(set = 0, binding = 7, scalar) writeonly buffer OutputOutgoingRadiance {
    vec3 output_outgoing_radiance[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint num_bounces;
    uint xsize;
    uint ysize;
};

void main() {
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;

    // compute the color for this sample
    vec3 outgoing_radiance = vec3(0.0);
    for(int bounce = int(num_bounces)-1; bounce >= 0; bounce--) {            
        // tensor layout: [bounce, y, x, channel]
        const uint bid = bounce * ysize * xsize 
                        + y   * xsize 
                        + x;

        // whether the ray is valid
        float ray_valid = input_direction[bid] == vec3(0.0) ? 0.0 : 1.0;

        // compute importance sampling data
        float bsdf_pdf = input_bsdf_pdf[bid];
        float nee_pdf = input_nee_pdf[bid];
        float nee_mis_weight = input_nee_mis_weight[bid];
        // this is our sampling distribution: 
        // mis_weight proportion of the time, we sample from the light source, and 1-mis_weight proportion of the time, we sample from the BSDF
        float qx = nee_pdf * nee_mis_weight + (1.0 - nee_mis_weight) * bsdf_pdf;
        // this is the distribution we are trying to compute the expectation over
        float px = bsdf_pdf;
        float reweighting_factor = px / qx;

        outgoing_radiance = input_emissivity[bid] + input_reflectivity[bid] * outgoing_radiance * reweighting_factor * ray_valid;
        
        // write to global memory
        output_outgoing_radiance[bid] = outgoing_radiance;
    }
}
",
}
