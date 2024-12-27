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

layout(set = 0, binding = 5, scalar) readonly restrict buffer InputTemporalReservoirZSeed {
    uint tr_z_seed[];
};

layout(set = 0, binding = 6, scalar) readonly restrict buffer InputTemporalReservoirW {
    float tr_w[];
};

layout(set = 0, binding = 7, scalar) readonly restrict buffer InputTemporalReservoirM {
    uint tr_m[];
};

layout(set = 0, binding = 8, scalar) readonly restrict buffer InputTemporalReservoirWSum {
    float tr_w_sum[];
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

layout(set = 0, binding = 9, scalar) restrict buffer SpatialReservoirZXV {
    vec3 sr_z_x_v[];
};

layout(set = 0, binding = 10, scalar) restrict buffer SpatialReservoirZNV {
    vec3 sr_z_n_v[];
};

layout(set = 0, binding = 11, scalar) restrict buffer SpatialReservoirZXS {
    vec3 sr_z_x_s[];
};

layout(set = 0, binding = 12, scalar) restrict buffer SpatialReservoirZNS {
    vec3 sr_z_n_s[];
};

layout(set = 0, binding = 13, scalar) restrict buffer SpatialReservoirZLOHat {
    vec3 sr_z_l_o_hat[];
};

layout(set = 0, binding = 14, scalar) restrict buffer SpatialReservoirZSeed {
    uint sr_z_seed[];
};

layout(set = 0, binding = 15, scalar) restrict buffer SpatialReservoirW {
    float sr_w[];
};

layout(set = 0, binding = 16, scalar) restrict buffer SpatialReservoirM {
    uint sr_m[];
};

layout(set = 0, binding = 17, scalar) restrict buffer SpatialReservoirWSum {
    float sr_w_sum[];
};


layout(push_constant, scalar) uniform PushConstants {
    // always zero is kept at zero, but prevents the compiler from optimizing out the buffer
    uint always_zero;
    uint xsize;
    uint ysize;
};

float dummyUse() {
    if(always_zero == 0) {
        return 0;
    }
    return tr_z_x_v[0].x
         + tr_z_n_v[0].x 
         + tr_z_x_s[0].x 
         + tr_z_n_s[0].x 
         + tr_z_l_o_hat[0].x 
         + float(tr_z_seed[0]) 
         + tr_w[0] 
         + tr_m[0] 
         + tr_w_sum[0]
         + sr_z_x_v[0].x
         + sr_z_n_v[0].x 
         + sr_z_x_s[0].x 
         + sr_z_n_s[0].x 
         + sr_z_l_o_hat[0].x 
         + float(sr_z_seed[0]) 
         + sr_w[0] 
         + sr_m[0] 
         + sr_w_sum[0];

}

void main() {
    dummyUse();
}

// void main() {
//     if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
//         return;
//     }
//     const uint srcxsize = xsize * srcscale;
//     const uint srcysize = ysize * srcscale;

//     vec3 color = vec3(0.0);
//     vec3 debug_info = vec3(0.0);
//     for (uint scaley = 0; scaley < srcscale; scaley++) {
//         const uint srcy = gl_GlobalInvocationID.y * srcscale + scaley;
//         for(uint scalex = 0; scalex < srcscale; scalex++) {
//             const uint srcx = gl_GlobalInvocationID.x * srcscale + scalex;
            
//             // compute id of the source pixel
//             const uint id = srcy * srcxsize + srcx;

//             // fetch the color for this sample
//             color += input_outgoing_radiance[id];
//             // fetch the debug info for this sample
//             debug_info += input_debug_info[id].xyz;
//         }
//     }

//     vec3 pixel_color;
//     if (debug_view == 0) {
//         pixel_color = color;
//     } else {
//         pixel_color = debug_info;
//     }

//     // average the samples
//     pixel_color = pixel_color / float(srcscale*srcscale);
//     u8vec4 pixel_data = u8vec4(pixel_color.zyx*255, 255);

//     // write to a patch of size dstscale*dstscale
//     for (uint scaley = 0; scaley < dstscale; scaley++) {
//         const uint dsty = gl_GlobalInvocationID.y * dstscale + scaley;
//         for(uint scalex = 0; scalex < dstscale; scalex++) {
//             const uint dstx = gl_GlobalInvocationID.x * dstscale + scalex;
//             output_image[dsty*xsize*dstscale + dstx] = u8vec4(pixel_color.zyx*255, 255);
//         }
//     }
// }
",
}
