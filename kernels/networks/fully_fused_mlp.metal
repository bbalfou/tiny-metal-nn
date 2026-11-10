// Baseline FullyFusedMLP: one GPU invocation traverses the entire model for each
// sample. Parameter gradients use canonical split-K kernels. No CPU data
// fallback is used by the Metal backend. All bindings are frozen at ABI version 1.
#include "fused_mlp_math.metal"

kernel void fused_mlp_forward(
    device const uchar* x [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device uchar* pre [[buffer(2)]],
    device uchar* hidden [[buffer(3)]],
    constant FusedArgs& a [[buffer(4)]],
    uint sample [[thread_position_in_grid]]
) {
    fused_forward_sample(
        a,
        x,
        weights,
        pre,
        hidden,
        sample
    );
}

kernel void fused_mlp_backward(
    device const uchar* weights [[buffer(0)]],
    device const uchar* pre [[buffer(1)]],
    device const uchar* hidden [[buffer(2)]],
    device const uchar* output [[buffer(3)]],
    device const uchar* incoming [[buffer(4)]],
    device uchar* delta [[buffer(5)]],
    device uchar* dx [[buffer(6)]],
    constant FusedArgs& a [[buffer(7)]],
    uint sample [[thread_position_in_grid]]
) {
    fused_backward_sample(
        a,
        weights,
        pre,
        hidden,
        output,
        incoming,
        delta,
        dx,
        sample
    );
}

kernel void fused_mlp_tangent(
    device const uchar* direction [[buffer(0)]],
    device const uchar* weights [[buffer(1)]],
    device const uchar* pre [[buffer(2)]],
    device const uchar* hidden [[buffer(3)]],
    device uchar* tangent_pre [[buffer(4)]],
    device uchar* tangent [[buffer(5)]],
    device uchar* ddy [[buffer(6)]],
    constant FusedArgs& a [[buffer(7)]],
    uint sample [[thread_position_in_grid]]
) {
    fused_tangent_sample(
        a,
        direction,
        weights,
        pre,
        hidden,
        tangent_pre,
        tangent,
        ddy,
        sample
    );
}

kernel void fused_mlp_second_backward(
    device const uchar* weights [[buffer(0)]],
    device const uchar* pre [[buffer(1)]],
    device const uchar* hidden [[buffer(2)]],
    device const uchar* tangent_pre [[buffer(3)]],
    device const uchar* incoming [[buffer(4)]],
    device uchar* delta [[buffer(5)]],
    device uchar* tangent_delta [[buffer(6)]],
    device uchar* dx [[buffer(7)]],
    constant FusedArgs& a [[buffer(8)]],
    uint sample [[thread_position_in_grid]]
) {
    fused_second_backward_sample(
        a,
        weights,
        pre,
        hidden,
        tangent_pre,
        incoming,
        delta,
        tangent_delta,
        dx,
        sample
    );
}
