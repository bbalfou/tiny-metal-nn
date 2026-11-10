// MSL body template preserving the supplied generated loss helpers, including
// their stop-gradient denominators. RelativeL1 generation intentionally differs
// from ordinary evaluate(): upstream uses prediction + 0.01 without abs here.
// Constants are specialized before compilation.
    constexpr uint kind = @KIND@;
    constexpr uint dimensions = @DIMS@;
    const metal::array<float, dimensions> constant_gradient = {{ @CONSTANTS@ }};
    metal::array<@TYPE@, dimensions> result = {};
    const float normalization = float(n_elements);
    // Every output channel shares this stop-gradient denominator. Compute it
    // once before the literal-width loop; other objectives remove it entirely.
    float luminance_denominator = 1.0f;
    if (kind == 2) {
        float r = float(prediction[0]), g = float(prediction[min(1u, dimensions-1u)]), b = float(prediction[min(2u, dimensions-1u)]);
        if (dimensions >= 6) {
            r += float(prediction[min(3u, dimensions-1u)]); g += float(prediction[min(4u, dimensions-1u)]); b += float(prediction[min(5u, dimensions-1u)]);
        }
        const float luminance = 0.299f*r + 0.587f*g + 0.114f*b;
        luminance_denominator = luminance*luminance + 0.01f;
    }
#pragma unroll
    for (uint row = 0; row < dimensions; ++row) {
        const float p = float(prediction[row]), t = target[row], difference = p-t;
        float loss_value = 0.0f, gradient = 0.0f, denominator = 1.0f;
        if (kind == 1) denominator = p*p + 0.01f;
        if (kind == 2) denominator = luminance_denominator;
        if (kind == 9) {
            gradient = constant_gradient[row] / pdf[row];
        } else if (kind == 4) {
            // The public upstream helper has no nonzero-count precondition.
            // Preserve its IEEE division results even when n_elements is zero.
            const float scale = (1.0f / normalization) / (p + 0.01f) / pdf[row];
            loss_value = abs(difference)*scale;
            gradient = copysign(scale, difference);
        } else if (n_elements != 0) {
            if (kind <= 2) {
                loss_value = difference*difference / denominator / pdf[row] / normalization;
                gradient = 2.0f*difference / denominator / pdf[row] / normalization;
            } else if (kind == 7) {
                const float factor = -t / pdf[row] / normalization;
                loss_value = factor*log(p); gradient = factor/p;
            } else if (kind == 8) {
                const float factor = t*t / pdf[row] / normalization;
                loss_value = factor/p - factor/pdf[row]; gradient = -factor/(p*p);
            } else {
                if (kind == 5) denominator = abs(t) + 0.01f;
                if (kind == 6) denominator = 0.5f*(abs(t)+abs(p)) + 0.01f;
                const float scale = 1.0f / denominator / pdf[row];
                loss_value = abs(difference)*scale / normalization;
                gradient = copysign(scale, difference) / normalization;
            }
        }
        if (value) (*value)[row] = loss_value;
        result[row] = @TYPE@(loss_scale*gradient);
    }
    return result;
