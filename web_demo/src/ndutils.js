import ndarray from "ndarray";

export function copy(arr) {
    return ndarray(arr.data.slice(), arr.shape);
}

export function getsz(shape) {
    var sz = 1;
    for (var i = 0; i < shape.length; ++i) {
        sz *= shape[i];
    }
    return sz;
}

export function view(ndarr, new_shape) {
    var sz = getsz(ndarr.shape);
    var sd = 1;
    for (var i = new_shape.length - 1; i > 0; --i) {
        sd *= new_shape[i];
    }
    if (new_shape[0] < 0) new_shape[0] = sz / sd;
    ndarr = ndarray(ndarr.data, new_shape);
    return ndarr;
}
