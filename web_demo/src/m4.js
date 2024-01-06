var m4 = {
    perspective: function (fieldOfViewInRadians, aspect, near, far) {
        var f = Math.tan(Math.PI * 0.5 - 0.5 * fieldOfViewInRadians);
        var rangeInv = 1.0 / (near - far);

        return new Float32Array([
            f / aspect,
            0,
            0,
            0,
            0,
            f,
            0,
            0,
            0,
            0,
            (near + far) * rangeInv,
            -1,
            0,
            0,
            near * far * rangeInv * 2,
            0,
        ]);
    },

    projection: function (width, height, depth) {
        // Note: This matrix flips the Y axis so 0 is at the top.
        return new Float32Array([
            2 / width,
            0,
            0,
            0,
            0,
            -2 / height,
            0,
            0,
            0,
            0,
            2 / depth,
            0,
            -1,
            1,
            0,
            1,
        ]);
    },

    multiply: function (a, b) {
        var a00 = a[0 * 4 + 0];
        var a01 = a[0 * 4 + 1];
        var a02 = a[0 * 4 + 2];
        var a03 = a[0 * 4 + 3];
        var a10 = a[1 * 4 + 0];
        var a11 = a[1 * 4 + 1];
        var a12 = a[1 * 4 + 2];
        var a13 = a[1 * 4 + 3];
        var a20 = a[2 * 4 + 0];
        var a21 = a[2 * 4 + 1];
        var a22 = a[2 * 4 + 2];
        var a23 = a[2 * 4 + 3];
        var a30 = a[3 * 4 + 0];
        var a31 = a[3 * 4 + 1];
        var a32 = a[3 * 4 + 2];
        var a33 = a[3 * 4 + 3];
        var b00 = b[0 * 4 + 0];
        var b01 = b[0 * 4 + 1];
        var b02 = b[0 * 4 + 2];
        var b03 = b[0 * 4 + 3];
        var b10 = b[1 * 4 + 0];
        var b11 = b[1 * 4 + 1];
        var b12 = b[1 * 4 + 2];
        var b13 = b[1 * 4 + 3];
        var b20 = b[2 * 4 + 0];
        var b21 = b[2 * 4 + 1];
        var b22 = b[2 * 4 + 2];
        var b23 = b[2 * 4 + 3];
        var b30 = b[3 * 4 + 0];
        var b31 = b[3 * 4 + 1];
        var b32 = b[3 * 4 + 2];
        var b33 = b[3 * 4 + 3];
        return new Float32Array([
            b00 * a00 + b01 * a10 + b02 * a20 + b03 * a30,
            b00 * a01 + b01 * a11 + b02 * a21 + b03 * a31,
            b00 * a02 + b01 * a12 + b02 * a22 + b03 * a32,
            b00 * a03 + b01 * a13 + b02 * a23 + b03 * a33,
            b10 * a00 + b11 * a10 + b12 * a20 + b13 * a30,
            b10 * a01 + b11 * a11 + b12 * a21 + b13 * a31,
            b10 * a02 + b11 * a12 + b12 * a22 + b13 * a32,
            b10 * a03 + b11 * a13 + b12 * a23 + b13 * a33,
            b20 * a00 + b21 * a10 + b22 * a20 + b23 * a30,
            b20 * a01 + b21 * a11 + b22 * a21 + b23 * a31,
            b20 * a02 + b21 * a12 + b22 * a22 + b23 * a32,
            b20 * a03 + b21 * a13 + b22 * a23 + b23 * a33,
            b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30,
            b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31,
            b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32,
            b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33,
        ]);
    },

    translation: function (tx, ty, tz) {
        return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1]);
    },

    xRotation: function (angleInRadians) {
        var c = Math.cos(angleInRadians);
        var s = Math.sin(angleInRadians);

        return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
    },

    yRotation: function (angleInRadians) {
        var c = Math.cos(angleInRadians);
        var s = Math.sin(angleInRadians);

        return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
    },

    zRotation: function (angleInRadians) {
        var c = Math.cos(angleInRadians);
        var s = Math.sin(angleInRadians);

        return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    },

    scaling: function (sx, sy, sz) {
        return new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1]);
    },

    translate: function (m, tx, ty, tz) {
        return m4.multiply(m, m4.translation(tx, ty, tz));
    },

    xRotate: function (m, angleInRadians) {
        return m4.multiply(m, m4.xRotation(angleInRadians));
    },

    yRotate: function (m, angleInRadians) {
        return m4.multiply(m, m4.yRotation(angleInRadians));
    },

    zRotate: function (m, angleInRadians) {
        return m4.multiply(m, m4.zRotation(angleInRadians));
    },

    scale: function (m, sx, sy, sz) {
        return m4.multiply(m, m4.scaling(sx, sy, sz));
    },

    inverse: function (m) {
        var r = new Float32Array(16);

        r[0] =
            m[5] * m[10] * m[15] -
            m[5] * m[14] * m[11] -
            m[6] * m[9] * m[15] +
            m[6] * m[13] * m[11] +
            m[7] * m[9] * m[14] -
            m[7] * m[13] * m[10];
        r[1] =
            -m[1] * m[10] * m[15] +
            m[1] * m[14] * m[11] +
            m[2] * m[9] * m[15] -
            m[2] * m[13] * m[11] -
            m[3] * m[9] * m[14] +
            m[3] * m[13] * m[10];
        r[2] =
            m[1] * m[6] * m[15] -
            m[1] * m[14] * m[7] -
            m[2] * m[5] * m[15] +
            m[2] * m[13] * m[7] +
            m[3] * m[5] * m[14] -
            m[3] * m[13] * m[6];
        r[3] =
            -m[1] * m[6] * m[11] +
            m[1] * m[10] * m[7] +
            m[2] * m[5] * m[11] -
            m[2] * m[9] * m[7] -
            m[3] * m[5] * m[10] +
            m[3] * m[9] * m[6];

        r[4] =
            -m[4] * m[10] * m[15] +
            m[4] * m[14] * m[11] +
            m[6] * m[8] * m[15] -
            m[6] * m[12] * m[11] -
            m[7] * m[8] * m[14] +
            m[7] * m[12] * m[10];
        r[5] =
            m[0] * m[10] * m[15] -
            m[0] * m[14] * m[11] -
            m[2] * m[8] * m[15] +
            m[2] * m[12] * m[11] +
            m[3] * m[8] * m[14] -
            m[3] * m[12] * m[10];
        r[6] =
            -m[0] * m[6] * m[15] +
            m[0] * m[14] * m[7] +
            m[2] * m[4] * m[15] -
            m[2] * m[12] * m[7] -
            m[3] * m[4] * m[14] +
            m[3] * m[12] * m[6];
        r[7] =
            m[0] * m[6] * m[11] -
            m[0] * m[10] * m[7] -
            m[2] * m[4] * m[11] +
            m[2] * m[8] * m[7] +
            m[3] * m[4] * m[10] -
            m[3] * m[8] * m[6];

        r[8] =
            m[4] * m[9] * m[15] -
            m[4] * m[13] * m[11] -
            m[5] * m[8] * m[15] +
            m[5] * m[12] * m[11] +
            m[7] * m[8] * m[13] -
            m[7] * m[12] * m[9];
        r[9] =
            -m[0] * m[9] * m[15] +
            m[0] * m[13] * m[11] +
            m[1] * m[8] * m[15] -
            m[1] * m[12] * m[11] -
            m[3] * m[8] * m[13] +
            m[3] * m[12] * m[9];
        r[10] =
            m[0] * m[5] * m[15] -
            m[0] * m[13] * m[7] -
            m[1] * m[4] * m[15] +
            m[1] * m[12] * m[7] +
            m[3] * m[4] * m[13] -
            m[3] * m[12] * m[5];
        r[11] =
            -m[0] * m[5] * m[11] +
            m[0] * m[9] * m[7] +
            m[1] * m[4] * m[11] -
            m[1] * m[8] * m[7] -
            m[3] * m[4] * m[9] +
            m[3] * m[8] * m[5];

        r[12] =
            -m[4] * m[9] * m[14] +
            m[4] * m[13] * m[10] +
            m[5] * m[8] * m[14] -
            m[5] * m[12] * m[10] -
            m[6] * m[8] * m[13] +
            m[6] * m[12] * m[9];
        r[13] =
            m[0] * m[9] * m[14] -
            m[0] * m[13] * m[10] -
            m[1] * m[8] * m[14] +
            m[1] * m[12] * m[10] +
            m[2] * m[8] * m[13] -
            m[2] * m[12] * m[9];
        r[14] =
            -m[0] * m[5] * m[14] +
            m[0] * m[13] * m[6] +
            m[1] * m[4] * m[14] -
            m[1] * m[12] * m[6] -
            m[2] * m[4] * m[13] +
            m[2] * m[12] * m[5];
        r[15] =
            m[0] * m[5] * m[10] -
            m[0] * m[9] * m[6] -
            m[1] * m[4] * m[10] +
            m[1] * m[8] * m[6] +
            m[2] * m[4] * m[9] -
            m[2] * m[8] * m[5];

        var det = m[0] * r[0] + m[1] * r[4] + m[2] * r[8] + m[3] * r[12];
        for (var i = 0; i < 16; i++) r[i] /= det;
        return r;
    },

    transpose: function (m) {
        var r = new Float32Array(16);
        r[0] = m[0];
        r[1] = m[4];
        r[2] = m[8];
        r[3] = m[12];
        r[4] = m[1];
        r[5] = m[5];
        r[6] = m[9];
        r[7] = m[13];
        r[8] = m[2];
        r[9] = m[6];
        r[10] = m[10];
        r[11] = m[14];
        r[12] = m[3];
        r[13] = m[7];
        r[14] = m[11];
        r[15] = m[15];
        return r;
    },
};

export default m4;
