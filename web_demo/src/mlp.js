import ndarray from "ndarray";
import zeros from "zeros";

class MLP {
    constructor(fc0_w, fc0_b, fc1_w, fc1_b, fc2_w, fc2_b, act0, act1, n_basis) {
        this.fc0_w = fc0_w;
        this.fc0_b = fc0_b;
        this.fc1_w = fc1_w;
        this.fc1_b = fc1_b;
        this.fc2_w = fc2_w;
        this.fc2_b = fc2_b;
        this.act0 = act0;
        this.act1 = act1;

        this.dim_middle = fc2_b.shape[0] / (14 + n_basis);
        this.sfc_dims = [14, this.dim_middle, n_basis];
        this.sfc_height = this.sfc_dims.slice(1).reduce((a, b) => a + b, 0);
        this.sfc_width = this.sfc_dims.slice(0, -1).reduce((a, b) => Math.max(a, b), 0);
        this.sfc = zeros([this.sfc_height, this.sfc_width], "float32");
    }

    linear(x, w, b, a) {
        var out = zeros([w.shape[0]], "float32");
        for (let j = 0; j < w.shape[0]; j++) {
            for (let i = 0; i < w.shape[1]; i++) {
                out.data[j] += w.data[j * w.shape[1] + i] * x[i];
            }
            out.data[j] += b.data[j];
            if (a != null) {
                out.data[j] = Math.max(out.data[j], a.data[j] * out.data[j]);
            }
        }
        return out;
    }

    forward(x) {
        let out0 = this.linear(x, this.fc0_w, this.fc0_b, this.act0);
        let out1 = this.linear(out0.data, this.fc1_w, this.fc1_b, this.act1);
        let out2 = this.linear(out1.data, this.fc2_w, this.fc2_b, null);

        let count = 0;
        let row_count = 0;
        for (let layer = 0; layer < this.sfc_dims.length - 1; layer++) {
            let dim_in = this.sfc_dims[layer];
            let dim_out = this.sfc_dims[layer + 1];
            for (let i = 0; i < dim_out; i++) {
                for (let j = 0; j < dim_in; j++) {
                    this.sfc.data[row_count * this.sfc_width + j] = out2.data[count++];
                }
                row_count++;
            }
        }

        return this.sfc;
    }
}

export default MLP;
