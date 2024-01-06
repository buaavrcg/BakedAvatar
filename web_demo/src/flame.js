import ndarray from "ndarray";
import zeros from "zeros";
import cwise from "cwise";
import gemm from "ndarray-gemm";
import unsqueeze from "ndarray-unsqueeze";
import ops from "ndarray-ops";
import concatColumns from "ndarray-concat-cols";
import { copy, view } from "./ndutils.js";

var addeq = cwise({
    args: ["array", "array"],
    body: function (a, b) {
        a += b;
    },
});

class FLAME {
    constructor(v_template, shapedirs, posedirs, J_regressor, parents, ghostbone) {
        //console.log("building flame");
        this.v_template = v_template;
        this.shapedirs = shapedirs;
        this.posedirs = posedirs;
        this.J_regressor = J_regressor;
        this.parents = parents;
        this.ghostbone = ghostbone;

        this.blendshapes = zeros([this.shapedirs.shape[0] * 3, 1], "float32");
        this.v_shaped = zeros(this.v_template.shape, "float32");
        this.J = zeros([this.J_regressor.shape[0], this.v_template.shape[1]], "float32");
    }

    lbs(betas, pose) {
        //1. Add shape contribution
        gemm(
            this.blendshapes,
            view(this.shapedirs, [-1, betas.shape[0]]),
            unsqueeze(betas, 0).transpose(1, 0)
        );
        ops.add(
            this.v_shaped,
            this.v_template,
            view(this.blendshapes, [this.shapedirs.shape[0], 3])
        );

        //2. Get the joints
        gemm(this.J, this.J_regressor, this.v_shaped);

        //3. Add pose blend shapes
        var rot_mats = this.rodrigues(view(pose, [-1, 3]));
        var J_nums = rot_mats.shape[0];
        var pose_feature = zeros([J_nums - 1, 3, 3], "float32");
        for (var i = 0; i < J_nums - 1; i++) {
            pose_feature.set(i, 0, 0, -1);
            pose_feature.set(i, 1, 1, -1);
            pose_feature.set(i, 2, 2, -1);
            addeq(pose_feature.pick(i, null, null), rot_mats.pick(1 + i, null, null));
        }
        pose_feature = view(pose_feature, [-1]);
        //4. Get the global joint location
        var A = this.rigid_transform(rot_mats, this.J);

        return {
            ret1: pose_feature, //poses
            ret2: A, //transform
        };
    }

    rodrigues(rot_vecs) {
        var angle = zeros([rot_vecs.shape[0], 1], "float32");
        var sin = zeros([rot_vecs.shape[0]], "float32");
        var one_mi_cos = zeros([rot_vecs.shape[0]], "float32");
        var rot_dir = ndarray(rot_vecs.data.slice(), rot_vecs.shape);
        for (var i = 0; i < rot_vecs.shape[0]; i++) {
            var norm = 0;
            for (var j = 0; j < 3; j++) {
                norm += Math.pow(rot_vecs.get(i, j) + 1e-8, 2);
            }
            norm = Math.sqrt(norm);
            sin.set(i, Math.sin(norm));
            one_mi_cos.set(i, 1 - Math.cos(norm));
            angle.set(i, 0, norm);

            ops.divseq(rot_dir.pick(i, null), norm);
        }

        var rx = rot_dir.pick(null, 0);
        var ry = rot_dir.pick(null, 1);
        var rz = rot_dir.pick(null, 2);
        var rx_m = copy(rx),
            ry_m = copy(ry),
            rz_m = copy(rz);
        ops.muls(rx_m, rx, -1.0);
        ops.muls(ry_m, ry, -1.0);
        ops.muls(rz_m, rz, -1.0);

        var Z = zeros([rot_vecs.shape[0]], "float32");

        var K = view(concatColumns([Z, rz_m, ry, rz, Z, rx_m, ry_m, rx, Z]), [-1, 3, 3]);

        var bmmK = zeros(K.shape, "float32");
        var J_nums = K.shape[0];
        for (var i = 0; i < J_nums; i++) {
            gemm(bmmK.pick(i, null, null), K.pick(i, null, null), K.pick(i, null, null));
        }

        var S = copy(K);
        var C = copy(bmmK);

        for (var i = 0; i < J_nums; i++) {
            ops.mulseq(S.pick(i, null, null), sin.get(i));
            ops.mulseq(C.pick(i, null, null), one_mi_cos.get(i));
        }

        var rot_mat = zeros([J_nums, 3, 3], "float32");
        for (var i = 0; i < J_nums; i++) {
            rot_mat.set(i, 0, 0, 1);
            rot_mat.set(i, 1, 1, 1);
            rot_mat.set(i, 2, 2, 1);
        }

        addeq(rot_mat, S);
        addeq(rot_mat, C);

        return rot_mat;
    }

    rigid_transform(rot_mats, joints) {
        var N = joints.shape[0];
        var rel_joints = copy(joints);
        for (var i = 1; i < N; i++) {
            ops.subeq(rel_joints.pick(i, null), joints.pick(this.parents.get(i), null));
        }
        rel_joints = view(rel_joints, [-1, 3, 1]);
        var transforms_mat = zeros([N, 4, 4], "float32");

        for (var i = 0; i < N; i++) {
            addeq(
                transforms_mat.pick(i, null, null),
                concatColumns([rot_mats.pick(i, null, null), rel_joints.pick(i, null)])
            );
            for (var j = 0; j < 3; j++) transforms_mat.set(i, 3, j, 0);
            transforms_mat.set(i, 3, 3, 1);
        }

        var transform_chain = zeros([N, 4, 4], "float32");
        addeq(transform_chain.pick(0, null, null), transforms_mat.pick(0, null, null));

        for (var i = 1; i < N; i++) {
            gemm(
                transform_chain.pick(i, null, null),
                transform_chain.pick(this.parents.get(i), null, null),
                transforms_mat.pick(i, null, null)
            );
        }

        var posed_joints = zeros([N, 3], "float32");
        for (var i = 0; i < N; i++) {
            addeq(posed_joints.pick(i, null), transform_chain.pick(i, null, 3));
        }
        var joints_homogen = zeros([N, 4], "float32");
        for (var i = 0; i < N; i++) {
            addeq(joints_homogen.pick(i, null), joints.pick(i, null));
            joints_homogen.set(i, 3, 0);
        }
        joints_homogen = view(joints_homogen, [N, 4, 1]);

        var rel_transforms = zeros([N, 4, 4], "float32");
        var rr = zeros([4, 1], "float32");
        for (var i = 0; i < N; i++) {
            gemm(rr, transform_chain.pick(i, null, null), joints_homogen.pick(i, null, null));
            addeq(rel_transforms.pick(i, null, 3), rr);
        }
        ops.sub(rel_transforms, transform_chain, rel_transforms);

        if (this.ghostbone) {
            var nd = new Float32Array((N + 1) * 16);
            nd.set(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), 0);
            nd.set(rel_transforms.data, 16);
            rel_transforms = ndarray(nd, [N + 1, 4, 4]);
        }

        return rel_transforms;
    }
}
export default FLAME;
