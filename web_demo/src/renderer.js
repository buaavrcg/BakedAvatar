"use strict";
import { getFragmentShaderSource, getVertexShaderSource } from "./shaders.js";

class Renderer {
    constructor(
        idx,
        vidx,
        vertices,
        faces,
        uvs,
        normals,
        pos_feature,
        tex,
        n_joints,
        n_basis,
        dim_middle
    ) {
        this.idx = idx;
        this.vidx = vidx;
        this.vertices = vertices;
        this.faces = faces;
        this.uvs = uvs;
        this.normals = normals;
        this.pos_feature = pos_feature;
        this.tex = tex;
        this.n_basis = n_basis;
        this.V = vertices.shape[0];
        this.F = faces.shape[0];
        this.J = n_joints;
        this.T = 6;
        this.I = this.tex.length;
        this.F_num = 1.0;
        this.preciseOcclusion = false;
        this.canvas = document.querySelector("#canvas");
        this.gl = this.canvas.getContext("webgl2");
        this.index = new Int32Array(this.V);
        for (var i = 0; i < this.V; i++) {
            this.index[i] = vidx + i;
        }
        this.vertexShaderSource = getVertexShaderSource(this.J);
        this.fragmentShaderSource = getFragmentShaderSource(this.n_basis, dim_middle);
        if (idx == 0) {
            webglUtils.resizeCanvasToDisplaySize(this.gl.canvas);
        }
    }

    render(global) {
        this.setGlobal(global);
        var gl = this.gl;
        if (this.idx == 0) {
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            gl.clearColor(1, 1, 1, 1); //white bg
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.CULL_FACE);
            gl.enable(gl.DEPTH_TEST);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        } else if (this.preciseOcclusion) {
            gl.clear(gl.DEPTH_BUFFER_BIT);
        }

        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);
        gl.uniformMatrix4fv(this.matrixLocation, false, this.vertex_matrix);
        gl.uniformMatrix4fv(this.normalMatrixLocation, false, this.normal_matrix);
        gl.uniformMatrix4fv(this.viewMatrixLocation, false, this.view_matrix);
        gl.uniform1i(this.betasLocation, 0);
        gl.uniform1i(this.shapedirsLocation, 1);
        gl.uniform1i(this.posesLocation, 2);
        gl.uniform1i(this.posedirsLocation, 3);
        gl.uniform1i(this.transformLocation, 4);
        gl.uniform1i(this.lbsweightLocation, 5);
        gl.uniform1i(this.sfcLocation, 6);
        gl.uniform1i(this.posfLocation, 16 + this.idx);
        gl.uniform1i(this.imgLocation, 24 + this.idx);

        var count = Math.floor(this.F * 3 * this.F_num);
        if (this.preciseOcclusion) {
            gl.disable(gl.BLEND);
            gl.depthFunc(gl.LESS);
            gl.colorMask(false, false, false, false);
            gl.uniform1i(this.renderModeLocation, 1);
            gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, 0);
        }
        gl.enable(gl.BLEND);
        gl.depthFunc(gl.LEQUAL);
        gl.colorMask(true, true, true, true);
        gl.uniform1i(
            this.renderModeLocation,
            global.showMesh
                ? 2
                : global.showNormal
                ? 3
                : global.showViewdir
                ? 4
                : global.showCoef
                ? 5
                : this.idx == 0
                ? 6
                : 0
        );
        gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, 0);
    }

    setStatics() {
        var gl = this.gl;

        // Use our boilerplate utils to compile the shaders and link into a program
        this.program = webglUtils.createProgramFromSources(gl, [
            this.vertexShaderSource,
            this.fragmentShaderSource,
        ]);
        var program = this.program;
        // look up where the vertex data needs to go.
        var positionAttributeLocation = gl.getAttribLocation(program, "a_position");
        var indexAttributeLocation = gl.getAttribLocation(program, "a_index");
        var uvsAttributeLocation = gl.getAttribLocation(program, "a_uv");
        var normalsAttributeLocation = gl.getAttribLocation(program, "a_normal");

        this.betasLocation = gl.getUniformLocation(program, "betasTex");
        this.shapedirsLocation = gl.getUniformLocation(program, "shapedirsTex");
        this.posesLocation = gl.getUniformLocation(program, "posesTex");
        this.posedirsLocation = gl.getUniformLocation(program, "posedirsTex");
        this.transformLocation = gl.getUniformLocation(program, "transformTex");
        this.lbsweightLocation = gl.getUniformLocation(program, "lbsweightTex");
        this.posfLocation = gl.getUniformLocation(program, "posfeatTex");
        this.imgLocation = gl.getUniformLocation(program, `radfeatTex`);
        this.sfcLocation = gl.getUniformLocation(program, "sfcTex");
        this.renderModeLocation = gl.getUniformLocation(program, "renderMode");
        // look up uniform locations
        this.matrixLocation = gl.getUniformLocation(program, "u_matrix");
        this.normalMatrixLocation = gl.getUniformLocation(program, "u_normal_matrix");
        this.viewMatrixLocation = gl.getUniformLocation(program, "u_view_matrix");

        // Create a buffer
        var positionBuffer = gl.createBuffer();

        // Create a vertex array object (attribute state)
        this.vao = gl.createVertexArray();

        // and make it the one we're currently working with
        gl.bindVertexArray(this.vao);

        // Turn on the attribute
        gl.enableVertexAttribArray(positionAttributeLocation);

        // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        // Set Geometry.
        gl.bufferData(gl.ARRAY_BUFFER, this.vertices.data, gl.STATIC_DRAW);
        this.vertices = null;

        // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
        var size = 3; // 3 components per iteration
        var type = gl.FLOAT; // the data is 32bit floats
        var normalize = false; // don't normalize the data
        var stride = 0; // 0 = move forward size * sizeof(type) each iteration to get the next position
        var offset = 0; // start at the beginning of the buffer
        gl.vertexAttribPointer(positionAttributeLocation, size, type, normalize, stride, offset);

        var vindexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vindexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.index, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(indexAttributeLocation);
        var size = 1;
        var type = gl.INT;
        var normalize = false;
        var stride = 0;
        var offset = 0;
        gl.vertexAttribPointer(indexAttributeLocation, size, type, normalize, stride, offset);

        var uvsBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, uvsBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.uvs.data, gl.STATIC_DRAW);
        this.uvs = null;
        gl.enableVertexAttribArray(uvsAttributeLocation);
        var size = 2;
        var type = gl.FLOAT;
        var normalize = false;
        var stride = 0;
        var offset = 0;
        gl.vertexAttribPointer(uvsAttributeLocation, size, type, normalize, stride, offset);

        var normalsBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normalsBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.normals.data, gl.STATIC_DRAW);
        this.normals = null;
        gl.enableVertexAttribArray(normalsAttributeLocation);
        var size = 3;
        var type = gl.FLOAT;
        var normalize = false;
        var stride = 0;
        var offset = 0;
        gl.vertexAttribPointer(normalsAttributeLocation, size, type, normalize, stride, offset);

        var indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.faces.data, gl.STATIC_DRAW);
        this.faces = null;

        this.posfTexture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + 16 + this.idx);
        gl.bindTexture(gl.TEXTURE_2D, this.posfTexture);
        var level = 0;
        var internalFormat = gl.RGBA8;
        var height = this.pos_feature.shape[0];
        var width = this.pos_feature.shape[1];
        var border = 0;
        var format = gl.RGBA;
        var type = gl.UNSIGNED_BYTE;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            this.pos_feature.data
        );
        this.pos_feature = null;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        this.imgTexture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + 24 + this.idx);
        gl.bindTexture(gl.TEXTURE_2D, this.imgTexture);
        var level = 0;
        var internalFormat = gl.RGBA8;
        var format = gl.RGBA;
        var type = gl.UNSIGNED_BYTE;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            this.tex.shape[1],
            this.tex.shape[0],
            0,
            format,
            type,
            this.tex.data
        );
        this.tex = null;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        this.betasTexture = gl.createTexture();
        this.posesTexture = gl.createTexture();
        this.transformTexture = gl.createTexture();
        this.sfcTexture = gl.createTexture();
    }

    setShared(shared, shapedirs, posedirs, lbs_weights) {
        if (shared) {
            this.shapedirsTexture = shared.shapedirsTexture;
            this.posedirsTexture = shared.posedirsTexture;
            this.lbsweightTexture = shared.lbsweightTexture;
            return;
        }

        var gl = this.gl;
        this.shapedirsTexture = gl.createTexture();
        this.posedirsTexture = gl.createTexture();
        this.lbsweightTexture = gl.createTexture();

        var height = Math.ceil(shapedirs.shape[0] / 64);
        var width = 50 * 64;
        var shapedata = new Float32Array(height * width * 3);
        shapedata.set(shapedirs.data, 0);
        gl.activeTexture(gl.TEXTURE0 + 1);
        gl.bindTexture(gl.TEXTURE_2D, this.shapedirsTexture);
        var level = 0;
        var internalFormat = gl.RGB32F;
        var border = 0;
        var format = gl.RGB;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            shapedata
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        var height = Math.ceil(posedirs.shape[0] / 64);
        var width = 36 * 64;
        var posedata = new Float32Array(height * width * 3);
        posedata.set(posedirs.data, 0);
        gl.activeTexture(gl.TEXTURE0 + 3);
        gl.bindTexture(gl.TEXTURE_2D, this.posedirsTexture);
        var level = 0;
        var internalFormat = gl.RGB32F;
        var border = 0;
        var format = gl.RGB;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            posedata
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        var height = Math.ceil(lbs_weights.shape[0] / 64);
        var width = this.J * 64;
        var lbswdata = new Float32Array(height * width);
        lbswdata.set(lbs_weights.data, 0);
        gl.activeTexture(gl.TEXTURE0 + 5);
        gl.bindTexture(gl.TEXTURE_2D, this.lbsweightTexture);
        var level = 0;
        var internalFormat = gl.R32F;
        var border = 0;
        var format = gl.RED;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            lbswdata
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }

    setGlobal(global) {
        this.preciseOcclusion = global.preciseOcclusion;
        this.vertex_matrix = global.vertex_matrix;
        this.normal_matrix = global.normal_matrix;
        this.view_matrix = global.view_matrix;

        var gl = this.gl;
        gl.activeTexture(gl.TEXTURE0 + 0);
        gl.bindTexture(gl.TEXTURE_2D, this.betasTexture);
        var level = 0;
        var internalFormat = gl.R32F;
        var width = 50;
        var height = 1;
        var border = 0;
        var format = gl.RED;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            global.betas.data
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.activeTexture(gl.TEXTURE0 + 2);
        gl.bindTexture(gl.TEXTURE_2D, this.posesTexture);
        var level = 0;
        var internalFormat = gl.RGBA32F;
        var width = 9;
        var height = 1;
        var border = 0;
        var format = gl.RGBA;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            global.poses.data
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.activeTexture(gl.TEXTURE0 + 4);
        gl.bindTexture(gl.TEXTURE_2D, this.transformTexture);
        var level = 0;
        var internalFormat = gl.RGBA32F;
        var width = 4;
        var height = this.J;
        var border = 0;
        var format = gl.RGBA;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            global.transform.data
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.activeTexture(gl.TEXTURE0 + 6);
        gl.bindTexture(gl.TEXTURE_2D, this.sfcTexture);
        var level = 0;
        var internalFormat = gl.RGBA32F;
        var height = global.sfc.shape[0];
        var width = global.sfc.shape[1] / 4;
        var border = 0;
        var format = gl.RGBA;
        var type = gl.FLOAT;
        gl.texImage2D(
            gl.TEXTURE_2D,
            level,
            internalFormat,
            width,
            height,
            border,
            format,
            type,
            global.sfc.data
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
}

export default Renderer;
