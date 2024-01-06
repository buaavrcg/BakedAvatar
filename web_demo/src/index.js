import npyjs from "npyjs";
import ndarray from "ndarray";
import zeros from "zeros";
import concatRows from "ndarray-concat-rows";
import ops from "ndarray-ops";
import FLAME from "./flame.js";
import Renderer from "./renderer.js";
import MLP from "./mlp.js";
import m4 from "./m4.js";
import { copy } from "./ndutils.js";

let n = new npyjs();

async function loadnpy(url) {
    var npy = await n.load(url);
    return ndarray(new Float32Array(npy.data), npy.shape);
}
async function loadnpyu(url) {
    var npy = await n.load(url);
    return ndarray(new Uint32Array(npy.data), npy.shape);
}
async function loadnpyu8(url) {
    var npy = await n.load(url);
    return ndarray(new Uint8Array(npy.data), npy.shape);
}
function degToRad(d) {
    return (d * Math.PI) / 180;
}

window.onload = async function () {
    this.canvas = document.querySelector("#canvas");
    this.gl = this.canvas.getContext("webgl2");
    if (!gl) {
        alert("WebGL isn't available on your broswer!");
        return;
    }
    var timeText = document.getElementById("timeText");
    var meshText = document.getElementById("meshText");

    let searchParams = new URLSearchParams(window.location.search);
    if (searchParams.has("width"))
        this.canvas.width = searchParams.get("width");
    if (searchParams.get("height"))
        this.canvas.height = searchParams.get("height");
    var meshRoot = searchParams.has("mesh_data")
        ? searchParams.get("mesh_data") + "/"
        : "mesh_data/";
    await fetch(meshRoot + "metadata.json")
        .then((response) => response.json())
        .then((json) => (this.metadata = json));
    this.M = this.metadata["meshes"].length;
    this.N_basis = this.metadata["num_texture_basis"];
    this.ghostbone = this.metadata["ghostbone"];

    var J_regressor_c = await loadnpy(meshRoot + "common/J_regressor.npy");
    var parents_c = await loadnpy(meshRoot + "common/parents.npy");
    var v_template = await loadnpy(meshRoot + "common/v_template.npy");
    var shapedirs_c = await loadnpy(meshRoot + "common/shapedirs.npy");
    var posedirs_c = await loadnpy(meshRoot + "common/posedirs.npy");
    this.flame = new FLAME(
        v_template,
        shapedirs_c,
        posedirs_c,
        J_regressor_c,
        parents_c,
        this.ghostbone
    );
    timeText.innerHTML = "common loaded\n";

    var fc0_weight = await loadnpy(meshRoot + "global_mlp/fc0_weight.npy");
    var fc0_bias = await loadnpy(meshRoot + "global_mlp/fc0_bias.npy");
    var fc1_weight = await loadnpy(meshRoot + "global_mlp/fc1_weight.npy");
    var fc1_bias = await loadnpy(meshRoot + "global_mlp/fc1_bias.npy");
    var fc2_weight = await loadnpy(meshRoot + "global_mlp/fc2_weight.npy");
    var fc2_bias = await loadnpy(meshRoot + "global_mlp/fc2_bias.npy");
    var act0 = await loadnpy(meshRoot + "global_mlp/act0_weight.npy");
    var act1 = await loadnpy(meshRoot + "global_mlp/act1_weight.npy");
    this.global_mlp = new MLP(
        fc0_weight,
        fc0_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        act0,
        act1,
        this.N_basis
    );
    this.global_input = new Float32Array(fc0_weight.shape[1]);
    timeText.innerHTML += "MLP loaded\n";
    this.betas = zeros([50], "float32");
    this.betas_bias = zeros([50], "float32");
    this.betas_conon = ndarray(
        new Float32Array([
            -0.7734001874923706, 0.47500404715538025, -0.2207159399986267, 0.4112471044063568,
            -0.7225434184074402, 0.7391737103462219, 0.048007089644670486, -0.17434833943843842,
            -0.03906625136733055, 0.6934881210327148, -0.04477059468626976, -0.3222707509994507,
            -0.4588749408721924, 0.7728955745697021, 0.11826960742473602, 0.10405569523572922,
            -0.3743153512477875, -0.1533493995666504, 0.10482098162174225, 0.23136195540428162,
            -0.14709198474884033, -0.17408138513565063, 0.15598450601100922, -0.3476805090904236,
            -0.1309555321931839, -0.06102199852466583, 0.1289907991886139, 0.03520803898572922,
            -0.1652863621711731, -0.22350919246673584, -0.2152254283428192, -0.00794155988842249,
            0.17952083051204681, -0.08767711371183395, -0.05959964171051979, -0.07291064411401749,
            0.10973446816205978, -0.15915675461292267, 0.042276158928871155, -0.007540557067841291,
            -0.10525673627853394, 0.0052323173731565475, -0.026331013068556786, 0.00783122144639492,
            -0.12349570542573929, -0.10160142928361893, -0.08464774489402771, 0.09305692464113235,
            -0.029000310227274895, 0.026307053864002228,
        ]),
        [50]
    );
    this.betas = copy(this.betas_conon);
    this.pose_params = zeros([15], "float32");
    this.pose_params_bias = zeros([15], "float32");
    this.pose_params_conon = ndarray(
        new Float32Array([
            0.12880776822566986, -0.0023364718072116375, -0.042034655809402466,
            -0.10783462971448898, -0.015666034072637558, 0.002099345438182354, 0.012164629995822906,
            0.012352745980024338, -0.012653038837015629, -0.06936541199684143, -0.10923106968402863,
            0.009163151495158672, -0.06980134546756744, 0.23808681964874268, 0.00034914835123345256,
        ]),
        [15]
    );
    this.pose_params = copy(this.pose_params_conon);
    this.onBias = 0;

    this.shapedirs = [];
    this.posedirs = [];
    this.lbs_weights = [];
    this.renderers = [];
    this.total_V = 0;
    this.total_F = 0;
    for (let i = 0; i < this.M; i++) {
        this.shapedirs.push(await loadnpy(meshRoot + `meshes_${i}/shapedirs.npy`));
        this.posedirs.push(await loadnpy(meshRoot + `meshes_${i}/posedirs.npy`));
        this.lbs_weights.push(await loadnpy(meshRoot + `meshes_${i}/lbs_weights.npy`));
        this.renderers.push(
            new Renderer(
                i,
                this.total_V,
                await loadnpy(meshRoot + `meshes_${i}/vertices.npy`),
                await loadnpyu(meshRoot + `meshes_${i}/faces.npy`),
                await loadnpy(meshRoot + `meshes_${i}/uvs.npy`),
                await loadnpy(meshRoot + `meshes_${i}/normals.npy`),
                await loadnpyu8(meshRoot + `meshes_${i}/position_texture.npy`),
                await loadnpyu8(meshRoot + `meshes_${i}/radiance_texture.npy`),
                this.lbs_weights[0].shape[1],
                this.N_basis,
                this.global_mlp.dim_middle
            )
        );
        this.renderers[i].setStatics();
        this.total_V += this.renderers[i].V;
        this.total_F += this.renderers[i].F;
    }
    meshText.innerHTML += `Vertices: ${this.total_V}\nFaces: ${this.total_F}`;
    // concat all FLAME weights
    this.shapedirs = concatRows(this.shapedirs);
    this.posedirs = concatRows(this.posedirs);
    this.lbs_weights = concatRows(this.lbs_weights);
    this.max_M = this.M;
    this.preciseOcclusion = false;
    this.showNormal = false;
    timeText.innerHTML += "meshes loaded\n";

    this.seq_data = await loadnpy("sequence_data/flame_sequences.npy");
    await fetch("sequence_data/flame_sequences.json")
        .then((response) => response.json())
        .then((json) => (this.seq_json = json));
    this.stframes = [0];
    this.edframes = [0];
    this.aninames = [""];
    this.anis = 0;
    for (var ani in this.seq_json) {
        this.anis++;
        this.aninames.push(ani);
        var ani_frame = this.seq_json[this.aninames[this.anis]];
        this.stframes.push(ani_frame.start);
        this.edframes.push(ani_frame.end);
    }
    timeText.innerHTML += "animation loaded\n";

    await forward(this);
    for (let i = 0; i < this.M; i++) {
        this.renderers[i].setShared(
            i > 0 ? this.renderers[0] : null,
            this.shapedirs,
            this.posedirs,
            this.lbs_weights
        );
    }
    uiInit(this);
};

async function forward(global) {
    var retVal = await global.flame.lbs(global.betas, global.pose_params);
    global.poses = retVal.ret1;
    global.transform = retVal.ret2;
    let pose_length = global.global_input.length - 50;
    for (let i = 0; i < pose_length; i++) global.global_input[i] = global.pose_params.data[6 + i];
    global.global_input.set(global.betas.data, pose_length);
    global.sfc = await global.global_mlp.forward(global.global_input);
}

function uiInit(global) {
    var frameCount = 0;
    var sandbox = 0;

    var smoothing = 0.2;
    var startplay = 0,
        onani_ = 0;
    var stframe = 0,
        edframe = -1,
        current = 0;
    var lastTime = performance.now();
    var gl = global.gl;
    var timeText = document.getElementById("timeText");
    var resText = document.getElementById("resText");
    timeText.innerHTML = "";

    var translation = [0, 0, -900];
    var rotation = [0, 0, 0];
    var rotationLimit = degToRad(15);

    $("#control_block")[0].style.display = "inline-block";
    $("#precise_occlusion")[0].onclick = function () {
        global.preciseOcclusion = $("#precise_occlusion")[0].checked;
    };
    // click the checkbox
    $("#precise_occlusion")[0].click();
    $("#show_mesh")[0].onclick = function () {
        global.showMesh = $("#show_mesh")[0].checked;
        global.showNormal = $("#show_normal")[0].checked = false;
        global.showViewdir = $("#show_viewdir")[0].checked = false;
        global.showCoef = $("#show_coef")[0].checked = false;
    };
    $("#show_normal")[0].onclick = function () {
        global.showNormal = $("#show_normal")[0].checked;
        global.showMesh = $("#show_mesh")[0].checked = false;
        global.showViewdir = $("#show_viewdir")[0].checked = false;
        global.showCoef = $("#show_coef")[0].checked = false;
    };
    $("#show_viewdir")[0].onclick = function () {
        global.showViewdir = $("#show_viewdir")[0].checked;
        global.showNormal = $("#show_normal")[0].checked = false;
        global.showMesh = $("#show_mesh")[0].checked = false;
        global.showCoef = $("#show_coef")[0].checked = false;
    };
    $("#show_coef")[0].onclick = function () {
        global.showCoef = $("#show_coef")[0].checked;
        global.showNormal = $("#show_normal")[0].checked = false;
        global.showViewdir = $("#show_viewdir")[0].checked = false;
        global.showMesh = $("#show_mesh")[0].checked = false;
    };
    webglLessonsUI.setupSlider("#M_num", {
        value: global.M,
        slide: (e, ui) => (global.max_M = Math.round(ui.value)),
        min: 1,
        max: global.M,
        step: 1,
    });
    webglLessonsUI.setupSlider("#F_num", {
        value: 1,
        slide: updateF(),
        min: 0,
        max: 1,
        step: 0.0001,
        precision: 4,
    });
    webglLessonsUI.setupSlider("#smoothing", {
        value: smoothing,
        slide: (e, ui) => (smoothing = ui.value),
        min: 0.0,
        max: 0.8,
        step: 0.01,
        precision: 2,
    });
    var is_mobile = isMobile();
    console.log("mobile:", is_mobile);
    if (is_mobile) {
        $("#uiContainer")[0].style.marginTop = "300px";
        $("#uiContainer")[0].style.marginBottom = "50px";
    }
    for (let i = 1; i <= global.anis; i++) {
        $("#btn_field")[0].innerHTML += `
        <div class="col waves-effect waves-light btn white-text skyblue ani" id="ani${i}">
                <div class="white-text aniname">${global.aninames[i]}</div>
                <div class="abv" id="abv${i}" style="background-color: darkblue;"></div>
            </div>
        `;
    }
    $("#sandbox")[0].onclick = function () {
        sandbox = sandbox ^ 1;
        if (sandbox) {
            $("#sandbox")[0].className = $("#sandbox")[0].className.replace("darken-1", "darken-4");
            $("#biasBtn")[0].className = $("#biasBtn")[0].className.replace(
                "darken-4",
                "lighten-1"
            );
            global.onBias = 0;
            uiSync();
            $("#uiContainer")[0].style.display = "inline-block";
        } else {
            $("#sandbox")[0].className = $("#sandbox")[0].className.replace("darken-4", "darken-1");
            $("#uiContainer")[0].style.display = "none";
        }
    };
    $("#resetBtn")[0].onclick = async function () {
        global.betas = copy(global.betas_conon);
        global.pose_params = copy(global.pose_params_conon);
        ops.subeq(global.betas_bias, global.betas_bias);
        ops.subeq(global.pose_params_bias, global.pose_params_bias);
        if (onani_) $(`#abv${onani_}`)[0].style.width = "0px";
        onani_ = 0;
        current = 0;
        edframe = -1;
        await forward(global);
        translation = [0, 0, -900];
        rotation = [0, 0, 0];
        uiSync();
    };
    $("#biasBtn")[0].onclick = function () {
        global.onBias = global.onBias ^ 1;
        if (global.onBias) {
            $("#biasBtn")[0].className = $("#biasBtn")[0].className.replace(
                "lighten-1",
                "darken-4"
            );
            $("#sandbox")[0].className = $("#sandbox")[0].className.replace("darken-4", "darken-1");
            sandbox = 0;
            uiSync();
            $("#uiContainer")[0].style.display = "inline-block";
        } else {
            $("#biasBtn")[0].className = $("#biasBtn")[0].className.replace(
                "darken-4",
                "lighten-1"
            );
            $("#uiContainer")[0].style.display = "none";
        }
    };
    for (let i = 1; i <= global.anis; i++) {
        $(`#ani${i}`)[0].onclick = function () {
            if (onani_ == i && edframe != -1) {
                onani_ = 0;
                current = 0;
                edframe = -1;
                $(`#abv${i}`)[0].style.width = "0px";
                return;
            }
            onani_ = i;
            for (let j = 1; j <= global.anis; j++)
                if (j != i) {
                    $(`#abv${j}`)[0].style.width = "0px";
                }
            current = 0;
            stframe = global.stframes[i];
            edframe = global.edframes[i];
            startplay = 1;
        };
    }
    for (var i = 0; i < 50; i++)
        webglLessonsUI.setupSlider("#exp" + (i + 1), {
            value: global.onBias ? global.betas_bias.data[i] : global.betas.data[i],
            slide: updateBetas(i),
            step: 0.001,
            min: -2,
            max: 2,
            precision: 3,
        });

    for (var i = 0; i < 15; i++)
        webglLessonsUI.setupSlider("#pose" + (i + 1), {
            value: global.onBias ? global.pose_params_bias.data[i] : global.pose_params.data[i],
            slide: updatePoses(i),
            step: 0.001,
            min: -1,
            max: 1,
            precision: 3,
        });
    function updateBetas(i) {
        return async function (e, ui) {
            if (global.onBias) global.betas_bias.set(i, ui.value);
            else {
                global.betas.set(i, ui.value);
                await forward(global);
            }
            var x = Math.floor((ui.value / 2) * 255);
            $(".gman-widget-outer")[i].style.backgroundColor = rgbaToHex([
                255 - Math.max(-x, 0),
                255 - Math.abs(x),
                255 - Math.max(x, 0),
                128,
            ]);
        };
    }
    function updatePoses(i) {
        return async function (e, ui) {
            if (global.onBias) global.pose_params_bias.set(i, ui.value);
            else {
                global.pose_params.set(i, ui.value);
                await forward(global);
            }
            var x = Math.floor((ui.value / 1) * 255);
            $(".gman-widget-outer")[50 + i].style.backgroundColor = rgbaToHex([
                255 - Math.max(-x, 0),
                255 - Math.abs(x),
                255 - Math.max(x, 0),
                128,
            ]);
        };
    }
    function updateF() {
        return function (event, ui) {
            for (let i = 0; i < global.M; i++) global.renderers[i].F_num = ui.value;
        };
    }
    function uiSync() {
        if (!sandbox && !global.onBias) return;
        if (global.onBias) {
            for (let i = 0; i < 50; i++) {
                $(".gman-widget-slider")[i].value = (global.betas_bias.data[i] * 1000).toFixed(0);
                $(".gman-widget-value")[i].innerHTML = global.betas_bias.data[i].toFixed(3);
            }
            for (let i = 0; i < 15; i++) {
                $(".gman-widget-slider")[50 + i].value = (
                    global.pose_params_bias.data[i] * 1000
                ).toFixed(0);
                $(".gman-widget-value")[50 + i].innerHTML =
                    global.pose_params_bias.data[i].toFixed(3);
            }
        } else {
            for (let i = 0; i < 50; i++) {
                $(".gman-widget-slider")[i].value = (global.betas.data[i] * 1000).toFixed(0);
                $(".gman-widget-value")[i].innerHTML = global.betas.data[i].toFixed(3);
            }
            for (let i = 0; i < 15; i++) {
                $(".gman-widget-slider")[50 + i].value = (
                    global.pose_params.data[i] * 1000
                ).toFixed(0);
                $(".gman-widget-value")[50 + i].innerHTML = global.pose_params.data[i].toFixed(3);
            }
        }
        for (let i = 0; i < 65; i++) {
            var m = i < 50 ? 2000 : 1000;
            var x = Math.floor(($(".gman-widget-slider")[i].value / m) * 255);
            $(".gman-widget-outer")[i].style.backgroundColor = rgbaToHex([
                255 - Math.max(-x, 0),
                255 - Math.abs(x),
                255 - Math.max(x, 0),
                128,
            ]);
        }
    }
    function rgbaToHex(rgba) {
        let hex = "#";
        for (let i = 0; i < 4; i++) {
            hex += rgba[i].toString(16).padStart(2, "0");
        }
        return hex;
    }
    global.canvas.addEventListener("contextmenu", function (e) {
        e.preventDefault();
    });
    var lastx = 0,
        lasty = 0,
        incanvas = true,
        down = false,
        basex,
        basey,
        btn = 0;
    var rotx, roty, lastrotx, lastroty;
    global.canvas.onmouseenter = function (e) {
        incanvas = true;
        disableWindowScroll();
    };
    global.canvas.onmouseleave = function (e) {
        incanvas = false;
        down = false;
        enableWindowScroll();
    };
    global.canvas.onmousedown = function (e) {
        if (!incanvas) return;
        down = true;
        var xx = is_mobile ? e.touches[0].clientX : e.clientX;
        var yy = is_mobile ? e.touches[0].clientY : e.clientY;
        if (e.button == 2 || (is_mobile && e.touches.length == 2)) {
            btn = 2;
            basex = translation[0];
            basey = translation[1];
            lastx = xx;
            lasty = yy;
        } else if (e.button == 0 || is_mobile) {
            btn = 0;
            rotx = rotation[0];
            roty = rotation[1];
            lastrotx = yy;
            lastroty = xx;
        }
    };
    global.canvas.onmousemove = function (e) {
        if (!down) return;
        var xx = is_mobile ? e.touches[0].clientX : e.clientX;
        var yy = is_mobile ? e.touches[0].clientY : e.clientY;
        if (btn == 2) {
            translation[0] = basex + xx - lastx;
            translation[1] = basey - (yy - lasty);
        }
        if (btn == 0) {
            rotation[0] = rotx + (yy - lastrotx) / 250;
            rotation[1] = roty + (xx - lastroty) / 250;
            rotation[0] = Math.min(Math.max(rotation[0], -rotationLimit), rotationLimit);
            rotation[1] = Math.min(Math.max(rotation[1], -rotationLimit), rotationLimit);
        }
    };
    global.canvas.onmouseup = function (e) {
        if (!incanvas) return;
        down = false;
    };

    //for mobile
    global.canvas.ontouchstart = global.canvas.onmousedown;
    global.canvas.ontouchmove = global.canvas.onmousemove;
    global.canvas.ontouchend = global.canvas.onmouseup;

    global.canvas.onwheel = function (e) {
        if (!incanvas) return;
        translation[2] += e.wheelDelta / 2;
        translation[2] = Math.min(Math.max(translation[2], -2000), 1);
        e.preventDefault();
    };
    var winX = null;
    var winY = null;
    window.addEventListener("scroll", function () {
        if (winX !== null && winY !== null) {
            window.scrollTo(winX, winY);
        }
    });
    function disableWindowScroll() {
        winX = window.scrollX;
        winY = window.scrollY;
    }
    function enableWindowScroll() {
        winX = null;
        winY = null;
    }
    var ani_st_time;
    var framedata = zeros([global.seq_data.shape[1]], "float32");
    var framedata_ema = zeros([global.seq_data.shape[1]], "float32");
    async function drawLoop() {
        // Compute the matrix
        var aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
        var zNear = 0.1;
        var zFar = 50;
        var projection_matrix = m4.perspective(degToRad(14), aspect, zNear, zFar);
        var view_matrix = m4.scaling(1, 1, 1);

        view_matrix = m4.translate(
            view_matrix,
            translation[0] / 200,
            translation[1] / 200,
            translation[2] / 200
        );
        view_matrix = m4.xRotate(view_matrix, rotation[0]);
        view_matrix = m4.yRotate(view_matrix, rotation[1]);
        view_matrix = m4.zRotate(view_matrix, rotation[2]);

        global.vertex_matrix = m4.multiply(projection_matrix, view_matrix);
        global.normal_matrix = m4.transpose(m4.inverse(view_matrix));
        global.projection_matrix = projection_matrix;
        global.view_matrix = view_matrix;

        if (startplay) {
            startplay = 0;
            current = stframe;
            ani_st_time = performance.now();
            framedata_ema = zeros([global.seq_data.shape[1]], "float32");
        }
        if (edframe != -1 && performance.now() - ani_st_time >= (current - stframe) * (1000 / 25)) {
            // 25fps
            current++;
            if (current % 10 == 0) {
                $(`#abv${onani_}`)[0].style.width =
                    (150 * (current - stframe)) / (edframe - stframe) + "px";
            }
            framedata.data = global.seq_data.data.slice(
                current * framedata.shape[0],
                (current + 1) * framedata.shape[0]
            );
            ops.mulseq(framedata_ema, smoothing);
            ops.mulseq(framedata, 1.0 - smoothing);
            ops.addeq(framedata_ema, framedata);
            global.betas.data = framedata.data.slice(0, 50);
            global.pose_params.data = framedata.data.slice(50, 65);
            ops.addeq(global.betas, global.betas_bias); //add bias
            ops.addeq(global.pose_params, global.pose_params_bias); //add bias
            await forward(global);
            uiSync();
        }
        if (current == edframe - 1) {
            edframe = -1;
        }
        for (let i = 0; i < Math.min(global.M, global.max_M); i++) {
            global.renderers[i].render(global);
        }

        gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        frameCount++;
        if (frameCount % 30 == 0) {
            var now = performance.now();
            var time = now - lastTime;
            lastTime = now;
            timeText.innerHTML = "FPS:" + (1000 / (time / 30)).toFixed(2);
            resText.innerHTML = "Res: " + gl.canvas.width + "x" + gl.canvas.height;
        }
        requestAnimationFrame(drawLoop);
    }
    requestAnimationFrame(drawLoop);
}

function isMobile() {
    var userAgentInfo = navigator.userAgent;
    var mobileAgents = ["Android", "iPhone", "SymbianOS", "Windows Phone", "iPad", "iPod"];
    var mobile_flag = false;

    for (var v = 0; v < mobileAgents.length; v++) {
        if (userAgentInfo.indexOf(mobileAgents[v]) > 0) {
            mobile_flag = true;
            break;
        }
    }
    var screen_width = window.screen.width;
    var screen_height = window.screen.height;

    if (screen_width < 1024) {
        mobile_flag = true;
    }

    return mobile_flag;
}
