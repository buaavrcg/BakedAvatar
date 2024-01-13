export function getVertexShaderSource(J) {
    return `#version 300 es
    #define VINROW 64
    in vec4 a_position;
    in vec3 a_normal;
    in vec2 a_index;
    in vec2 a_uv;
    uniform sampler2D betasTex;
    uniform sampler2D shapedirsTex;
    uniform sampler2D posesTex;
    uniform sampler2D posedirsTex;
    uniform sampler2D transformTex;
    uniform sampler2D lbsweightTex;
    uniform mat4 u_matrix;
    uniform mat4 u_normal_matrix;
    uniform mat4 u_view_matrix;
    out vec2 v_uv;
    out vec3 v_normal;
    out vec3 v_viewpos;
    int idx;

    vec3 shapedirs(int i,int j){
        j=(i%VINROW)*50+j;
        i=i/VINROW;
        return texelFetch(shapedirsTex,ivec2(j,i),0).xyz;
    }
    vec3 posedirs(int i,int j){
        j=(i%VINROW)*36+j;
        i=i/VINROW;
        return texelFetch(posedirsTex,ivec2(j,i),0).xyz;
    }
    float lbsweight(int i,int j){
        j=(i%VINROW)*${J}+j;
        i=i/VINROW;
        return texelFetch(lbsweightTex,ivec2(j,i),0).x;
    }
    vec3 shapeMatMul(){
        vec3 sum=vec3(0.0);
        for(int i=0;i<50;i++){
            sum+=shapedirs(idx,i)*texelFetch(betasTex,ivec2(i,0),0).x;
        }
        return sum;
    }
    vec3 poseMatMul(){
        vec3 sum=vec3(0.0);
        for(int i=0;i<36;i+=4){
            vec4 p=texelFetch(posesTex,ivec2(i,0),0);
            sum+=posedirs(idx,i+0)*p.x;
            sum+=posedirs(idx,i+1)*p.y;
            sum+=posedirs(idx,i+2)*p.z;
            sum+=posedirs(idx,i+3)*p.w;
        }
        return sum;
    }
    mat4 lbsMatMul(){
        mat4 rot=mat4(0.0);
        for(int j=0;j<${J};j++){
            float w=lbsweight(idx,j);
            rot[0]+=w*texelFetch(transformTex,ivec2(0,j),0);
            rot[1]+=w*texelFetch(transformTex,ivec2(1,j),0);
            rot[2]+=w*texelFetch(transformTex,ivec2(2,j),0);
            rot[3]+=w*texelFetch(transformTex,ivec2(3,j),0);
        }
        return transpose(rot);
    }
    void main() {
        idx = int(a_index.x);
        vec3 blendshape = shapeMatMul() + poseMatMul();
        mat4 transform = lbsMatMul();
        mat3 transform_normal = mat3(transpose(inverse(transform)));
        vec4 pos = transform * (a_position + vec4(blendshape, 0));

        v_viewpos = (u_view_matrix * pos).xyz;
        v_uv = a_uv;
        v_normal = normalize(mat3(u_normal_matrix) * transform_normal * a_normal);

        gl_Position = u_matrix * pos;
    }`;
}

export function getFragmentShaderSource(n_basis, dim_middle) {
    return `#version 300 es
    precision highp float;
    in vec2 v_uv;
    in vec3 v_normal;
    in vec3 v_viewpos;
    uniform sampler2D posfeatTex;
    uniform sampler2D radfeatTex;
    uniform sampler2D sfcTex;
    uniform int renderMode;
    out vec4 outColor;

    vec2 posfeat_s = 1.0 / vec2(2.0,1.0);
    vec2 radiance_s = 1.0 / vec2(${Math.min(n_basis, 4)}.0,${Math.ceil(n_basis / 4)}.0);

    float coef[${n_basis}];
    vec4 sfc(int i,int j) {
        return texelFetch(sfcTex,ivec2(j,i),0);
    }
    vec4 posfeat(int i) {
        vec2 uv_offset = vec2(float(i),0.0);
        return texture(posfeatTex,(v_uv+uv_offset)*posfeat_s);
    }
    vec4 radianceBasis(int i) {
        vec2 uv_offset = vec2(float(i&3),float(i>>2));
        return texture(radfeatTex,(v_uv+uv_offset)*radiance_s);
    }
    void runSpatialMLP(vec4 posf0, vec4 posf1, vec3 normal, vec3 viewdir) {
        mat4 out1;
        vec4 f2=vec4(normal.xyz, viewdir.x);
        vec4 f3=vec4(viewdir.yz, 0.0, 0.0);
        for(int i=0;i<${dim_middle};i++){
            float o=dot(sfc(i,0),posf0);
            o+=dot(sfc(i,1),posf1);
            o+=dot(sfc(i,2),f2);
            o+=dot(sfc(i,3),f3);
            out1[i>>2][i&3]=max(o,0.0);
        }
        float esum=0.0;
        for(int i=0;i<${n_basis};i++){
            coef[i]=0.0;
            for(int j=0;j<${dim_middle / 4};j++){
                coef[i]+=dot(sfc(i+${dim_middle},j),out1[j]);
            }
            coef[i]=exp(10.0*coef[i]);
            esum+=coef[i];
        }
        float inv_esum=1.0/esum;
        for(int i=0;i<${n_basis};i++){
            coef[i]*=inv_esum;
        }
    }
    vec4 coefColor() {
        vec4 c[10] = vec4[10](
            vec4(0.12156862745098039, 0.4666666666666667, 0.7058823529411765,1),
            vec4(1.0, 0.4980392156862745, 0.054901960784313725,1),
            vec4(0.17254901960784313, 0.6274509803921569, 0.17254901960784313,1),
            vec4(0.8392156862745098, 0.15294117647058825, 0.1568627450980392,1),
            vec4(0.5803921568627451, 0.403921568627451, 0.7411764705882353,1),
            vec4(0.5490196078431373, 0.33725490196078434, 0.29411764705882354,1),
            vec4(0.8901960784313725, 0.4666666666666667, 0.7607843137254902,1),
            vec4(0.4980392156862745, 0.4980392156862745, 0.4980392156862745,1),
            vec4(0.7372549019607844, 0.7411764705882353, 0.13333333333333333,1),
            vec4(0.09019607843137255, 0.7450980392156863, 0.8117647058823529,1)
        );
        vec4 color=vec4(0.0);
        for(int i=0;i<${n_basis};i++){
            color+=coef[i]*c[i%10];
        }
        return color;
    }
    void main() {
        if (renderMode == 1)
            return;
        if (renderMode == 2) {
            outColor=vec4(0.2039,0.5960,0.8588,0.1);
            return;
        }
        if (renderMode == 3) {
            outColor=vec4((normalize(v_normal)+1.0)/2.0,1.0);
            return;
        }
        vec3 v_viewdir=normalize(v_viewpos);
        if (renderMode == 4) {
            outColor=vec4((v_viewdir+1.0)/2.0,1.0);
            return;
        }
        
        vec4 v_posf0=posfeat(0);
        vec4 v_posf1=posfeat(1);
        runSpatialMLP(v_posf0, v_posf1, v_normal, v_viewdir);

        if (renderMode == 5) {
            outColor=coefColor();
            return;
        }
        
        vec4 sumColor=vec4(0);
        for(int i=0;i<${n_basis};i++){
            sumColor+=coef[i]*radianceBasis(i);
        }
        outColor=sumColor;
        if (renderMode == 6) {
            outColor.w=1.0;
        }
        //outColor=vec4(vec3(outColor)*outColor.w+vec3(1.0)*(1.0-outColor.w),1.0);
    }`;
}
