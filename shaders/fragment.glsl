// fragmentShader.glsl
uniform sampler2D uTexture;
uniform float uFloatArray[96]; // Assuming a fixed-size array for simplicity
varying vec3 vPosition;

void main() {
    // Example: Modulating color based on the array data
    // float value = uFloatArray[int(mod(vPosition.x * 96.0, 96.0))];
    gl_FragColor = vec4((vPosition.x + 1.0) * 0.5, (vPosition.y + 1.0) * 0.5, (vPosition.z + 1.0) * 0.5, 1.0);
}
