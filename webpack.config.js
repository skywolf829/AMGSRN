const path = require('path');

module.exports = {
    entry: './js/index.js',  // Updated to reflect the new path
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist'),
    },
    mode: 'development',  // Use 'production' for optimized builds
};