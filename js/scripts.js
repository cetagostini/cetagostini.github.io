/*!
* Start Bootstrap - Personal v1.0.1 (https://startbootstrap.com/template-overviews/personal)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-personal/blob/master/LICENSE)
*/
// This file is intentionally blank
// Use this file to add JavaScript to your project

document.addEventListener('DOMContentLoaded', function() {
    var images = document.querySelectorAll('.img-fluid');

    images.forEach(function(img) {
        img.onload = function() {
            // Check if the image is more vertical (height > width)
            if (img.naturalHeight > img.naturalWidth) {
                img.classList.add('vertical-aspect');
            }
        }
    });
});