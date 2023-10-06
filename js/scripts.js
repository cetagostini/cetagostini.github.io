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

// Collapsable code-box
var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.parentElement.nextElementSibling;  // Navigate up to 'button-container' then to the next sibling 'content'
    if (content.style.maxHeight){
      content.style.maxHeight = null;
    } else {
      content.style.maxHeight = content.scrollHeight + "px";
    } 
  });
}

// Copy code to clipboard
var copyButtons = document.querySelectorAll('.copy-button');
copyButtons.forEach(function(button) {
  button.addEventListener('click', function() {
    var codeId = this.getAttribute('data-target');
    var code = document.getElementById(codeId).textContent;
    navigator.clipboard.writeText(code).then(function() {
      console.log('Code successfully copied to clipboard');
    }).catch(function() {
      console.log('Failed to copy code to clipboard');
    });
  });
});
