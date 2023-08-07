let openbtn, sidebar, content;

document.addEventListener("DOMContentLoaded", function() {
    openbtn = document.querySelector(".openbtn");
    sidebar = document.querySelector(".side_bar");
    content = document.querySelector(".content");
});

function openMenu() {
    sidebar.style.width = "250px";
    content.style.filter = "brightness(80%)";
    content.style.backgroundColor = "rgba(0, 0, 0, 0.2)";
}

function closeMenu() {
    sidebar.style.width = "0";
    openbtn.style.display = 'block';
    content.style.filter = "brightness(100%)";
    content.style.backgroundColor = "#fff";
}