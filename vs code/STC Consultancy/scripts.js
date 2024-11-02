document.addEventListener("DOMContentLoaded", function() {
    // Your JS code can go here for interactivity
    // Example: Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('nav ul li a');
    
    for (let link of navLinks) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            window.scrollTo({
                top: targetSection.offsetTop,
                behavior: 'smooth'
            });
        });
    }
});
