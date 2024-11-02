let slideIndex = 0;
const slides = document.querySelectorAll('.slider img');
const totalSlides = slides.length;

function changeSlide(direction) {
    slideIndex += direction;
    if (slideIndex >= totalSlides) {
        slideIndex = 0;
    } else if (slideIndex < 0) {
        slideIndex = totalSlides - 1;
    }
    updateSlider();
}

function updateSlider() {
    const slider = document.querySelector('.slider');
    slider.style.transform = `translateX(${-slideIndex * 100}%)`;
}

setInterval(() => {
    changeSlide(1);
}, 5000); // Auto-slide every 5 seconds
