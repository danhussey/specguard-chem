// SpecGuard-Chem Website JavaScript

class SpecGuardWebsite {
    constructor() {
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupScrollEffects();
        this.loadStats();
        this.setupAnimations();
    }

    setupNavigation() {
        // Mobile menu toggle
        const navToggle = document.getElementById('nav-toggle');
        const navMenu = document.getElementById('nav-menu');

        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('active');
                navToggle.classList.toggle('active');
            });
        }

        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    const headerOffset = 80;
                    const elementPosition = target.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });

        // Active navigation highlighting
        this.updateActiveNavigation();
        window.addEventListener('scroll', () => this.updateActiveNavigation());
    }

    updateActiveNavigation() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link[href^="#"]');

        let current = '';
        const scrollPos = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.getBoundingClientRect().top + window.pageYOffset;
            const sectionHeight = section.offsetHeight;

            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    }

    setupScrollEffects() {
        // Navbar background on scroll
        const navbar = document.querySelector('.navbar');

        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });

        // Intersection Observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        document.querySelectorAll('.overview-card, .domain-card, .step').forEach(el => {
            observer.observe(el);
        });
    }

    async loadStats() {
        try {
            // Load leaderboard data to get current stats
            const response = await fetch('./leaderboard/data/leaderboard.json');
            const data = await response.json();

            // Update total models count
            const totalModelsElement = document.getElementById('total-models');
            if (totalModelsElement && data) {
                totalModelsElement.textContent = data.length;
                this.animateCounter(totalModelsElement, 0, data.length, 1000);
            }
        } catch (error) {
            console.log('Could not load stats data:', error);
            // Keep default values
        }
    }

    animateCounter(element, start, end, duration) {
        const range = end - start;
        const stepTime = Math.abs(Math.floor(duration / range));
        const timer = setInterval(() => {
            start += 1;
            element.textContent = start;
            if (start === end) {
                clearInterval(timer);
            }
        }, stepTime);
    }

    setupAnimations() {
        // Add staggered animation delays to cards
        document.querySelectorAll('.overview-card').forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
        });

        document.querySelectorAll('.domain-card').forEach((card, index) => {
            card.style.animationDelay = `${index * 0.05}s`;
        });

        // Parallax effect for hero section
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const hero = document.querySelector('.hero');
            if (hero) {
                const rate = scrolled * -0.5;
                hero.style.transform = `translateY(${rate}px)`;
            }
        });
    }

    // Utility method for copying code snippets
    setupCodeCopy() {
        document.querySelectorAll('.code-block').forEach(block => {
            const button = document.createElement('button');
            button.innerHTML = '📋 Copy';
            button.className = 'copy-button';
            button.addEventListener('click', () => {
                navigator.clipboard.writeText(block.textContent.trim()).then(() => {
                    button.innerHTML = '✅ Copied!';
                    setTimeout(() => {
                        button.innerHTML = '📋 Copy';
                    }, 2000);
                });
            });

            block.style.position = 'relative';
            block.appendChild(button);
        });
    }

    // Method to highlight current page in navigation
    highlightCurrentPage() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');

        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && (href === currentPath ||
                (currentPath === '/' && href === '/') ||
                (currentPath.includes(href) && href !== '/'))) {
                link.classList.add('current-page');
            }
        });
    }
}

// Initialize website functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SpecGuardWebsite();
});

// Additional CSS for JavaScript-added elements
const additionalStyles = `
.navbar.scrolled {
    background: rgba(255, 255, 255, 0.98);
    box-shadow: var(--shadow-md);
}

.nav-link.active {
    color: var(--primary-blue);
    font-weight: 600;
}

.nav-link.current-page {
    color: var(--primary-blue);
    font-weight: 600;
}

.copy-button {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    background: var(--gray-700);
    color: white;
    border: none;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    cursor: pointer;
    transition: var(--transition-fast);
}

.copy-button:hover {
    background: var(--gray-600);
}

@media (max-width: 768px) {
    .nav-menu.active {
        display: flex;
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid var(--gray-200);
        flex-direction: column;
        padding: 1rem;
        box-shadow: var(--shadow-lg);
    }

    .nav-toggle.active span:nth-child(1) {
        transform: rotate(-45deg) translate(-5px, 6px);
    }

    .nav-toggle.active span:nth-child(2) {
        opacity: 0;
    }

    .nav-toggle.active span:nth-child(3) {
        transform: rotate(45deg) translate(-5px, -6px);
    }
}

/* Enhanced animation classes */
.fade-in {
    animation: fadeIn 0.6s ease-out forwards;
}

.slide-up {
    animation: slideUp 0.6s ease-out forwards;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Loading states */
.loading-skeleton {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
}

@keyframes loading {
    0% {
        background-position: 200% 0;
    }
    100% {
        background-position: -200% 0;
    }
}
`;

// Add additional styles to head
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);