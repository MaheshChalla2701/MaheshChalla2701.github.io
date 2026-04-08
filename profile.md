---
layout: default
title: Mahesh Challa
permalink: /profile/
---

<style>
.profile-header {
    display: flex;
    align-items: center;
    gap: 40px;
    margin-top: 20px;
    margin-bottom: 40px;
}
.profile-image img {
    width: 280px;
    height: 280px;
    border-radius: 50%;
    object-fit: cover;
    border: 4px solid #fff;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.profile-details {
    display: flex;
    flex-direction: column;
}
.profile-details h1 {
    margin: 0 0 10px 0;
    font-size: 3.5rem;
    font-weight: 500;
    color: #333;
}
.profile-details .bio {
    font-size: 1.35rem;
    font-style: italic;
    color: #666;
    margin: 0 0 20px 0;
    max-width: 500px;
}
.social-links {
    display: flex;
    gap: 15px;
}
.social-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 45px;
    height: 45px;
    background-color: #333;
    color: white;
    border-radius: 50%;
    text-decoration: none;
    transition: background-color 0.3s;
}
.social-icon:hover {
    background-color: #555;
}
.social-icon svg {
    width: 22px;
    height: 22px;
    fill: currentColor;
}

@media (max-width: 768px) {
    .profile-header {
        flex-direction: column;
        text-align: center;
    }
    .profile-details h1 {
        font-size: 2.5rem;
    }
    .profile-details .bio {
        margin: 0 auto 20px auto;
    }
    .social-links {
        justify-content: center;
    }
}

/* Light/Dark mode adjustments for Architect theme */
body[data-theme="dark"] .profile-details h1 { color: #f5f5f5; }
body[data-theme="dark"] .profile-details .bio { color: #aaa; }
body[data-theme="dark"] .social-icon { background-color: #f5f5f5; color: #333; }
body[data-theme="dark"] .social-icon:hover { background-color: #ccc; }
/* Timeline CSS */
.timeline {
    position: relative;
    max-width: 900px;
    margin: 40px auto;
}
.timeline::before {
    content: '';
    position: absolute;
    top: 5px;
    bottom: 0;
    left: 120px;
    width: 2px;
    background: #e0e0e0;
}
.timeline-item {
    position: relative;
    display: flex;
    margin-bottom: 50px;
}
.timeline-date {
    width: 100px;
    padding-top: 5px;
    text-align: right;
    position: relative;
    color: #999;
    font-size: 1.1rem;
}
.timeline-date::after {
    content: '';
    position: absolute;
    top: 10px;
    right: -27px;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: #ccc;
    border: 3px solid #fff;
    z-index: 1;
}
.timeline-content {
    flex: 1;
    display: flex;
    padding-left: 60px;
    gap: 30px;
}
.timeline-logo {
    flex-shrink: 0;
    width: 90px;
    display: flex;
    justify-content: center;
}
.timeline-logo img {
    max-width: 100%;
    max-height: 90px;
    object-fit: contain;
    border-radius: 8px;
}
.timeline-text {
    flex: 1;
    font-size: 1.15rem;
    line-height: 1.5;
    color: #333;
    padding-top: 2px;
}
.timeline-text a {
    color: #0366d6;
    text-decoration: none;
    font-weight: 500;
}
.timeline-text a:hover {
    text-decoration: underline;
}

@media (max-width: 768px) {
    .timeline::before { left: 15px; }
    .timeline-date { width: 100%; text-align: left; padding-left: 35px; margin-bottom: 10px; }
    .timeline-date::after { right: auto; left: 9px; top: 10px; }
    .timeline-content { padding-left: 35px; flex-direction: column; gap: 15px; }
    .timeline-logo { width: 70px; justify-content: flex-start; }
    .timeline-logo img { max-height: 70px; }
}

/* Architect theme compatibility */
body[data-theme="dark"] .timeline::before { background: #444; }
body[data-theme="dark"] .timeline-date::after { border-color: #222; background: #666; }
body[data-theme="dark"] .timeline-text { color: #ccc; }
</style>

<div class="profile-header">
    <div class="profile-image">
        <img src="{{ '/assets/images/profile-photo.jpeg' | relative_url }}" alt="Mahesh Challa">
    </div>
    <div class="profile-details">
        <h1>Mahesh Challa</h1>
        <p class="bio">I like to build intelligent cross-platform mobile apps and explore deep learning algorithms and building ai models 🧠📱🚀</p>
        <div class="social-links">
            <a href="https://github.com/MaheshChalla2701" class="social-icon" target="_blank" title="GitHub">
                <svg viewBox="0 0 24 24"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
            </a>
            <a href="https://www.linkedin.com/in/mahesh-challa-54b60a283/" class="social-icon" target="_blank" title="LinkedIn">
                <svg viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
            </a>
        </div>
    </div>
</div>

<hr style="margin: 40px 0; border: 0; border-top: 1px solid #ddd;">

## Experience & Projects

<div class="timeline">
    
    <div class="timeline-item">
        <div class="timeline-date">2026 - Present</div>
        <div class="timeline-content">
            <div class="timeline-logo">
                <img src="{{ '/assets/images/splitplan-logo.png' | relative_url }}" alt="SplitPlan">
            </div>
            <div class="timeline-text">
                Founder of <strong>SplitPlan</strong>, a social expense-sharing application built with Flutter and Firebase that makes tracking group spending and settling debts with friends effortless. The platform offers secure real-time syncing, group bill divisions, and automated balance calculations through a clean, intuitive interface.
            </div>
        </div>
    </div>

    <div class="timeline-item">
        <div class="timeline-date">2025 - Present</div>
        <div class="timeline-content">
            <div class="timeline-logo">
                <img src="{{ '/assets/images/fizi-logo.png' | relative_url }}" alt="FIZI">
            </div>
            <div class="timeline-text">
                Founder of <strong>FIZI</strong>, a cutting-edge cross-platform mobile application providing real-time exercise feedback. Powered by Google's MediaPipe and a customized Python streaming backend, FIZI tracks workouts and form, offering personalized diet planning and advanced gamification to create a complete fitness ecosystem. <br>
                <a href="https://play.google.com/store/apps/details?id=com.maheshchalla.fizi" target="_blank">View on Google Play</a>
            </div>
        </div>
    </div>

    <div class="timeline-item">
        <div class="timeline-date">2023 - 2027</div>
        <div class="timeline-content">
            <div class="timeline-logo">
                <img src="{{ '/assets/images/kmec-logo.png' | relative_url }}" alt="Keshav Memorial Engineering College">
            </div>
            <div class="timeline-text">
                Bachelor of Engineering in <strong>Computer Science</strong> at <strong>Keshav Memorial Engineering College</strong>, This is where I first got into deep learning!
            </div>
        </div>
    </div>

</div>
