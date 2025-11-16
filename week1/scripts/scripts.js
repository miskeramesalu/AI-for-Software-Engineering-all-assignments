// User Data
let currentUser = null;
let userProgress = JSON.parse(localStorage.getItem("userProgress")) || {};

// DOM Elements
const coursesContainer = document.getElementById("courses-container");
const courseDetailsSection = document.getElementById("course-details-section");
const coursesSection = document.getElementById("courses-section");
const userSection = document.getElementById("user-section");
const backToCoursesBtn = document.getElementById("back-to-courses");
const markCompleteBtn = document.getElementById("mark-complete-btn");
const loginLink = document.getElementById("login-link");
const homeLink = document.getElementById("home-link");
const coursesLink = document.getElementById("courses-link");
const loginTab = document.getElementById("login-tab");
const signupTab = document.getElementById("signup-tab");
const loginForm = document.getElementById("login-form");
const signupForm = document.getElementById("signup-form");
const loginFormElement = document.getElementById("loginForm");
const signupFormElement = document.getElementById("signupForm");
const userAvatar = document.getElementById("user-avatar");

// Initialize the application
function init() {
  renderCourses();
  setupEventListeners();

  // Check if user is logged in
  const savedUser = localStorage.getItem("currentUser");
  if (savedUser) {
    currentUser = JSON.parse(savedUser);
    updateUserInterface();
  }
}

// Render courses on the homepage
function renderCourses() {
  coursesContainer.innerHTML = "";

  courses.forEach((course) => {
    const progress = calculateCourseProgress(course.id);
    const courseCard = document.createElement("div");
    courseCard.className = "col-md-6 col-lg-3 mb-4";
    courseCard.innerHTML = `
            <div class="card course-card">
                <div class="position-relative">
                    <img src="${course.image}" class="card-img-top" alt="${
      course.title
    }" style="height: 200px; object-fit: cover;">
                    ${
                      course.completed
                        ? '<span class="course-badge">Completed</span>'
                        : ""
                    }
                </div>
                <div class="card-body">
                    <h5 class="card-title">${course.title}</h5>
                    <p class="card-text">${course.description.substring(
                      0,
                      100
                    )}...</p>
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="badge bg-secondary">${course.level}</span>
                        <span class="text-muted">${course.duration}</span>
                    </div>
                    <div class="progress mt-3">
                        <div class="progress-bar" role="progressbar" style="width: ${progress}%"></div>
                    </div>
                    <p class="text-muted small mt-1">${progress}% Complete</p>
                    <button class="btn btn-primary w-100 mt-2 view-course-btn" data-id="${
                      course.id
                    }">
                        ${course.completed ? "Review Course" : "View Course"}
                    </button>
                </div>
            </div>
        `;
    coursesContainer.appendChild(courseCard);
  });

  // Add event listeners to view course buttons
  document.querySelectorAll(".view-course-btn").forEach((button) => {
    button.addEventListener("click", function () {
      const courseId = parseInt(this.getAttribute("data-id"));
      showCourseDetails(courseId);
    });
  });
}

// Show course details
function showCourseDetails(courseId) {
  const course = courses.find((c) => c.id === courseId);
  if (!course) return;

  // Update course details
  document.getElementById("course-detail-title").textContent = course.title;
  document.getElementById("course-title").textContent = course.title;
  document.getElementById("course-description").textContent =
    course.description;
  document.getElementById("course-duration").textContent = course.duration;
  document.getElementById("course-level").textContent = course.level;
  document.getElementById("course-instructor").textContent = course.instructor;
  document.getElementById("course-prerequisites").textContent =
    course.prerequisites;

  // Update course status
  const courseStatus = document.getElementById("course-status");
  if (course.completed) {
    courseStatus.textContent = "Completed";
    courseStatus.className = "badge bg-success";
    markCompleteBtn.textContent = "Course Completed";
    markCompleteBtn.disabled = true;
  } else {
    courseStatus.textContent = "In Progress";
    courseStatus.className = "badge bg-primary";
    markCompleteBtn.textContent = "Mark Course as Completed";
    markCompleteBtn.disabled = false;
  }

  // Render lessons
  const lessonsContainer = document.getElementById("lessons-container");
  lessonsContainer.innerHTML = "";

  course.lessons.forEach((lesson) => {
    const lessonElement = document.createElement("div");
    lessonElement.className = `lesson-item ${
      lesson.completed ? "completed" : ""
    }`;
    lessonElement.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h6 class="lesson-title mb-0">${lesson.title}</h6>
                    <small class="text-muted">Duration: ${
                      lesson.duration
                    }</small>
                </div>
                ${
                  lesson.completed
                    ? '<span class="badge bg-success"><i class="fas fa-check"></i></span>'
                    : '<span class="badge bg-secondary">Not Started</span>'
                }
            </div>
        `;
    lessonsContainer.appendChild(lessonElement);
  });

  // Update progress
  const progress = calculateCourseProgress(courseId);
  document.getElementById("course-progress").style.width = `${progress}%`;
  document.getElementById(
    "progress-text"
  ).textContent = `${progress}% Complete`;

  // Update user progress display
  updateUserProgressDisplay();

  // Show course details section
  coursesSection.classList.add("d-none");
  courseDetailsSection.classList.remove("d-none");
  userSection.classList.add("d-none");

  // Set up mark complete button
  markCompleteBtn.onclick = function () {
    markCourseAsCompleted(courseId);
  };
}

// Calculate course progress
function calculateCourseProgress(courseId) {
  const course = courses.find((c) => c.id === courseId);
  if (!course) return 0;

  if (course.completed) return 100;

  const completedLessons = course.lessons.filter(
    (lesson) => lesson.completed
  ).length;
  return Math.round((completedLessons / course.lessons.length) * 100);
}

// Mark course as completed
function markCourseAsCompleted(courseId) {
  const course = courses.find((c) => c.id === courseId);
  if (!course) return;

  // Mark all lessons as completed
  course.lessons.forEach((lesson) => {
    lesson.completed = true;
  });

  // Mark course as completed
  course.completed = true;

  // Update user progress in localStorage
  if (currentUser) {
    if (!userProgress[currentUser.email]) {
      userProgress[currentUser.email] = {};
    }
    userProgress[currentUser.email][courseId] = {
      completed: true,
      completedDate: new Date().toISOString(),
    };
    localStorage.setItem("userProgress", JSON.stringify(userProgress));
  }

  // Update UI
  showCourseDetails(courseId);
  renderCourses();

  // Show success message
  alert(`Congratulations! You've completed "${course.title}"!`);
}

// Update user progress display
function updateUserProgressDisplay() {
  const userProgressElement = document.getElementById("user-progress");
  if (!currentUser) {
    userProgressElement.innerHTML =
      '<p class="text-muted">Please log in to track your progress.</p>';
    return;
  }

  const userCourses = userProgress[currentUser.email] || {};
  let html = "";

  courses.forEach((course) => {
    const progress = calculateCourseProgress(course.id);
    html += `
            <div class="mb-3">
                <h6>${course.title}</h6>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${progress}%"></div>
                </div>
                <small class="text-muted">${progress}% Complete</small>
            </div>
        `;
  });

  userProgressElement.innerHTML = html;
}

// Setup event listeners
function setupEventListeners() {
  // Navigation
  backToCoursesBtn.addEventListener("click", function () {
    courseDetailsSection.classList.add("d-none");
    coursesSection.classList.remove("d-none");
    userSection.classList.add("d-none");
  });

  homeLink.addEventListener("click", function (e) {
    e.preventDefault();
    coursesSection.classList.remove("d-none");
    courseDetailsSection.classList.add("d-none");
    userSection.classList.add("d-none");
  });

  coursesLink.addEventListener("click", function (e) {
    e.preventDefault();
    coursesSection.classList.remove("d-none");
    courseDetailsSection.classList.add("d-none");
    userSection.classList.add("d-none");
  });

  loginLink.addEventListener("click", function (e) {
    e.preventDefault();
    coursesSection.classList.add("d-none");
    courseDetailsSection.classList.add("d-none");
    userSection.classList.remove("d-none");
  });

  // Login/Signup tabs
  loginTab.addEventListener("click", function () {
    loginTab.classList.add("active");
    signupTab.classList.remove("active");
    loginForm.classList.add("active-form");
    signupForm.classList.remove("active-form");
  });

  signupTab.addEventListener("click", function () {
    signupTab.classList.add("active");
    loginTab.classList.remove("active");
    signupForm.classList.add("active-form");
    loginForm.classList.remove("active-form");
  });

  // Login form
  loginFormElement.addEventListener("submit", function (e) {
    e.preventDefault();
    const email = document.getElementById("login-email").value;
    const password = document.getElementById("login-password").value;

    // Simple validation (in a real app, this would be more secure)
    if (email && password) {
      currentUser = {
        email: email,
        name: email.split("@")[0],
      };
      localStorage.setItem("currentUser", JSON.stringify(currentUser));
      updateUserInterface();
      alert("Login successful!");

      // Redirect to courses
      coursesSection.classList.remove("d-none");
      userSection.classList.add("d-none");
    } else {
      alert("Please fill in all fields.");
    }
  });

  // Signup form
  signupFormElement.addEventListener("submit", function (e) {
    e.preventDefault();
    const name = document.getElementById("signup-name").value;
    const email = document.getElementById("signup-email").value;
    const password = document.getElementById("signup-password").value;
    const confirmPassword = document.getElementById(
      "signup-confirm-password"
    ).value;

    // Simple validation
    if (name && email && password && confirmPassword) {
      if (password !== confirmPassword) {
        alert("Passwords do not match.");
        return;
      }

      currentUser = {
        email: email,
        name: name,
      };
      localStorage.setItem("currentUser", JSON.stringify(currentUser));
      updateUserInterface();
      alert("Account created successfully!");

      // Redirect to courses
      coursesSection.classList.remove("d-none");
      userSection.classList.add("d-none");
    } else {
      alert("Please fill in all fields.");
    }
  });
}

// Update user interface based on login status
function updateUserInterface() {
  if (currentUser) {
    loginLink.textContent = "Logout";
    loginLink.onclick = function (e) {
      e.preventDefault();
      currentUser = null;
      localStorage.removeItem("currentUser");
      updateUserInterface();
      coursesSection.classList.remove("d-none");
      userSection.classList.add("d-none");
    };

    userAvatar.textContent = currentUser.name.charAt(0).toUpperCase();
    userAvatar.title = currentUser.name;
  } else {
    loginLink.textContent = "Login";
    loginLink.onclick = function (e) {
      e.preventDefault();
      coursesSection.classList.add("d-none");
      courseDetailsSection.classList.add("d-none");
      userSection.classList.remove("d-none");
    };

    userAvatar.textContent = "U";
    userAvatar.title = "User";
  }

  updateUserProgressDisplay();
}

// Initialize the application
document.addEventListener("DOMContentLoaded", init);
