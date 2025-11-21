/**
 * Sticky2Jira - Main Application JavaScript
 * Handles UI interactions, SocketIO events, and API calls
 */

// ============================================================================
// Global State
// ============================================================================

const appState = {
  jiraSettings: null,
  currentImage: null,
  ocrRegions: [],
  colorMappings: {},
  issues: [],
  projects: [],
  issueTypes: {},
  socket: null,
};

// ============================================================================
// SocketIO Connection
// ============================================================================

function initSocketIO() {
  appState.socket = io();

  appState.socket.on("connect", () => {
    console.log("Connected to server via SocketIO");
    showAlert("Connected to server", "success", 3000);
  });

  appState.socket.on("disconnect", () => {
    console.log("Disconnected from server");
    showAlert("Disconnected from server", "warning");
  });

  // OCR Progress
  appState.socket.on("ocr_progress", (data) => {
    console.log("OCR Progress:", data);
    updateProgress(data.percent, data.message);

    if (data.status === "complete") {
      appState.ocrRegions = data.regions;
      hideProgress();
      renderOCRResults();
      showAlert("OCR processing complete!", "success");
      document.getElementById("proceedToMappingBtn").style.display = "block";
    } else if (data.status === "error") {
      hideProgress();
      showAlert(`OCR Error: ${data.message}`, "danger");
    }
  });

  // Import Progress
  appState.socket.on("import_progress", (data) => {
    console.log("Import Progress:", data);
    const message = `${data.current}/${data.total}: ${data.message}`;
    updateProgress(data.percent, message);

    if (data.status === "complete") {
      hideProgress();
      renderImportResults(data);
      showAlert("Import complete!", "success");
      switchTab("results-tab");
    } else if (data.status === "error") {
      hideProgress();
      showAlert(`Import Error: ${data.message}`, "danger");
    }
  });
}

// ============================================================================
// Jira Setup
// ============================================================================

function loadJiraSettings() {
  fetch("/api/jira/settings")
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.settings) {
        appState.jiraSettings = data.settings;
        populateJiraForm(data.settings);
      }
    })
    .catch((error) => console.error("Failed to load Jira settings:", error));
}

function populateJiraForm(settings) {
  document.getElementById("serverUrl").value = settings.server_url || "";
  document.getElementById("username").value = settings.username || "";
  document.getElementById("apiToken").value = settings.api_token || "";
  document.getElementById("defaultProject").value =
    settings.default_project_key || "";
}

function testJiraConnection() {
  const settings = getJiraFormData();

  showAlert("Testing connection...", "info");

  fetch("/api/jira/test-connection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showAlert(`Connected to ${data.server_title}`, "success");
      } else {
        showAlert(`Connection failed: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      showAlert(`Connection error: ${error.message}`, "danger");
    });
}

function saveJiraSettings(event) {
  event.preventDefault();
  const settings = getJiraFormData();

  fetch("/api/jira/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        appState.jiraSettings = settings;
        showAlert("Settings saved successfully", "success");
        loadProjects();
      } else {
        showAlert(`Failed to save settings: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      showAlert(`Save error: ${error.message}`, "danger");
    });
}

function getJiraFormData() {
  return {
    server_url: document.getElementById("serverUrl").value,
    username: document.getElementById("username").value,
    api_token: document.getElementById("apiToken").value,
    default_project_key: document.getElementById("defaultProject").value,
  };
}

// ============================================================================
// Image Upload
// ============================================================================

function handleImageSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Validate file size (5MB max)
  if (file.size > 5 * 1024 * 1024) {
    showAlert("File too large. Maximum size is 5MB.", "danger");
    return;
  }

  // Preview image
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("previewImg").src = e.target.result;
    document.getElementById("imagePreview").style.display = "block";
    document.getElementById("uploadImageBtn").disabled = false;
  };
  reader.readAsDataURL(file);
}

function uploadImage() {
  const fileInput = document.getElementById("imageFile");
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  showProgress(0, "Uploading image...");

  fetch("/api/ocr/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        appState.currentImage = data.filename;
        hideProgress();
        showAlert("Image uploaded successfully", "success");
        document.getElementById("processOcrBtn").style.display = "block";
      } else {
        hideProgress();
        showAlert(`Upload failed: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      hideProgress();
      showAlert(`Upload error: ${error.message}`, "danger");
    });
}

function startOCRProcessing() {
  if (!appState.currentImage) return;

  showProgress(0, "Starting OCR processing...");

  fetch("/api/ocr/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: appState.currentImage }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        // Progress updates will come via SocketIO
        console.log("OCR processing started");
      } else {
        hideProgress();
        showAlert(`OCR failed: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      hideProgress();
      showAlert(`OCR error: ${error.message}`, "danger");
    });
}

// ============================================================================
// OCR Results
// ============================================================================

function renderOCRResults() {
  const container = document.getElementById("ocrResults");
  if (appState.ocrRegions.length === 0) {
    container.innerHTML = '<p class="text-muted">No regions detected.</p>';
    return;
  }

  let html = "";
  appState.ocrRegions.forEach((region) => {
    const linkedText =
      region.linked_to.length > 0
        ? `<span class="linked-indicator">Linked to: ${region.linked_to.join(
            ", "
          )}</span>`
        : "";

    html += `
            <div class="ocr-region">
                <div class="ocr-region-header">
                    <div>
                        <span class="color-badge" style="background-color: ${
                          region.color_hex
                        };"></span>
                        <strong>Region ${region.id}</strong>
                        <span class="badge bg-secondary ms-2">${
                          region.color_name
                        }</span>
                        ${linkedText}
                    </div>
                    <span class="confidence-badge badge bg-${getConfidenceBadgeClass(
                      region.confidence
                    )}">
                        ${region.confidence}% confidence
                    </span>
                </div>
                <div class="ocr-region-content">
                    <p class="mb-0">${escapeHtml(region.text)}</p>
                </div>
            </div>
        `;
  });

  container.innerHTML = html;
}

function getConfidenceBadgeClass(confidence) {
  if (confidence >= 80) return "success";
  if (confidence >= 60) return "warning";
  return "danger";
}

// ============================================================================
// Color Mapping
// ============================================================================

function loadProjects() {
  fetch("/api/jira/projects")
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        appState.projects = data.projects;
        populateProjectSelect();
      }
    })
    .catch((error) => console.error("Failed to load projects:", error));
}

function populateProjectSelect() {
  const select = document.getElementById("mappingProject");
  select.innerHTML = '<option value="">Select project...</option>';
  appState.projects.forEach((project) => {
    const option = document.createElement("option");
    option.value = project.key;
    option.textContent = `${project.key} - ${project.name}`;
    select.appendChild(option);
  });
}

function loadIssueTypes(projectKey) {
  fetch(`/api/jira/issue-types/${projectKey}`)
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        appState.issueTypes[projectKey] = data.issue_types;
        renderColorMappings();
      }
    })
    .catch((error) => console.error("Failed to load issue types:", error));
}

function renderColorMappings() {
  const container = document.getElementById("colorMappings");
  const projectKey = document.getElementById("mappingProject").value;

  if (!projectKey || !appState.issueTypes[projectKey]) {
    container.innerHTML =
      '<p class="text-muted">Select a project to configure mappings.</p>';
    return;
  }

  const uniqueColors = [
    ...new Set(
      appState.ocrRegions.map((r) => ({
        hex: r.color_hex,
        name: r.color_name,
      }))
    ),
  ];

  let html = "";
  uniqueColors.forEach((color) => {
    html += `
            <div class="color-mapping-row">
                <span class="color-badge" style="background-color: ${
                  color.hex
                };"></span>
                <label>${color.name}</label>
                <select class="form-select color-mapping-select" data-color="${
                  color.hex
                }">
                    <option value="">Select issue type...</option>
                    ${appState.issueTypes[projectKey]
                      .map(
                        (it) => `<option value="${it.name}">${it.name}</option>`
                      )
                      .join("")}
                </select>
            </div>
        `;
  });

  container.innerHTML = html;
}

// ============================================================================
// UI Helpers
// ============================================================================

function showAlert(message, type, duration = null) {
  const container = document.getElementById("alertContainer");
  const alert = document.createElement("div");
  alert.className = `alert alert-${type} alert-dismissible fade show`;
  alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
  container.appendChild(alert);

  if (duration) {
    setTimeout(() => {
      alert.remove();
    }, duration);
  }
}

function showProgress(percent, message) {
  const container = document.getElementById("progressContainer");
  const bar = document.getElementById("progressBar");
  const msg = document.getElementById("progressMessage");

  container.style.display = "block";
  bar.style.width = `${percent}%`;
  bar.setAttribute("aria-valuenow", percent);
  msg.textContent = message;
}

function updateProgress(percent, message) {
  const bar = document.getElementById("progressBar");
  const msg = document.getElementById("progressMessage");

  bar.style.width = `${percent}%`;
  bar.setAttribute("aria-valuenow", percent);
  msg.textContent = message;
}

function hideProgress() {
  document.getElementById("progressContainer").style.display = "none";
}

function switchTab(tabId) {
  const tab = new bootstrap.Tab(document.getElementById(tabId));
  tab.show();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ============================================================================
// Event Listeners
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Initialize SocketIO
  initSocketIO();

  // Load saved Jira settings
  loadJiraSettings();

  // Setup tab
  document
    .getElementById("testConnectionBtn")
    .addEventListener("click", testJiraConnection);
  document
    .getElementById("jiraSettingsForm")
    .addEventListener("submit", saveJiraSettings);

  // Upload tab
  document
    .getElementById("imageFile")
    .addEventListener("change", handleImageSelect);
  document
    .getElementById("uploadImageBtn")
    .addEventListener("click", uploadImage);
  document
    .getElementById("processOcrBtn")
    .addEventListener("click", startOCRProcessing);

  // OCR tab
  document
    .getElementById("proceedToMappingBtn")
    .addEventListener("click", () => {
      switchTab("mapping-tab");
    });

  // Mapping tab
  document.getElementById("mappingProject").addEventListener("change", (e) => {
    loadIssueTypes(e.target.value);
  });

  // New session button
  document.getElementById("newSessionBtn").addEventListener("click", () => {
    if (
      confirm("Start a new import session? This will clear all current data.")
    ) {
      fetch("/api/session/new", { method: "POST" })
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            location.reload();
          }
        });
    }
  });
});
