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
        // Load projects to populate default project dropdown
        loadProjects();
      }
    })
    .catch((error) => console.error("Failed to load Jira settings:", error));
}

function populateJiraForm(settings) {
  document.getElementById("serverUrl").value = settings.server_url || "";
  document.getElementById("apiToken").value = settings.api_token || "";
  document.getElementById("defaultProject").value =
    settings.default_project_key || "";

  // Store field defaults in appState for later rendering
  if (settings.field_defaults) {
    appState.jiraSettings = appState.jiraSettings || {};
    appState.jiraSettings.field_defaults = settings.field_defaults;
  }

  // Auto-populate field defaults project and issue type if saved
  if (settings.field_defaults_project) {
    document.getElementById("fieldDefaultsProject").value =
      settings.field_defaults_project;
  }
  if (settings.field_defaults_issue_type) {
    document.getElementById("fieldDefaultsIssueType").value =
      settings.field_defaults_issue_type;
  }

  // Auto-load field defaults if both project and issue type are set
  if (settings.field_defaults_project && settings.field_defaults_issue_type) {
    // Wait for projects to load first, then load fields
    setTimeout(() => {
      loadFieldDefaults();
    }, 500);
  }
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
        // Reload projects to update both dropdowns
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
  // Collect field defaults from UI inputs
  const fieldDefaults = {};
  document.querySelectorAll(".field-default-input").forEach((input) => {
    const fieldId = input.dataset.fieldId;
    const fieldType = input.dataset.fieldType;
    const value = input.value.trim();

    if (value) {
      // Handle different field types
      if (fieldType === "option" || fieldType === "array") {
        // For select/option fields, value is already the ID
        if (input.tagName === "SELECT") {
          const selectedOption = input.options[input.selectedIndex];
          if (selectedOption && selectedOption.dataset.valueFormat) {
            // Use the data-value-format to determine structure
            fieldDefaults[fieldId] = JSON.parse(
              selectedOption.dataset.valueFormat
            );
          } else if (value !== "") {
            fieldDefaults[fieldId] = { id: value };
          }
        } else {
          fieldDefaults[fieldId] = value;
        }
      } else if (fieldType === "number") {
        fieldDefaults[fieldId] = parseFloat(value);
      } else {
        fieldDefaults[fieldId] = value;
      }
    }
  });

  return {
    server_url: document.getElementById("serverUrl").value,
    api_token: document.getElementById("apiToken").value,
    default_project_key: document.getElementById("defaultProject").value,
    field_defaults: fieldDefaults,
    field_defaults_project: document.getElementById("fieldDefaultsProject")
      .value,
    field_defaults_issue_type: document.getElementById("fieldDefaultsIssueType")
      .value,
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

        if (data.projects.length === 0) {
          showAlert(
            "No Jira projects found. Please check your Jira connection.",
            "warning"
          );
        }
      } else {
        showAlert(`Failed to load projects: ${data.error}`, "danger");
        // Clear project select on error
        const input = document.getElementById("mappingProject");
        const datalist = document.getElementById("projectList");
        input.value = "";
        input.placeholder = "Jira not configured - go to Setup tab";
        datalist.innerHTML = "";
      }
    })
    .catch((error) => {
      console.error("Failed to load projects:", error);
      showAlert(
        "Failed to load projects. Please configure Jira connection in Setup tab.",
        "danger"
      );
      const input = document.getElementById("mappingProject");
      const datalist = document.getElementById("projectList");
      input.value = "";
      input.placeholder = "Jira not configured - go to Setup tab";
      datalist.innerHTML = "";
    });
}

function populateProjectSelect() {
  const mappingInput = document.getElementById("mappingProject");
  const mappingDatalist = document.getElementById("projectList");
  const defaultInput = document.getElementById("defaultProject");
  const defaultDatalist = document.getElementById("defaultProjectList");
  const fieldDefaultsInput = document.getElementById("fieldDefaultsProject");
  const fieldDefaultsDatalist = document.getElementById(
    "fieldDefaultsProjectList"
  );

  mappingDatalist.innerHTML = "";
  if (defaultDatalist) {
    defaultDatalist.innerHTML = "";
  }
  if (fieldDefaultsDatalist) {
    fieldDefaultsDatalist.innerHTML = "";
  }

  appState.projects.forEach((project) => {
    // Populate mapping tab datalist
    const mappingOption = document.createElement("option");
    mappingOption.value = project.key;
    mappingOption.textContent = `${project.key} - ${project.name}`;
    mappingDatalist.appendChild(mappingOption);

    // Populate default project datalist in setup tab
    if (defaultDatalist) {
      const defaultOption = document.createElement("option");
      defaultOption.value = project.key;
      defaultOption.textContent = `${project.key} - ${project.name}`;
      defaultDatalist.appendChild(defaultOption);
    }

    // Populate field defaults project datalist
    if (fieldDefaultsDatalist) {
      const fieldDefaultOption = document.createElement("option");
      fieldDefaultOption.value = project.key;
      fieldDefaultOption.textContent = `${project.key} - ${project.name}`;
      fieldDefaultsDatalist.appendChild(fieldDefaultOption);
    }
  });

  // Auto-select default project in mapping tab if configured
  if (appState.jiraSettings && appState.jiraSettings.default_project_key) {
    const defaultProject = appState.projects.find(
      (p) => p.key === appState.jiraSettings.default_project_key
    );
    if (defaultProject) {
      mappingInput.value = defaultProject.key;
      // Trigger issue types loading for default project
      loadIssueTypes(defaultProject.key);
    }
  }
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
  const individualContainer = document.getElementById("individualMappings");
  const projectKey = document.getElementById("mappingProject").value;

  if (!projectKey || !appState.issueTypes[projectKey]) {
    container.innerHTML =
      '<p class="text-muted">Select a project to configure mappings.</p>';
    individualContainer.innerHTML = "";
    return;
  }

  // Get unique colors using Set based on hex value
  const uniqueColorHexes = [
    ...new Set(appState.ocrRegions.map((r) => r.color_hex)),
  ];

  console.log("Total regions:", appState.ocrRegions.length);
  console.log("Unique color hexes:", uniqueColorHexes);

  // Create color map with unique colors only
  const colorMap = new Map();
  uniqueColorHexes.forEach((hex) => {
    // Find first region with this color to get the name
    const region = appState.ocrRegions.find((r) => r.color_hex === hex);
    if (region) {
      colorMap.set(hex, region.color_name || "Unknown");
    }
  });

  console.log("Color map entries:", colorMap.size);

  // Render color-to-issue-type mappings
  let colorHtml = "<h6>Color to Issue Type Mapping</h6>";
  colorMap.forEach((name, hex) => {
    colorHtml += `
      <div class="color-mapping-row mb-2">
        <span class="color-badge" style="background-color: ${hex};"></span>
        <label class="me-2">${name}</label>
        <select class="form-select color-mapping-select" data-color="${hex}" style="max-width: 300px;">
          <option value="">Select issue type...</option>
          ${appState.issueTypes[projectKey]
            .map((it) => `<option value="${it.name}">${it.name}</option>`)
            .join("")}
        </select>
      </div>
    `;
  });
  container.innerHTML = colorHtml;

  // Add change listeners to auto-apply mappings to all regions with that color
  document.querySelectorAll(".color-mapping-select").forEach((select) => {
    select.addEventListener("change", (e) => {
      const color = e.target.dataset.color;
      const issueType = e.target.value;

      // Apply this issue type to all regions with this color
      appState.ocrRegions.forEach((region) => {
        if (region.color_hex === color) {
          region.issue_type = issueType;
        }
      });

      // Refresh individual mappings to show the updated selections
      renderIndividualMappings();
    });
  });

  // Render individual issue mappings
  renderIndividualMappings();
}

function renderIndividualMappings() {
  const individualContainer = document.getElementById("individualMappings");
  const projectKey = document.getElementById("mappingProject").value;

  if (!projectKey || !appState.issueTypes[projectKey]) {
    return;
  }

  // Collect current color mappings
  const colorMappings = {};
  document.querySelectorAll(".color-mapping-select").forEach((select) => {
    const color = select.dataset.color;
    const issueType = select.value;
    if (issueType) {
      colorMappings[color] = issueType;
    }
  });

  // Render individual regions with inherited or overridden issue types
  let html =
    '<div class="table-responsive"><table class="table table-sm table-striped"><thead><tr>';
  html +=
    "<th>Color</th><th>Text Preview</th><th>Issue Type</th></tr></thead><tbody>";

  appState.ocrRegions.forEach((region, index) => {
    const inheritedType = colorMappings[region.color_hex] || "";
    const currentType = region.issue_type || inheritedType;
    const textPreview =
      (region.text || "").substring(0, 60) +
      (region.text && region.text.length > 60 ? "..." : "");

    html += `
      <tr>
        <td><span class="color-badge" style="background-color: ${
          region.color_hex
        };"></span></td>
        <td>${textPreview}</td>
        <td>
          <select class="form-select form-select-sm individual-mapping-select" data-index="${index}" style="min-width: 150px;">
            <option value="">Select issue type...</option>
            ${appState.issueTypes[projectKey]
              .map(
                (it) =>
                  `<option value="${it.name}" ${
                    it.name === currentType ? "selected" : ""
                  }>${it.name}</option>`
              )
              .join("")}
          </select>
        </td>
      </tr>
    `;
  });

  html += "</tbody></table></div>";
  individualContainer.innerHTML = html;

  // Add change listeners for individual overrides
  document.querySelectorAll(".individual-mapping-select").forEach((select) => {
    select.addEventListener("change", (e) => {
      const index = parseInt(e.target.dataset.index);
      const issueType = e.target.value;
      if (appState.ocrRegions[index]) {
        appState.ocrRegions[index].issue_type = issueType;
      }
    });
  });
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

function renderImportResults(data) {
  const container = document.getElementById("importResults");

  // Build results HTML
  let html = `
    <div class="alert alert-success">
      <h5 class="alert-heading">Import Complete!</h5>
      <p class="mb-0">
        Created: <strong>${data.created || 0}</strong> | 
        Updated: <strong>${data.updated || 0}</strong> | 
        Failed: <strong>${data.failed || 0}</strong> | 
        Total: <strong>${data.total || 0}</strong>
      </p>
    </div>
  `;

  // Show results if available
  if (data.results && data.results.length > 0) {
    html +=
      '<div class="table-responsive"><table class="table table-striped table-sm">';
    html +=
      "<thead><tr><th>Status</th><th>Issue Key</th><th>Summary</th><th>Error</th></tr></thead><tbody>";

    data.results.forEach((result) => {
      const statusBadge = result.success
        ? '<span class="badge bg-success">✓ Success</span>'
        : '<span class="badge bg-danger">✗ Failed</span>';

      const issueKeyLink = result.issue_key
        ? `<a href="${appState.jiraSettings.server_url}/browse/${result.issue_key}" target="_blank">${result.issue_key}</a>`
        : "-";

      const error = result.error ? escapeHtml(result.error) : "-";
      const summary = escapeHtml(result.summary || "Unknown");

      html += `<tr>
        <td>${statusBadge}</td>
        <td>${issueKeyLink}</td>
        <td>${summary}</td>
        <td class="text-danger small">${error}</td>
      </tr>`;
    });

    html += "</tbody></table></div>";
  }

  container.innerHTML = html;
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
      // Load projects when navigating to mapping tab
      loadProjects();
    });

  // Mapping tab - listen for tab shown event to load projects
  document
    .getElementById("mapping-tab")
    .addEventListener("shown.bs.tab", () => {
      // Load projects if not already loaded
      if (!appState.projects || appState.projects.length === 0) {
        loadProjects();
      }
    });

  // Issue Review tab - load issues from database when shown
  document.getElementById("issues-tab").addEventListener("shown.bs.tab", () => {
    loadIssuesFromDatabase();
  });

  // Mapping tab
  document.getElementById("mappingProject").addEventListener("change", (e) => {
    loadIssueTypes(e.target.value);
  });

  // Mapping form submit - prevent default and show success message
  document.getElementById("mappingForm").addEventListener("submit", (e) => {
    e.preventDefault();
    showAlert("Mappings saved successfully!", "success", 3000);
  });

  // Proceed to Issue Review button
  document
    .getElementById("proceedToIssuesBtn")
    .addEventListener("click", () => {
      applyColorMappingsAndProceed();
    });

  // Start Import button
  document.getElementById("startImportBtn").addEventListener("click", () => {
    startJiraImport();
  });

  // Load Fields button for field defaults
  document.getElementById("loadFieldsBtn").addEventListener("click", () => {
    loadFieldDefaults();
  });

  // Save Field Defaults button
  document
    .getElementById("saveFieldDefaultsBtn")
    .addEventListener("click", () => {
      saveCurrentFieldDefaults();
    });

  // Populate field defaults project dropdown when projects load
  document
    .getElementById("fieldDefaultsProject")
    .addEventListener("change", () => {
      const projectKey = document.getElementById("fieldDefaultsProject").value;
      if (projectKey) {
        // Load issue types for the selected project could go here
      }
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

function applyColorMappingsAndProceed() {
  const projectKey = document.getElementById("mappingProject").value;

  if (!projectKey) {
    showAlert("Please select a project first", "warning");
    return;
  }

  // Validate that all regions have issue types assigned (either from color mapping or individual override)
  const unmappedRegions = [];
  appState.ocrRegions.forEach((region, index) => {
    if (!region.issue_type) {
      unmappedRegions.push(index + 1);
    }
  });

  if (unmappedRegions.length > 0) {
    showAlert(
      `Please assign issue types to all colors in the mapping section above. ${unmappedRegions.length} issue(s) still need mapping.`,
      "warning"
    );
    return;
  }

  // Set project key for all regions
  appState.ocrRegions.forEach((region) => {
    region.project_key = projectKey;
  });

  // Save mappings to database
  const updates = appState.ocrRegions.map((region) => ({
    id: region.db_id,
    project_key: projectKey,
    issue_type: region.issue_type,
  }));

  fetch("/api/issues/bulk-update-mapping", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ updates }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        // Navigate to review tab
        switchTab("issues-tab");

        // Populate review table
        populateReviewTable();
      } else {
        showAlert("Failed to save mappings", "danger");
      }
    })
    .catch((error) => {
      console.error("Failed to save mappings:", error);
      showAlert("Failed to save mappings", "danger");
    });
}

function populateReviewTable() {
  const tbody = document.querySelector("#issuesTable tbody");
  tbody.innerHTML = "";

  appState.ocrRegions.forEach((region, index) => {
    const issueKeyCell = region.issue_key
      ? `<a href="${appState.jiraSettings?.server_url || ""}/browse/${
          region.issue_key
        }" target="_blank" class="badge bg-success">${region.issue_key}</a>`
      : '<span class="badge bg-secondary">-</span>';

    const row = tbody.insertRow();
    row.innerHTML = `
      <td>${index + 1}</td>
      <td><span class="color-badge" style="background-color: ${
        region.color_hex
      };"></span></td>
      <td>${issueKeyCell}</td>
      <td>${region.issue_type || "N/A"}</td>
      <td contenteditable="true" data-field="summary" data-id="${index}">${
      region.text || ""
    }</td>
      <td contenteditable="true" data-field="description" data-id="${index}">${
      region.linked_text || ""
    }</td>
      <td>${Math.round(region.confidence || 0)}%</td>
      <td>
        <button class="btn btn-sm btn-danger" onclick="deleteIssue(${index})">Delete</button>
      </td>
    `;
  });

  // Add inline editing handlers
  document
    .querySelectorAll("#issuesTable [contenteditable]")
    .forEach((cell) => {
      cell.addEventListener("blur", (e) => {
        const id = parseInt(e.target.dataset.id);
        const field = e.target.dataset.field;
        const value = e.target.textContent;
        if (appState.ocrRegions[id]) {
          if (field === "summary") {
            appState.ocrRegions[id].text = value;
          } else if (field === "description") {
            appState.ocrRegions[id].linked_text = value;
          }

          // Update database if db_id exists
          if (appState.ocrRegions[id].db_id) {
            const updateData = {};
            updateData[field] = value;
            fetch(`/api/issues/${appState.ocrRegions[id].db_id}`, {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(updateData),
            }).catch((error) =>
              console.error("Failed to update issue:", error)
            );
          }
        }
      });
    });

  // Update issue count badge
  document.getElementById("issueCount").textContent =
    appState.ocrRegions.length;
}

function loadIssuesFromDatabase() {
  fetch("/api/issues/preview")
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.issues && data.issues.length > 0) {
        // Merge database issues with current OCR regions
        // If OCR regions exist (from current session), keep them
        // Otherwise, load from database
        if (appState.ocrRegions.length === 0) {
          appState.ocrRegions = data.issues;
        } else {
          // Update existing regions with database data (issue_key, etc.)
          appState.ocrRegions.forEach((region) => {
            const dbIssue = data.issues.find((i) => i.db_id === region.db_id);
            if (dbIssue) {
              region.issue_key = dbIssue.issue_key;
              region.project_key = dbIssue.project_key;
              region.issue_type = dbIssue.issue_type;
            }
          });
        }
        populateReviewTable();
      }
    })
    .catch((error) => console.error("Failed to load issues:", error));
}

function deleteIssue(index) {
  if (confirm("Delete this issue?")) {
    appState.ocrRegions.splice(index, 1);
    populateReviewTable();
  }
}

function updateImportSummary() {
  const issueCount = appState.ocrRegions.length;
  document.getElementById("issueCount").textContent = issueCount;
}

function startJiraImport() {
  if (appState.ocrRegions.length === 0) {
    showAlert("No issues to import", "warning");
    return;
  }

  const importBtn = document.getElementById("startImportBtn");
  importBtn.disabled = true;
  importBtn.textContent = "Importing...";

  showProgress(0, "Starting Jira import...");

  // Prepare issues data
  const issues = appState.ocrRegions.map((region, index) => ({
    db_id: region.db_id, // Database ID for updating records
    id: region.id, // Region ID for reference
    issue_key: region.issue_key, // Existing Jira key if already imported
    project_key: region.project_key,
    issue_type: region.issue_type,
    summary: region.text || "No summary",
    description: region.linked_text || "",
    color: region.color_hex,
  }));

  fetch("/api/jira/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ issues: issues }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        const message =
          data.failed > 0
            ? `Import complete! Created ${data.created}, updated ${
                data.updated || 0
              }, ${data.failed} failed.`
            : `Successfully imported to Jira! Created ${
                data.created
              }, updated ${data.updated || 0}.`;

        const alertType = data.failed > 0 ? "warning" : "success";
        showAlert(message, alertType);
        showProgress(100, "Import complete!");

        // Update regions with issue keys if returned
        if (data.results) {
          data.results.forEach((result, index) => {
            if (result.issue_key && appState.ocrRegions[index]) {
              appState.ocrRegions[index].issue_key = result.issue_key;
            }
          });
          // Refresh the review table to show updated issue keys
          populateReviewTable();
        }

        // Show detailed errors if any failed
        if (data.failed > 0 && data.results) {
          const errors = data.results
            .filter((r) => !r.success)
            .map((r) => `• ${r.summary}: ${r.error}`)
            .join("\n");
          console.error("Import errors:\n" + errors);
          showAlert(
            `${data.failed} issue(s) failed. Check console for details.`,
            "danger",
            10000
          );
        }
      } else {
        showAlert(`Import failed: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      showAlert(`Import error: ${error.message}`, "danger");
    })
    .finally(() => {
      importBtn.disabled = false;
      importBtn.textContent = "Start Import";
      hideProgress();
    });
}

function loadFieldDefaults() {
  const projectKey = document.getElementById("fieldDefaultsProject").value;
  const issueType = document.getElementById("fieldDefaultsIssueType").value;

  if (!projectKey || !issueType) {
    showAlert("Please select both project and issue type", "warning");
    return;
  }

  showAlert("Loading fields...", "info", 2000);

  fetch(`/api/jira/fields?project_key=${projectKey}&issue_type=${issueType}`)
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        renderFieldDefaultsUI(data.fields);

        // Load existing defaults for this project/issue type if they exist
        return fetch(`/api/field-defaults/${projectKey}/${issueType}`);
      } else {
        showAlert(`Failed to load fields: ${data.error}`, "danger");
        throw new Error(data.error);
      }
    })
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.field_defaults) {
        // Populate existing values
        for (const [fieldId, fieldValue] of Object.entries(
          data.field_defaults
        )) {
          const input = document.querySelector(`[data-field-id="${fieldId}"]`);
          if (input) {
            if (input.tagName === "SELECT") {
              const valueToSelect = fieldValue.id || fieldValue;
              input.value = valueToSelect;
            } else {
              input.value = fieldValue;
            }
          }
        }
        showAlert(
          `Loaded existing configuration for ${issueType}`,
          "info",
          3000
        );
      }
      // Show save button
      document.getElementById("saveFieldDefaultsSection").style.display =
        "block";
    })
    .catch((error) => {
      showAlert(`Error loading fields: ${error.message}`, "danger");
    });
}

function saveCurrentFieldDefaults() {
  const projectKey = document.getElementById("fieldDefaultsProject").value;
  const issueType = document.getElementById("fieldDefaultsIssueType").value;

  if (!projectKey || !issueType) {
    showAlert("Please select both project and issue type", "warning");
    return;
  }

  // Collect field defaults from UI
  const fieldDefaults = {};
  document.querySelectorAll(".field-default-input").forEach((input) => {
    const fieldId = input.dataset.fieldId;
    const fieldType = input.dataset.fieldType;
    const value = input.value.trim();

    if (value) {
      // Handle different field types
      if (fieldType === "option" || fieldType === "array") {
        if (input.tagName === "SELECT") {
          const selectedOption = input.options[input.selectedIndex];
          if (selectedOption && selectedOption.dataset.valueFormat) {
            fieldDefaults[fieldId] = JSON.parse(
              selectedOption.dataset.valueFormat
            );
          } else if (value !== "") {
            fieldDefaults[fieldId] = { id: value };
          }
        } else {
          fieldDefaults[fieldId] = value;
        }
      } else if (fieldType === "number") {
        fieldDefaults[fieldId] = parseFloat(value);
      } else {
        fieldDefaults[fieldId] = value;
      }
    }
  });

  // Save to database
  fetch("/api/field-defaults", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_key: projectKey,
      issue_type: issueType,
      field_defaults: fieldDefaults,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showAlert(
          `Field defaults saved for ${projectKey}/${issueType}`,
          "success",
          3000
        );
        loadConfiguredIssueTypes();
      } else {
        showAlert(`Failed to save: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      showAlert(`Save error: ${error.message}`, "danger");
    });
}

function loadConfiguredIssueTypes() {
  const projectKey = document.getElementById("fieldDefaultsProject").value;
  if (!projectKey) return;

  fetch(`/api/field-defaults?project_key=${projectKey}`)
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.configs && data.configs.length > 0) {
        const container = document.getElementById("configuredTypesContainer");
        const list = document.getElementById("configuredTypesList");

        let html = "<ul class='list-unstyled mb-0'>";
        data.configs.forEach((config) => {
          html += `<li class='small'><span class='badge bg-success me-2'>${
            config.issue_type
          }</span> (${Object.keys(config.field_defaults).length} fields)</li>`;
        });
        html += "</ul>";

        list.innerHTML = html;
        container.style.display = "block";
      }
    });
}

function renderFieldDefaultsUI(fields) {
  const container = document.getElementById("fieldDefaultsContainer");

  if (fields.length === 0) {
    container.innerHTML =
      '<p class="text-muted small">No additional fields found for this issue type.</p>';
    return;
  }

  let html =
    '<div class="border rounded p-2" style="max-height: 300px; overflow-y: auto;">';

  // Get existing field defaults
  const existingDefaults = appState.jiraSettings?.field_defaults || {};

  fields.forEach((field) => {
    const fieldId = field.id;
    const fieldName = field.name;
    const isRequired = field.required;
    const existingValue = existingDefaults[fieldId];

    html += `<div class="mb-2 pb-2 border-bottom">`;
    html += `<label class="form-label small mb-1"><strong>${fieldName}</strong>`;
    if (isRequired) {
      html += ' <span class="badge bg-danger">Required</span>';
    }
    html += `</label>`;

    // Render input based on field type and allowed values
    if (field.allowedValues && field.allowedValues.length > 0) {
      // Dropdown for fields with allowed values
      html += `<select class="form-select form-select-sm field-default-input" 
                      data-field-id="${fieldId}" 
                      data-field-type="${field.type || "option"}">`;
      html += `<option value="">-- Select ${fieldName} --</option>`;

      field.allowedValues.forEach((allowedValue) => {
        const valueId = allowedValue.id || allowedValue.value;
        const valueName = allowedValue.name || allowedValue.value;
        const isSelected =
          existingValue &&
          (existingValue.id === valueId || existingValue === valueId);

        html += `<option value="${valueId}" 
                        data-value-format='${JSON.stringify({ id: valueId })}'
                        ${isSelected ? "selected" : ""}>
                  ${valueName}
                 </option>`;
      });

      html += `</select>`;
    } else {
      // Text input for other fields
      const inputType = field.type === "number" ? "number" : "text";
      const value = existingValue || "";
      html += `<input type="${inputType}" 
                      class="form-control form-control-sm field-default-input" 
                      data-field-id="${fieldId}"
                      data-field-type="${field.type || "string"}"
                      value="${value}"
                      placeholder="Enter ${fieldName}">`;
    }

    html += `<div class="form-text" style="font-size: 0.75rem;">Field ID: ${fieldId}`;
    if (fieldId === "reporter") {
      html += ` <span class="text-info">• If empty, defaults to API token user</span>`;
    }
    html += `</div>`;
    html += `</div>`;
  });

  html += "</div>";
  container.innerHTML = html;
}
