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
  maxRegionId: 0, // Track highest region ID for multi-image support
  failedIssues: [], // Store failed imports for retry
  selectedImages: [], // Images selected for upload
  uploadedImages: [], // Images that have been uploaded
  processNextImage: null, // Function to continue processing queue
};

// ============================================================================
// SocketIO Connection
// ============================================================================

function initSocketIO() {
  appState.socket = io();

  appState.socket.on("connect", () => {
    console.log("Connected to server via SocketIO");
    showToast("Connected to server", "success", 3000);
  });

  appState.socket.on("disconnect", () => {
    console.log("Disconnected from server");
    showToast("Disconnected from server", "warning");
  });

  // OCR Progress
  appState.socket.on("ocr_progress", (data) => {
    console.log("OCR Progress:", data);

    // Calculate overall progress for multi-image processing
    let overallPercent = data.percent;
    let overallMessage = data.message;

    const totalImages = appState.uploadedImages
      ? appState.uploadedImages.length
      : 1;

    if (totalImages > 1 && appState.currentImage) {
      // Multi-image: calculate combined progress
      const currentImageIndex = appState.uploadedImages.indexOf(
        appState.currentImage
      );
      const imagesCompleted = currentImageIndex; // Images before current one
      const currentImageProgress = data.percent; // Progress on current image (0-100)

      // Overall = (completed images + current image progress) / total images
      overallPercent =
        (imagesCompleted * 100 + currentImageProgress) / totalImages;
      overallMessage = `Image ${currentImageIndex + 1}/${totalImages}: ${
        data.message
      }`;
    }

    updateProgress(overallPercent, overallMessage);

    if (data.status === "complete") {
      // OCR service already saved to database and included db_id in regions
      const currentImageFilename = appState.currentImage;
      const newRegions = data.regions.map((region) => {
        appState.maxRegionId++;
        return {
          ...region,
          id: appState.maxRegionId,
          image_filename: currentImageFilename,
          text: region.text || "",
          linked_text: region.linked_text || "",
          // db_id is already included by OCR service
        };
      });
      appState.ocrRegions.push(...newRegions);

      hideProgress();
      renderOCRResults();
      showToast(
        `OCR processing complete! Found ${newRegions.length} regions (${appState.ocrRegions.length} total)`,
        "success"
      );

      // Update badges
      updateTabBadge("upload", "complete");
      updateTabBadge("ocr", "count", appState.ocrRegions.length);

      // If processing multiple images, continue to next
      if (appState.processNextImage) {
        console.log(
          `SocketIO complete: calling processNextImage for next image`
        );
        appState.processNextImage();
      } else {
        // Single image workflow - auto-advance to OCR Review tab
        console.log(
          `SocketIO complete: single image mode, auto-advancing to OCR tab`
        );
        setTimeout(() => switchTab("ocr-tab"), 1000);
      }
    } else if (data.status === "error") {
      hideProgress();
      // User-friendly error message
      const userMsg = data.message.includes("PaddleOCR")
        ? "OCR engine failed - please try a different image or restart the application"
        : data.message.includes("FileNotFoundError")
        ? "Image file not found - please upload again"
        : data.message.includes("OutOfMemory")
        ? "Image too large - please use a smaller image (max 2000px)"
        : "OCR processing failed - please try again";
      showToast(userMsg, "danger");
      console.error("OCR Error Details:", data.message);

      // If processing multiple images, continue to next even on error
      if (appState.processNextImage) {
        appState.processNextImage();
      }
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
      showToast("Import complete!", "success");
      switchTab("results-tab");

      // Update Results badge with created + updated count
      const totalImported = (data.created || 0) + (data.updated || 0);
      updateTabBadge("results", "count", totalImported);
    } else if (data.status === "error") {
      hideProgress();
      // User-friendly error message
      const userMsg = data.message.includes("JIRAError")
        ? "Jira connection failed - please check credentials in Setup tab"
        : data.message.includes("404")
        ? "Jira project not found - verify project key"
        : data.message.includes("401") || data.message.includes("403")
        ? "Jira authentication failed - check API token"
        : data.message.includes("field")
        ? "Missing required Jira field - configure defaults in Setup tab"
        : "Import failed - check Results tab for details";
      showToast(userMsg, "danger");
      console.error("Import Error Details:", data.message);
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
        // Update action bar state (enables Import button if configured)
        if (typeof updateActionBar === "function") {
          updateActionBar();
        }
        // Update settings badge if configuration exists
        updateSettingsBadge();
      }
    })
    .catch((error) => console.error("Failed to load Jira settings:", error));
}

function updateSettingsBadge() {
  const badge = document.getElementById("settingsBadge");
  if (!badge) return;

  // Check if we have basic Jira configuration
  if (
    appState.jiraSettings &&
    appState.jiraSettings.server_url &&
    appState.jiraSettings.api_token
  ) {
    // Show checkmark to indicate configuration exists
    badge.innerHTML = '<span class="badge bg-success ms-1">✓</span>';
    badge.title = "Jira configured";
  } else {
    // Clear badge if no configuration
    badge.innerHTML = "";
    badge.title = "";
  }
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
    // Wait for projects to load first, then load summary panel
    setTimeout(() => {
      loadConfiguredDefaultsSummary();
    }, 500);
  } else {
    // Just load the summary panel to show what's configured
    setTimeout(() => {
      loadConfiguredDefaultsSummary();
    }, 500);
  }
}

function testJiraConnection() {
  const settings = getJiraFormData();

  showToast("Testing connection...", "info");

  fetch("/api/jira/test-connection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showToast(`Connected to ${data.server_title}`, "success");
      } else {
        showToast(`Connection failed: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      showToast(`Connection error: ${error.message}`, "danger");
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
        showToast("Settings saved successfully", "success");
        // Reload projects to update both dropdowns
        loadProjects();
        // Mark setup as complete
        updateTabBadge("setup", "complete");
        // Update settings button badge
        updateSettingsBadge();
        // Update action bar state (enables Import button)
        if (typeof updateActionBar === "function") {
          updateActionBar();
        }
      } else {
        showToast(`Failed to save settings: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      showToast(`Save error: ${error.message}`, "danger");
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

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;

  if (files.length > 0) {
    // Convert FileList to Array
    const filesArray = Array.from(files);

    // Validate all are image files
    const validFiles = filesArray.filter((file) => {
      if (!file.type.startsWith("image/")) {
        showToast(`Skipping ${file.name} - not an image file`, "warning");
        return false;
      }
      if (file.size > 5 * 1024 * 1024) {
        showToast(`Skipping ${file.name} - exceeds 5MB limit`, "warning");
        return false;
      }
      return true;
    });

    if (validFiles.length === 0) {
      showToast("No valid image files dropped", "danger");
      return;
    }

    // Append to existing selection
    if (!appState.selectedImages) {
      appState.selectedImages = [];
    }
    appState.selectedImages.push(...validFiles);

    // Show gallery preview with all selected images
    displayImageGallery(appState.selectedImages);

    // Enable upload button
    document.getElementById("uploadImageBtn").disabled = false;
    document.getElementById("uploadBtnCount").textContent =
      appState.selectedImages.length;
    document.getElementById("uploadBtnCount").style.display = "inline";

    showToast(
      `Added ${validFiles.length} image(s). Total: ${appState.selectedImages.length}`,
      "info",
      3000
    );
  }
}

function handlePaste(e) {
  // Only handle paste on upload tab and if not typing in an input
  const activeTab = document.querySelector(".tab-pane.active");
  if (!activeTab || activeTab.id !== "upload") return;

  const target = e.target;
  if (
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.contentEditable === "true"
  ) {
    return; // Don't intercept paste in editable fields
  }

  const items = e.clipboardData?.items;
  if (!items) return;

  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf("image") !== -1) {
      e.preventDefault();
      const blob = items[i].getAsFile();

      // Create File object from blob
      const file = new File([blob], `pasted-image-${Date.now()}.png`, {
        type: blob.type,
      });

      // Validate file size
      if (file.size > 5 * 1024 * 1024) {
        showToast("Pasted image exceeds 5MB limit", "warning");
        return;
      }

      // Append to selection
      if (!appState.selectedImages) {
        appState.selectedImages = [];
      }
      appState.selectedImages.push(file);

      // Update gallery
      displayImageGallery(appState.selectedImages);

      // Enable upload button
      document.getElementById("uploadImageBtn").disabled = false;
      document.getElementById("uploadBtnCount").textContent =
        appState.selectedImages.length;
      document.getElementById("uploadBtnCount").style.display = "inline";

      showToast(
        `Image pasted from clipboard. Total: ${appState.selectedImages.length}`,
        "info",
        2000
      );
      break;
    }
  }
}

function handleImageSelect(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;

  // Validate all files
  const validFiles = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    // Validate file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
      showToast(
        `File ${file.name} is too large. Maximum size is 5MB per image.`,
        "warning"
      );
      continue;
    }

    validFiles.push(file);
  }

  if (validFiles.length === 0) {
    showToast("No valid images selected.", "danger");
    return;
  }

  // Append new files to existing selection (allow adding more images)
  if (!appState.selectedImages) {
    appState.selectedImages = [];
  }
  appState.selectedImages.push(...validFiles);

  // Show gallery preview with all selected images
  displayImageGallery(appState.selectedImages);

  // Enable upload button
  document.getElementById("uploadImageBtn").disabled = false;
  document.getElementById("uploadBtnCount").textContent =
    appState.selectedImages.length;
  document.getElementById("uploadBtnCount").style.display = "inline";

  showToast(
    `Added ${validFiles.length} image(s). Total: ${appState.selectedImages.length}`,
    "info",
    3000
  );
}

function displayImageGallery(files) {
  const gallery = document.getElementById("previewGallery");
  const container = document.getElementById("imagePreviewGallery");
  const count = document.getElementById("imageCount");

  gallery.innerHTML = "";

  files.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const card = document.createElement("div");
      card.className = "card";
      card.style.width = "150px";
      card.innerHTML = `
        <img src="${
          e.target.result
        }" class="card-img-top" style="height: 100px; object-fit: cover;" alt="${
        file.name
      }">
        <div class="card-body p-2">
          <p class="card-text small text-truncate mb-0" title="${file.name}">${
        file.name
      }</p>
          <small class="text-muted">${(file.size / 1024).toFixed(1)} KB</small>
        </div>
      `;
      gallery.appendChild(card);
    };
    reader.readAsDataURL(file);
  });

  count.textContent = files.length;
  container.style.display = "block";
}

function clearImageGallery() {
  // Reset all image-related state
  appState.selectedImages = [];
  appState.uploadedImages = [];

  // Clear UI
  document.getElementById("previewGallery").innerHTML = "";
  document.getElementById("imagePreviewGallery").style.display = "none";
  document.getElementById("imageCount").textContent = "0";
  document.getElementById("uploadBtnCount").textContent = "0";
  document.getElementById("uploadBtnCount").style.display = "none";

  // Reset file input
  document.getElementById("imageFile").value = "";

  // Reset buttons
  document.getElementById("uploadImageBtn").disabled = true;
  setButtonState("#processOcrBtn", false, "Upload an image first");
}

function uploadImage() {
  const files = appState.selectedImages || [];
  if (files.length === 0) return;

  showProgress(0, `Uploading ${files.length} image(s)...`);

  let uploadedCount = 0;
  const uploadedFiles = [];

  // Upload files sequentially
  const uploadNext = (index) => {
    if (index >= files.length) {
      // All uploaded
      hideProgress();
      appState.uploadedImages = uploadedFiles;
      console.log(
        `Upload complete. Total files uploaded: ${uploadedFiles.length}`,
        uploadedFiles
      );
      setButtonState("#processOcrBtn", true);
      document.getElementById("uploadImageBtn").disabled = true;
      showToast(`${uploadedCount} image(s) uploaded successfully`, "success");
      return;
    }

    const file = files[index];
    const formData = new FormData();
    formData.append("file", file);

    const percent = Math.round(((index + 1) / files.length) * 100);
    showProgress(
      percent,
      `Uploading ${index + 1}/${files.length}: ${file.name}...`
    );

    fetch("/api/ocr/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          uploadedCount++;
          uploadedFiles.push(data.filename);
          uploadNext(index + 1);
        } else {
          hideProgress();
          showToast(`Upload failed for ${file.name}: ${data.error}`, "danger");
        }
      })
      .catch((error) => {
        hideProgress();
        showToast(`Upload error for ${file.name}: ${error.message}`, "danger");
      });
  };

  uploadNext(0);
}

function startOCRProcessing() {
  const images = appState.uploadedImages || [];
  console.log(`Starting OCR processing for ${images.length} images:`, images);
  if (images.length === 0) return;

  let currentIndex = 0;

  const processNext = () => {
    console.log(
      `processNext called: currentIndex=${currentIndex}, total=${images.length}`
    );
    if (currentIndex >= images.length) {
      // All processed
      appState.processNextImage = null;
      setButtonState("#processOcrBtn", false, "All images processed");

      // Auto-advance to OCR Review tab
      setTimeout(() => switchTab("ocr-tab"), 1000);
      return;
    }

    const filename = images[currentIndex];
    const imageNumber = currentIndex + 1;

    // Show initial progress for all images (single or multi-image)
    if (images.length === 1) {
      showProgress(0, `Processing ${filename}...`);
    } else {
      // Multi-image: show overall progress
      const overallPercent = Math.round((currentIndex / images.length) * 100);
      showProgress(
        overallPercent,
        `Processing image ${imageNumber}/${images.length}: ${filename}...`
      );
    }

    // Increment currentIndex now, before async operations
    currentIndex++;

    fetch("/api/ocr/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: filename }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          // Progress updates will come via SocketIO
          console.log(
            `OCR processing started for ${filename} (index ${imageNumber - 1}/${
              images.length
            })`
          );
          // Store current image for SocketIO handler
          appState.currentImage = filename;
          // SocketIO handler will call processNext() when OCR completes
        } else {
          hideProgress();
          showToast(`OCR failed for ${filename}: ${data.error}`, "danger");
          // Skip this image and move to next
          processNext();
        }
      })
      .catch((error) => {
        hideProgress();
        showToast(`OCR error: ${error.message}`, "danger");
        // Skip this image and move to next
        processNext();
      });
  };

  // Store processNext in appState so SocketIO handler can call it
  appState.processNextImage = processNext;
  processNext();
}

// ============================================================================
// OCR Results
// ============================================================================

function renderOCRResults() {
  const container = document.getElementById("ocrResults");
  const summary = document.getElementById("ocrSummary");

  if (appState.ocrRegions.length === 0) {
    container.innerHTML = '<p class="text-muted">No regions detected.</p>';
    summary.style.display = "none";
    return;
  }

  // Calculate statistics
  const totalRegions = appState.ocrRegions.length;
  const avgConfidence = Math.round(
    appState.ocrRegions.reduce((sum, r) => sum + (r.confidence || 0), 0) /
      totalRegions
  );
  const lowConfCount = appState.ocrRegions.filter(
    (r) => (r.confidence || 0) < 70
  ).length;

  // Color breakdown
  const colorCounts = {};
  appState.ocrRegions.forEach((r) => {
    colorCounts[r.color_hex] = (colorCounts[r.color_hex] || 0) + 1;
  });
  const colorCount = Object.keys(colorCounts).length;

  // Update summary
  document.getElementById("totalRegions").textContent = totalRegions;
  document.getElementById("avgConfidence").textContent = avgConfidence;
  document.getElementById("colorCount").textContent = colorCount;
  document.getElementById("lowConfCount").textContent = lowConfCount;

  const colorBreakdown = document.getElementById("colorBreakdown");
  colorBreakdown.innerHTML = Object.entries(colorCounts)
    .map(([hex, count]) => {
      const region = appState.ocrRegions.find((r) => r.color_hex === hex);
      const colorName = region ? region.color_name : "Unknown";
      return `<span class="badge" style="background-color: ${hex}; color: white;">${colorName}: ${count}</span>`;
    })
    .join("");

  summary.style.display = "block";

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
          showToast(
            "No Jira projects found. Please check your Jira connection.",
            "warning"
          );
        }
      } else {
        showToast(`Failed to load projects: ${data.error}`, "danger");
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
      showToast(
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

  // Load saved color mappings from database first
  fetch(`/api/mapping/load?project_key=${projectKey}`)
    .then((response) => response.json())
    .then((data) => {
      const savedMappings = data.success ? data.mappings : {};
      console.log(
        `Loaded ${Object.keys(savedMappings).length} saved color mappings`
      );
      renderColorMappingsUI(projectKey, savedMappings);
    })
    .catch((error) => {
      console.error("Failed to load color mappings:", error);
      renderColorMappingsUI(projectKey, {});
    });
}

function renderColorMappingsUI(projectKey, savedMappings = {}) {
  const container = document.getElementById("colorMappings");

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

  // Render color-to-issue-type mappings with saved values
  let colorHtml = "<h6>Color to Issue Type Mapping</h6>";
  colorMap.forEach((name, hex) => {
    const savedIssueType = savedMappings[hex] || "";

    colorHtml += `
      <div class="color-mapping-row mb-2">
        <span class="color-badge" style="background-color: ${hex};"></span>
        <label class="color-name-label">${name}</label>
        <select class="form-select color-mapping-select" data-color="${hex}" style="max-width: 300px;">
          <option value="">Select issue type...</option>
          ${appState.issueTypes[projectKey]
            .map(
              (it) =>
                `<option value="${it.name}" ${
                  it.name === savedIssueType ? "selected" : ""
                }>${it.name}</option>`
            )
            .join("")}
        </select>
      </div>
    `;
  });
  container.innerHTML = colorHtml;

  // Apply saved mappings to regions
  if (Object.keys(savedMappings).length > 0) {
    appState.ocrRegions.forEach((region) => {
      if (savedMappings[region.color_hex]) {
        region.issue_type = savedMappings[region.color_hex];
      }
    });
  }

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

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Show toast notification (replaces old alert system that pushed UI around)
 * @param {string} message - Message to display
 * @param {string} type - Bootstrap color variant: success, danger, warning, info
 * @param {number} duration - Auto-hide delay in ms (default: 5000)
 */
function showToast(message, type = "info", duration = 5000) {
  const toastEl = document.getElementById("appToast");
  const toastBody = document.getElementById("toastBody");
  const toastHeader = toastEl.querySelector(".toast-header");

  // Set message and color
  toastBody.textContent = message;
  toastHeader.className = `toast-header bg-${type}`;

  // Show toast
  const toast = new bootstrap.Toast(toastEl, {
    autohide: true,
    delay: duration,
  });
  toast.show();
}

/**
 * Set button enabled/disabled state with optional tooltip
 * @param {string} selector - CSS selector or element ID
 * @param {boolean} enabled - Whether button should be enabled
 * @param {string} tooltipText - Tooltip text when disabled (optional)
 */
function setButtonState(selector, enabled, tooltipText = "") {
  const btn =
    typeof selector === "string"
      ? document.querySelector(
          selector.startsWith("#") || selector.startsWith(".")
            ? selector
            : `#${selector}`
        )
      : selector;

  if (!btn) return;

  btn.disabled = !enabled;

  if (!enabled && tooltipText) {
    btn.setAttribute("data-bs-toggle", "tooltip");
    btn.setAttribute("data-bs-placement", "top");
    btn.setAttribute("title", tooltipText);
    // Initialize tooltip
    new bootstrap.Tooltip(btn);
  } else if (enabled) {
    // Remove tooltip when enabled
    btn.removeAttribute("data-bs-toggle");
    btn.removeAttribute("title");
    const tooltip = bootstrap.Tooltip.getInstance(btn);
    if (tooltip) tooltip.dispose();
  }
}

/**
 * Update tab badge content and variant
 * @param {string} tabId - Tab ID (e.g., 'setup', 'upload')
 * @param {string|number} content - Badge text/number
 * @param {string} variant - Bootstrap color variant (default: 'secondary')
 */
function updateTabBadge(tabId, content, variant = "secondary") {
  const badge = document.querySelector(`#${tabId}-tab .badge, #${tabId}-badge`);
  if (badge) {
    badge.textContent = content;
    badge.className = `badge bg-${variant} ms-2`;
  }
}

/**
 * Refresh DataTable with new data
 * Reusable function to avoid repetitive table refresh code
 */
function refreshIssuesTable() {
  if (issuesTable) {
    issuesTable.clear();
    issuesTable.rows.add(globalIssues);
    issuesTable.draw();
  }
}

/**
 * Show confirmation modal dialog
 * @param {string} message - Message to display
 * @param {string} title - Modal title (default: "Confirm Action")
 * @param {string} confirmBtnText - Confirm button text (default: "Confirm")
 * @param {string} confirmBtnVariant - Bootstrap color variant for confirm button (default: "danger")
 * @param {string} titleIcon - Optional icon to display before title (e.g., "⚠️", "🗑️", "✓")
 * @returns {Promise<boolean>} - Resolves to true if confirmed, false if canceled
 */
function showConfirm(
  message,
  title = "Confirm Action",
  confirmBtnText = "Confirm",
  confirmBtnVariant = "danger",
  titleIcon = ""
) {
  return new Promise((resolve) => {
    const modal = document.getElementById("confirmModal");
    const modalInstance = new bootstrap.Modal(modal);
    const titleEl = document.getElementById("confirmModalLabel");
    const bodyEl = document.getElementById("confirmModalBody");
    const confirmBtn = document.getElementById("confirmModalConfirmBtn");

    // Set content
    titleEl.innerHTML = titleIcon ? `${titleIcon} ${title}` : title;
    bodyEl.innerHTML = message.replace(/\n/g, "<br>");
    confirmBtn.textContent = confirmBtnText;
    confirmBtn.className = `btn btn-${confirmBtnVariant}`;

    // Handle confirm
    const handleConfirm = () => {
      modalInstance.hide();
      cleanup();
      resolve(true);
    };

    // Handle cancel/dismiss
    const handleCancel = () => {
      cleanup();
      resolve(false);
    };

    // Cleanup listeners
    const cleanup = () => {
      confirmBtn.removeEventListener("click", handleConfirm);
      modal.removeEventListener("hidden.bs.modal", handleCancel);
    };

    // Attach listeners
    confirmBtn.addEventListener("click", handleConfirm);
    modal.addEventListener("hidden.bs.modal", handleCancel, { once: true });

    // Show modal
    modalInstance.show();
  });
}

/**
 * Validate prerequisites for a tab
 * @param {string} tabId - Tab ID to validate
 * @returns {string|null} - Error message if invalid, null if valid
 */
function validateTabPrerequisites(tabId) {
  const prerequisites = {
    upload: () => {
      // Check if Jira is configured
      if (!appState.jiraSettings || !appState.jiraSettings.server_url) {
        return "💡 Tip: Configure Jira settings using the gear icon (⚙️) in the header to enable import functionality.";
      }
      return null;
    },
    setup: null,
    ocr: () => {
      // Check if Jira is configured first
      if (!appState.jiraSettings || !appState.jiraSettings.server_url) {
        return "💡 Tip: Configure Jira settings using the gear icon (⚙️) in the header before processing images.";
      }
      // Check if images uploaded
      return appState.uploadedImages.length > 0
        ? null
        : "Tip: Upload images in the Upload tab to get started";
    },
    mapping: () =>
      appState.ocrRegions.length > 0
        ? null
        : "Tip: Process an image with OCR first in the OCR Review tab",
    issues: () => {
      // Check if OCR regions exist and have issue types assigned
      if (appState.ocrRegions.length === 0) {
        return "Tip: Process an image with OCR first in the OCR Review tab";
      }
      // Check if all regions have issue types assigned (color mappings applied)
      const unmappedCount = appState.ocrRegions.filter(
        (r) => !r.issue_type
      ).length;
      if (unmappedCount > 0) {
        return "Tip: Configure color mappings in the Mapping tab first";
      }
      return null;
    },
    results: () => {
      // Check if import results container has content
      const resultsContainer = document.getElementById("importResults");
      return resultsContainer && resultsContainer.innerHTML.trim() !== ""
        ? null
        : "Tip: Import issues to Jira from the Issue Review tab first";
    },
  };

  const prerequisiteCheck = prerequisites[tabId];
  return prerequisiteCheck ? prerequisiteCheck() : null;
}

function showProgress(percent, message) {
  const container = document.getElementById("progressContainer");
  const messageContainer = document.getElementById("progressMessage");
  const bar = document.getElementById("progressBar");
  const text = document.getElementById("progressText");
  const percentText = document.getElementById("progressPercent");

  // Add a small delay before showing progress to avoid flash for quick operations
  clearTimeout(appState.progressTimeout);
  appState.progressTimeout = setTimeout(() => {
    // Show progress bar
    container.style.display = "block";
    bar.style.width = `${percent}%`;
    bar.setAttribute("aria-valuenow", percent);

    // Activate message container (fade in content)
    const messageContent = document.getElementById("progressMessageContent");
    messageContainer.style.backgroundColor = "#f8f9fa";
    messageContent.style.opacity = "1";
    text.textContent = message;
    percentText.textContent = `${Math.round(percent)}%`;
  }, 300); // Only show if operation takes longer than 300ms
}

function updateProgress(percent, message) {
  const container = document.getElementById("progressContainer");
  const messageContainer = document.getElementById("progressMessage");
  const messageContent = document.getElementById("progressMessageContent");
  const bar = document.getElementById("progressBar");
  const text = document.getElementById("progressText");
  const percentText = document.getElementById("progressPercent");

  // If progress isn't visible yet, cancel timeout and show immediately
  if (container.style.display === "none") {
    clearTimeout(appState.progressTimeout);
    container.style.display = "block";
    messageContainer.style.backgroundColor = "#f8f9fa";
    messageContent.style.opacity = "1";
  }

  bar.style.width = `${percent}%`;
  bar.setAttribute("aria-valuenow", percent);
  text.textContent = message;
  percentText.textContent = `${Math.round(percent)}%`;
}

function hideProgress() {
  // Cancel any pending progress display
  clearTimeout(appState.progressTimeout);

  const container = document.getElementById("progressContainer");
  const messageContainer = document.getElementById("progressMessage");
  const messageContent = document.getElementById("progressMessageContent");

  // Hide progress bar
  container.style.display = "none";

  // Deactivate message container (fade out content, keep space)
  messageContainer.style.backgroundColor = "transparent";
  messageContent.style.opacity = "0";

  // Reset for next use
  const bar = document.getElementById("progressBar");
  bar.style.width = "0%";
}
function switchTab(tabId) {
  // Extract the tab name without the '-tab' suffix for validation
  const tabName = tabId.replace("-tab", "");

  // Check prerequisites and show informative message if missing
  const tipMessage = validateTabPrerequisites(tabName);
  if (tipMessage) {
    showToast(tipMessage, "info", 4000);
  }

  // Always allow switching to the tab
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

  // Add retry button if there are failed issues
  if (data.failed && data.failed > 0) {
    html += `
      <div class="alert alert-warning">
        <h6><i class="bi bi-exclamation-triangle"></i> ${data.failed} issue(s) failed to import</h6>
        <button type="button" class="btn btn-warning btn-sm mt-2" id="retryFailedBtn">
          <i class="bi bi-arrow-clockwise"></i> Retry Failed Issues Only
        </button>
      </div>
    `;

    // Store failed issues for retry
    appState.failedIssues = data.results.filter((r) => !r.success);
  }

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

  // Attach retry button handler if button exists
  const retryBtn = document.getElementById("retryFailedBtn");
  if (retryBtn) {
    retryBtn.addEventListener("click", retryFailedImport);
  }
}

function retryFailedImport() {
  if (!appState.failedIssues || appState.failedIssues.length === 0) {
    showToast("No failed issues to retry", "info");
    return;
  }

  showToast(
    `Retrying ${appState.failedIssues.length} failed issue(s)...`,
    "info"
  );
  showProgress();

  // Find the original issues that failed
  const failedSummaries = appState.failedIssues.map((f) => f.summary);
  const issuesToRetry = appState.ocrRegions.filter((r) =>
    failedSummaries.includes(r.text)
  );

  // Start import for failed issues only
  fetch("/api/import/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ issues: issuesToRetry }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showToast("Retry started - check Results tab for progress", "success");
      } else {
        hideProgress();
        showToast(`Retry failed: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      hideProgress();
      showToast(`Retry error: ${error.message}`, "danger");
    });
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

  // Settings button in header - opens Settings tab
  document.getElementById("settingsBtn").addEventListener("click", () => {
    switchTab("setup-tab");
  });

  // Add tab prerequisite validation (informative, non-blocking)
  document.querySelectorAll('[data-bs-toggle="tab"]').forEach((tabTrigger) => {
    tabTrigger.addEventListener("click", (e) => {
      const targetId =
        tabTrigger.getAttribute("data-bs-target")?.replace("#", "") ||
        tabTrigger.getAttribute("id")?.replace("-tab", "");

      if (targetId) {
        const tipMessage = validateTabPrerequisites(targetId);
        if (tipMessage) {
          showToast(tipMessage, "info", 4000);
        }
      }
    });
  });

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
  document
    .getElementById("clearImagesBtn")
    .addEventListener("click", clearImageGallery);

  // Drag and drop functionality
  const dropZone = document.getElementById("dropZone");

  // Click to browse
  dropZone.addEventListener("click", () => {
    document.getElementById("imageFile").click();
  });

  // Prevent default drag behaviors
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  // Highlight drop zone when dragging over it
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.remove("drag-over");
    });
  });

  // Handle dropped files
  dropZone.addEventListener("drop", handleDrop);

  // Clipboard paste (Ctrl+V)
  document.addEventListener("paste", handlePaste);

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

  // Proceed to Issue Review button
  document
    .getElementById("proceedToIssuesBtn")
    .addEventListener("click", () => {
      applyColorMappingsAndProceed();
    });

  // Proceed to Results button
  document
    .getElementById("proceedToResultsBtn")
    .addEventListener("click", () => {
      switchTab("results-tab");
    });

  // Start Import button
  document.getElementById("startImportBtn").addEventListener("click", () => {
    startJiraImport();
  });

  // Setup Next button - go to Upload tab
  document.getElementById("setupNextBtn").addEventListener("click", () => {
    switchTab("upload-tab");
  });

  // Upload Next button - go to OCR Review tab (shown after OCR completes)
  document.getElementById("uploadNextBtn")?.addEventListener("click", () => {
    switchTab("ocr-tab");
  });

  // Confidence filter
  document
    .getElementById("confidenceFilter")
    ?.addEventListener("change", (e) => {
      filterByConfidence(e.target.value);
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

  // Add New Default button
  document.getElementById("addNewDefaultBtn").addEventListener("click", () => {
    showFieldEditor("new");
  });

  // Cancel Edit button
  document.getElementById("cancelEditBtn").addEventListener("click", () => {
    resetFieldEditor();
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

  // Bulk operations buttons
  document.getElementById("selectAllBtn")?.addEventListener("click", () => {
    document
      .querySelectorAll(".row-checkbox")
      .forEach((cb) => (cb.checked = true));
    updateActionBar();
  });

  document.getElementById("deselectAllBtn")?.addEventListener("click", () => {
    document
      .querySelectorAll(".row-checkbox")
      .forEach((cb) => (cb.checked = false));
    updateActionBar();
  });

  document.getElementById("bulkDeleteBtn")?.addEventListener("click", () => {
    const selectedIndices = Array.from(
      document.querySelectorAll(".row-checkbox:checked")
    )
      .map((cb) => parseInt(cb.dataset.index))
      .sort((a, b) => b - a); // Sort descending to delete from end

    if (selectedIndices.length === 0) return;

    showConfirm(
      `Delete ${selectedIndices.length} selected issue(s)?`,
      "Delete Issues",
      "Delete",
      "danger",
      "🗑️"
    ).then((confirmed) => {
      if (confirmed) {
        // Collect db_ids before deleting from frontend
        const dbIds = selectedIndices
          .map((index) => appState.ocrRegions[index]?.db_id)
          .filter((id) => id); // Filter out undefined/null

        // Delete from frontend state
        selectedIndices.forEach((index) => {
          appState.ocrRegions.splice(index, 1);
        });

        populateReviewTable();
        updateActionBar(); // Reset selection UI
        showToast(
          `Deleted ${selectedIndices.length} issue(s)`,
          "success",
          3000
        );

        // Delete from database
        if (dbIds.length > 0) {
          fetch("/api/issues/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ids: dbIds }),
          })
            .then((response) => response.json())
            .then((data) => {
              if (!data.success) {
                console.error("Failed to delete issues from database");
              }
            })
            .catch((error) => {
              console.error("Bulk delete error:", error);
            });
        }
      }
    });
  });

  // Delete All button
  document.getElementById("deleteAllBtn")?.addEventListener("click", () => {
    const totalIssues = appState.ocrRegions.length;
    if (totalIssues === 0) return;

    showConfirm(
      `Delete all ${totalIssues} issue(s)? This cannot be undone.`,
      "Delete All Issues",
      "Delete All",
      "danger",
      "🗑️"
    ).then((confirmed) => {
      if (confirmed) {
        // Collect all db_ids before clearing
        const dbIds = appState.ocrRegions
          .map((region) => region.db_id)
          .filter((id) => id); // Filter out undefined/null

        // Clear frontend state
        appState.ocrRegions = [];
        populateReviewTable();
        updateActionBar();
        showToast("All issues deleted", "success", 3000);

        // Delete from database
        if (dbIds.length > 0) {
          fetch("/api/issues/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ids: dbIds }),
          })
            .then((response) => response.json())
            .then((data) => {
              if (!data.success) {
                console.error("Failed to delete all issues from database");
              }
            })
            .catch((error) => {
              console.error("Delete all error:", error);
            });
        }
      }
    });
  });

  // New session button
  document.getElementById("newSessionBtn").addEventListener("click", () => {
    showConfirm(
      "This will permanently delete:<br>" +
        "• All OCR regions and imported issues<br>" +
        "• Color mappings<br>" +
        "• Import history<br>" +
        "• Uploaded images<br><br>" +
        "Your Jira configuration and field defaults will be preserved.<br><br>" +
        "Are you sure you want to start a new session?",
      "Clear Session",
      "Yes, Clear Everything",
      "danger",
      "⚠️"
    ).then((confirmed) => {
      if (confirmed) {
        fetch("/api/session/new", { method: "POST" })
          .then((response) => response.json())
          .then((data) => {
            if (data.success) {
              // Clear frontend state
              appState.ocrRegions = [];
              appState.colorMappings = {};
              appState.issues = [];
              appState.currentImage = null;
              appState.maxRegionId = 0;

              // Reload to reset UI
              location.reload();
            } else {
              showToast(`Failed to start new session: ${data.error}`, "danger");
            }
          })
          .catch((error) => {
            console.error("New session error:", error);
            showToast("Failed to start new session", "danger");
          });
      }
    });
  });

  // Global keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Ctrl+A - Select all issues (in Issue Review tab, but not when editing)
    if (
      e.ctrlKey &&
      e.key === "a" &&
      document.getElementById("issues").classList.contains("active") &&
      !e.target.hasAttribute("contenteditable") &&
      e.target.tagName !== "INPUT" &&
      e.target.tagName !== "TEXTAREA"
    ) {
      e.preventDefault();
      document
        .querySelectorAll(".row-checkbox")
        .forEach((cb) => (cb.checked = true));
      updateActionBar();
    }

    // Delete key - Delete selected issues (if any selected)
    if (e.key === "Delete" && !e.target.hasAttribute("contenteditable")) {
      const selected = document.querySelectorAll(".row-checkbox:checked");
      if (selected.length > 0) {
        e.preventDefault();
        document.getElementById("bulkDeleteBtn")?.click();
      }
    }

    // Escape - Deselect all
    if (e.key === "Escape") {
      document
        .querySelectorAll(".row-checkbox")
        .forEach((cb) => (cb.checked = false));
      updateActionBar();
    }
  });
});

function applyColorMappingsAndProceed() {
  const projectKey = document.getElementById("mappingProject").value;

  if (!projectKey) {
    showToast("Please select a project first", "warning");
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
    showToast(
      `Please assign issue types to all colors in the mapping section above. ${unmappedRegions.length} issue(s) still need mapping.`,
      "warning"
    );
    return;
  }

  // Set project key for all regions
  appState.ocrRegions.forEach((region) => {
    region.project_key = projectKey;
  });

  // Collect color mappings for persistence
  const colorMappings = {};
  document.querySelectorAll(".color-mapping-select").forEach((select) => {
    const color = select.dataset.color;
    const issueType = select.value;
    if (color && issueType) {
      colorMappings[color] = issueType;
    }
  });

  // Save color mappings to database
  if (Object.keys(colorMappings).length > 0) {
    fetch("/api/mapping/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_key: projectKey,
        mappings: colorMappings,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          console.log(`Saved ${data.saved} color mappings to database`);
        }
      })
      .catch((error) => console.error("Failed to save color mappings:", error));
  }

  // Save issue mappings to database
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

        // Update badges
        updateTabBadge("mapping", "complete");
        updateTabBadge("issues", "count", appState.ocrRegions.length);
      } else {
        showToast("Failed to save mappings", "danger");
      }
    })
    .catch((error) => {
      console.error("Failed to save mappings:", error);
      showToast("Failed to save mappings", "danger");
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
      <td data-label="Select"><input type="checkbox" class="row-checkbox" data-index="${index}"></td>
      <td data-label="ID">${index + 1}</td>
      <td data-label="Image">
        ${
          region.image_filename
            ? `<img src="/uploads/${region.image_filename}" 
                 alt="Thumbnail" 
                 class="img-thumbnail preview-thumbnail" 
                 style="width: 60px; height: 60px; object-fit: cover; cursor: pointer;"
                 data-filename="${region.image_filename}"
                 onclick="showImagePreview('${region.image_filename}')">`
            : '<span class="text-muted">-</span>'
        }
      </td>
      <td data-label="Color"><span class="color-badge" style="background-color: ${
        region.color_hex
      };"></span></td>
      <td data-label="Issue Key">${issueKeyCell}</td>
      <td data-label="Project" class="editable-autocomplete" data-field="project_key" data-id="${index}" data-type="project">${
      region.project_key || '<span class="text-muted">Click to set</span>'
    }</td>
      <td data-label="Type" class="editable-autocomplete" data-field="issue_type" data-id="${index}" data-type="issuetype">${
      region.issue_type || '<span class="text-muted">Click to set</span>'
    }</td>
      <td data-label="Summary" contenteditable="true" data-field="summary" data-id="${index}">${
      region.text || ""
    }</td>
      <td data-label="Description" contenteditable="true" data-field="description" data-id="${index}">${
      region.linked_text || ""
    }</td>
      <td data-label="Confidence">
        <span class="badge bg-${getConfidenceBadgeClass(
          region.confidence || 0
        )}">
          ${Math.round(region.confidence || 0)}%
        </span>
      </td>
      <td data-label="Actions">
        <button class="btn btn-sm btn-danger" onclick="deleteIssue(${index})">Delete</button>
      </td>
    `;
  });

  // Add inline editing handlers
  document
    .querySelectorAll("#issuesTable [contenteditable]")
    .forEach((cell) => {
      // Show visual indicator on focus
      cell.addEventListener("focus", (e) => {
        e.target.style.outline = "2px solid #0d6efd";
      });

      // Enter key to save and move to next row
      cell.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          e.target.blur(); // Trigger save

          // Move to same cell in next row
          const currentRow = e.target.closest("tr");
          const nextRow = currentRow.nextElementSibling;
          if (nextRow) {
            const cellIndex = Array.from(currentRow.cells).indexOf(
              e.target.closest("td")
            );
            const nextCell =
              nextRow.cells[cellIndex]?.querySelector("[contenteditable]");
            if (nextCell) {
              nextCell.focus();
            }
          }
        }
      });

      cell.addEventListener("blur", (e) => {
        e.target.style.outline = "none";

        const id = parseInt(e.target.dataset.id);
        const field = e.target.dataset.field;
        const value = e.target.textContent;

        if (appState.ocrRegions[id]) {
          if (field === "summary") {
            appState.ocrRegions[id].text = value;
          } else if (field === "description") {
            appState.ocrRegions[id].linked_text = value;
          } else if (field === "project_key") {
            appState.ocrRegions[id].project_key = value;
          }

          // Update database if db_id exists
          if (appState.ocrRegions[id].db_id) {
            // Show saving indicator
            const originalBg = e.target.style.backgroundColor;
            e.target.style.backgroundColor = "#fff3cd";
            e.target.textContent = e.target.textContent + " ⏳";

            const updateData = {};
            updateData[field] = value;

            fetch(`/api/issues/${appState.ocrRegions[id].db_id}`, {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(updateData),
            })
              .then((response) => response.json())
              .then((data) => {
                if (data.success) {
                  // Show success feedback
                  e.target.textContent = value + " ✓";
                  e.target.style.backgroundColor = "#d1e7dd";
                  setTimeout(() => {
                    e.target.textContent = value;
                    e.target.style.backgroundColor = originalBg;
                  }, 1500);
                } else {
                  // Show error feedback
                  e.target.style.backgroundColor = "#f8d7da";
                  showToast("Failed to save changes", "danger", 3000);
                  setTimeout(() => {
                    e.target.style.backgroundColor = originalBg;
                  }, 2000);
                }
              })
              .catch((error) => {
                console.error("Failed to update issue:", error);
                e.target.textContent = value;
                e.target.style.backgroundColor = "#f8d7da";
                showToast("Network error - changes not saved", "danger", 3000);
                setTimeout(() => {
                  e.target.style.backgroundColor = originalBg;
                }, 2000);
              });
          }
        }
      });
    });

  // Update issue count badge
  document.getElementById("issueCount").textContent =
    appState.ocrRegions.length;

  // Attach checkbox event listeners
  attachBulkSelectionHandlers();

  // Update action bar state (Import/Delete All buttons)
  updateActionBar();

  // Attach autocomplete handlers for Project and Type cells
  attachAutocompleteHandlers();
}

function attachAutocompleteHandlers() {
  document.querySelectorAll(".editable-autocomplete").forEach((cell) => {
    cell.addEventListener("click", function () {
      if (this.querySelector("input")) return; // Already editing

      const dataType = this.dataset.type;
      const currentValue = this.textContent.trim();
      const cleanValue = currentValue.includes("Click to set")
        ? ""
        : currentValue;

      // Create input with datalist for autocomplete
      const inputId = `autocomplete-${Date.now()}`;
      const listId = `datalist-${Date.now()}`;

      const input = document.createElement("input");
      input.type = "text";
      input.className = "form-control form-control-sm";
      input.value = cleanValue;
      input.setAttribute("list", listId);
      input.style.width = "100%";

      const datalist = document.createElement("datalist");
      datalist.id = listId;

      // Populate datalist based on type
      if (dataType === "project") {
        appState.projects?.forEach((project) => {
          const option = document.createElement("option");
          option.value = project.key;
          datalist.appendChild(option);
        });
      } else if (dataType === "issuetype") {
        // Common Jira issue types
        ["Task", "Story", "Bug", "Epic", "Sub-task"].forEach((type) => {
          const option = document.createElement("option");
          option.value = type;
          datalist.appendChild(option);
        });
      }

      this.innerHTML = "";
      this.appendChild(input);
      this.appendChild(datalist);
      input.focus();
      input.select();

      const saveValue = () => {
        const newValue = input.value.trim();
        const index = parseInt(this.dataset.id);
        const field = this.dataset.field;

        if (newValue === "") {
          this.innerHTML = '<span class="text-muted">Click to set</span>';
          return;
        }

        // Validate project exists
        if (dataType === "project" && appState.projects) {
          const validProject = appState.projects.find(
            (p) => p.key === newValue
          );
          if (!validProject) {
            showToast(
              `Invalid project: ${newValue}. Please select from the list.`,
              "warning",
              4000
            );
            this.innerHTML =
              cleanValue || '<span class="text-muted">Click to set</span>';
            return;
          }
        }

        this.textContent = newValue;
        appState.ocrRegions[index][field] = newValue;

        // Save to backend
        if (appState.ocrRegions[index].db_id) {
          const updateData = { [field]: newValue };
          fetch(`/api/issues/${appState.ocrRegions[index].db_id}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(updateData),
          })
            .then((response) => response.json())
            .then((data) => {
              if (data.success) {
                this.style.backgroundColor = "#d1e7dd";
                setTimeout(() => {
                  this.style.backgroundColor = "";
                }, 1500);
              } else {
                showToast("Failed to save changes", "danger");
                this.style.backgroundColor = "#f8d7da";
                setTimeout(() => {
                  this.style.backgroundColor = "";
                }, 2000);
              }
            })
            .catch((error) => {
              console.error("Failed to update:", error);
              showToast("Network error - changes not saved", "danger");
            });
        }
      };

      input.addEventListener("blur", saveValue);
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          saveValue();
        } else if (e.key === "Escape") {
          this.textContent =
            cleanValue || '<span class="text-muted">Click to set</span>';
        }
      });
    });
  });
}

function attachBulkSelectionHandlers() {
  // Select all checkbox
  const selectAllCheckbox = document.getElementById("selectAllCheckbox");
  if (selectAllCheckbox) {
    selectAllCheckbox.addEventListener("change", (e) => {
      const checkboxes = document.querySelectorAll(".row-checkbox");
      checkboxes.forEach((cb) => (cb.checked = e.target.checked));
      updateActionBar();
    });
  }

  // Individual row checkboxes
  document.querySelectorAll(".row-checkbox").forEach((cb) => {
    cb.addEventListener("change", updateActionBar);
  });
}

function updateActionBar() {
  const selectedCheckboxes = document.querySelectorAll(".row-checkbox:checked");
  const count = selectedCheckboxes.length;
  const selectedCountSpan = document.getElementById("selectedCount");
  const deleteAllBtn = document.getElementById("deleteAllBtn");
  const importBtn = document.getElementById("startImportBtn");

  // Bulk action controls
  const selectAllBtn = document.getElementById("selectAllBtn");
  const deselectAllBtn = document.getElementById("deselectAllBtn");
  const bulkDeleteBtn = document.getElementById("bulkDeleteBtn");

  // Update selected count (always visible, shows 0 when nothing selected)
  selectedCountSpan.textContent = count;

  // Enable/disable bulk action controls based on selection
  const hasSelection = count > 0;
  const totalIssues = appState.ocrRegions ? appState.ocrRegions.length : 0;

  // Select All: enabled when there are issues and not all selected
  selectAllBtn.disabled = totalIssues === 0 || count === totalIssues;
  if (selectAllBtn.disabled) {
    selectAllBtn.setAttribute(
      "title",
      totalIssues === 0 ? "No issues" : "All issues already selected"
    );
  } else {
    selectAllBtn.removeAttribute("title");
  }

  // Deselect All: enabled when there are selections
  deselectAllBtn.disabled = !hasSelection;
  if (deselectAllBtn.disabled) {
    deselectAllBtn.setAttribute("title", "No items selected");
  } else {
    deselectAllBtn.removeAttribute("title");
  }

  // Bulk delete: enabled when there are selections
  bulkDeleteBtn.disabled = !hasSelection;

  if (!hasSelection) {
    bulkDeleteBtn.setAttribute("title", "Select items to delete");
  } else {
    bulkDeleteBtn.removeAttribute("title");
  }

  // Enable/disable primary action buttons
  const hasJiraConfig =
    appState.jiraSettings && appState.jiraSettings.server_url;

  // Delete All: enabled when there are issues
  if (totalIssues > 0) {
    deleteAllBtn.disabled = false;
    deleteAllBtn.removeAttribute("title");
  } else {
    deleteAllBtn.disabled = true;
    deleteAllBtn.setAttribute("title", "No issues to delete");
  }

  // Import: enabled when there are issues AND Jira is configured
  if (totalIssues > 0 && hasJiraConfig) {
    importBtn.disabled = false;
    importBtn.removeAttribute("title");
  } else if (totalIssues === 0) {
    importBtn.disabled = true;
    importBtn.setAttribute("title", "No issues to import");
  } else if (!hasJiraConfig) {
    importBtn.disabled = true;
    importBtn.setAttribute("title", "Configure Jira settings first");
  }

  // Update select all checkbox state
  const allCheckboxes = document.querySelectorAll(".row-checkbox");
  const selectAllCheckbox = document.getElementById("selectAllCheckbox");
  if (selectAllCheckbox) {
    selectAllCheckbox.checked =
      allCheckboxes.length > 0 && count === allCheckboxes.length;
    selectAllCheckbox.indeterminate = count > 0 && count < allCheckboxes.length;
  }
}

function filterByConfidence(filterType) {
  const rows = document.querySelectorAll("#issuesTable tbody tr");

  rows.forEach((row) => {
    const confidenceText = row.cells[8].textContent; // Confidence column
    const confidence = parseInt(confidenceText);

    let show = true;
    if (filterType === "high") {
      show = confidence >= 80;
    } else if (filterType === "medium") {
      show = confidence >= 60 && confidence < 80;
    } else if (filterType === "low") {
      show = confidence < 60;
    }

    row.style.display = show ? "" : "none";
  });

  // Update count badge
  const visibleCount = Array.from(rows).filter(
    (r) => r.style.display !== "none"
  ).length;
  document.getElementById("issueCount").textContent = visibleCount;
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
  showConfirm(
    "Delete this issue?",
    "Delete Issue",
    "Delete",
    "danger",
    "🗑️"
  ).then((confirmed) => {
    if (confirmed) {
      const issue = appState.ocrRegions[index];
      const dbId = issue.db_id;

      // Delete from frontend state
      appState.ocrRegions.splice(index, 1);
      populateReviewTable();

      // Delete from database if it has a db_id
      if (dbId) {
        fetch("/api/issues/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids: [dbId] }),
        })
          .then((response) => response.json())
          .then((data) => {
            if (!data.success) {
              console.error("Failed to delete issue from database");
            }
          })
          .catch((error) => {
            console.error("Delete error:", error);
          });
      }
    }
  });
}

function showImagePreview(filename) {
  const modal = new bootstrap.Modal(
    document.getElementById("imagePreviewModal")
  );
  const img = document.getElementById("modalPreviewImage");
  img.src = `/uploads/${filename}`;
  document.getElementById(
    "imagePreviewModalLabel"
  ).textContent = `Image: ${filename}`;
  modal.show();
}

function updateImportSummary() {
  const issueCount = appState.ocrRegions.length;
  document.getElementById("issueCount").textContent = issueCount;
}

function startJiraImport() {
  // Check if Jira is configured
  if (!appState.jiraSettings || !appState.jiraSettings.server_url) {
    showToast(
      "⚙️ Please configure Jira settings first using the gear icon in the header",
      "warning"
    );
    return;
  }

  if (appState.ocrRegions.length === 0) {
    showToast("No issues to import", "warning");
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
        showToast(message, alertType);
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
          showToast(
            `${data.failed} issue(s) failed. Check console for details.`,
            "danger",
            10000
          );
        }
      } else {
        showToast(`Import failed: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      showToast(`Import error: ${error.message}`, "danger");
    })
    .finally(() => {
      importBtn.disabled = false;
      importBtn.innerHTML = '<i class="bi bi-cloud-upload"></i> Import';
      hideProgress();
    });
}

function loadFieldDefaults() {
  const projectKey = document.getElementById("fieldDefaultsProject").value;
  const issueType = document.getElementById("fieldDefaultsIssueType").value;

  if (!projectKey || !issueType) {
    showToast("Please select both project and issue type", "warning");
    return;
  }

  showToast("Loading fields...", "info", 2000);

  fetch(`/api/jira/fields?project_key=${projectKey}&issue_type=${issueType}`)
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        renderFieldDefaultsUI(data.fields);

        // Load existing defaults for this project/issue type if they exist
        return fetch(`/api/field-defaults/${projectKey}/${issueType}`);
      } else {
        showToast(`Failed to load fields: ${data.error}`, "danger");
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
        showToast(
          `Loaded existing configuration for ${issueType}`,
          "info",
          3000
        );
      }
      // Enable save and cancel buttons
      setButtonState("#saveFieldDefaultsBtn", true);
      setButtonState("#cancelEditBtn", true);
    })
    .catch((error) => {
      showToast(`Error loading fields: ${error.message}`, "danger");
    });
}

function saveCurrentFieldDefaults() {
  const projectKey = document.getElementById("fieldDefaultsProject").value;
  const issueType = document.getElementById("fieldDefaultsIssueType").value;

  if (!projectKey || !issueType) {
    showToast("Please select both project and issue type", "warning");
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
        showToast(
          `Field defaults saved for ${projectKey}/${issueType}`,
          "success",
          3000
        );
        loadConfiguredDefaultsSummary(); // Refresh the summary panel
        resetFieldEditor(); // Reset editor to clean state
      } else {
        showToast(`Failed to save: ${data.error}`, "danger");
      }
    })
    .catch((error) => {
      showToast(`Save error: ${error.message}`, "danger");
    });
}

function loadConfiguredDefaultsSummary() {
  fetch("/api/field-defaults")
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.configs && data.configs.length > 0) {
        renderConfiguredDefaultsSummary(data.configs);
      } else {
        document.getElementById("configuredDefaultsList").innerHTML =
          '<p class="text-muted small mb-0">No field defaults configured yet.</p>';
      }
    })
    .catch((error) => {
      console.error("Failed to load configured defaults:", error);
    });
}

function renderConfiguredDefaultsSummary(configs) {
  const container = document.getElementById("configuredDefaultsList");

  let html = '<div class="list-group list-group-flush">';

  configs.forEach((config) => {
    const fieldCount = config.field_count || 0;
    const fieldsLabel = fieldCount === 1 ? "field" : "fields";

    html += `
      <div class="list-group-item px-2 py-2 d-flex justify-content-between align-items-start">
        <div class="flex-grow-1" style="cursor: pointer;" 
             onclick="loadExistingDefault('${config.project_key}', '${config.issue_type}')">
          <div class="fw-bold small">${config.project_key} / ${config.issue_type}</div>
          <div class="text-muted" style="font-size: 0.75rem;">
            ${fieldCount} ${fieldsLabel} configured
          </div>
        </div>
        <button class="btn btn-sm btn-outline-danger" 
                onclick="deleteFieldDefault('${config.project_key}', '${config.issue_type}', event)"
                title="Delete this configuration">
          <i class="bi bi-trash"></i>
        </button>
      </div>
    `;
  });

  html += "</div>";
  container.innerHTML = html;
}

function loadExistingDefault(projectKey, issueType) {
  // Set the values
  document.getElementById("fieldDefaultsProject").value = projectKey;
  document.getElementById("fieldDefaultsIssueType").value = issueType;

  // Update editor title
  document.getElementById(
    "editorTitle"
  ).textContent = `Editing: ${projectKey} / ${issueType}`;

  // Hide selection controls
  document.getElementById("selectionControls").style.display = "none";

  // Load the fields
  loadFieldDefaults();
}

function showFieldEditor(mode) {
  if (mode === "new") {
    // Show selection controls
    document.getElementById("selectionControls").style.display = "block";
    document.getElementById("editorTitle").textContent = "Add New Issue Type";

    // Clear inputs
    document.getElementById("fieldDefaultsProject").value = "";
    document.getElementById("fieldDefaultsIssueType").value = "";
    document.getElementById("fieldDefaultsContainer").innerHTML =
      '<p class="text-muted small">Select project and issue type, then click "Load Fields".</p>';
    setButtonState("#saveFieldDefaultsBtn", false);
    setButtonState("#cancelEditBtn", false);
  }
}

function resetFieldEditor() {
  // Show selection controls
  document.getElementById("selectionControls").style.display = "block";
  document.getElementById("editorTitle").textContent = "Field Editor";

  // Clear inputs
  document.getElementById("fieldDefaultsProject").value = "";
  document.getElementById("fieldDefaultsIssueType").value = "";
  document.getElementById("fieldDefaultsContainer").innerHTML =
    '<p class="text-muted small">Select a configured type from the left, or create a new one.</p>';
  setButtonState("#saveFieldDefaultsBtn", false);
  setButtonState("#cancelEditBtn", false);
}

function deleteFieldDefault(projectKey, issueType, event) {
  event.stopPropagation(); // Prevent triggering loadExistingDefault

  showConfirm(
    `Delete field defaults for <strong>${projectKey} / ${issueType}</strong>?<br><br>This cannot be undone.`,
    "Delete Field Defaults",
    "Delete",
    "danger"
  ).then((confirmed) => {
    if (confirmed) {
      fetch(`/api/field-defaults/${projectKey}/${issueType}`, {
        method: "DELETE",
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            showToast(
              `Deleted defaults for ${projectKey}/${issueType}`,
              "success"
            );
            loadConfiguredDefaultsSummary();
            resetFieldEditor();
          } else {
            showToast(`Failed to delete: ${data.error}`, "danger");
          }
        })
        .catch((error) => {
          showToast(`Delete error: ${error.message}`, "danger");
        });
    }
  });
}

function loadConfiguredIssueTypes() {
  // Deprecated - replaced by loadConfiguredDefaultsSummary
  // Keep for backward compatibility
  loadConfiguredDefaultsSummary();
}

function renderFieldDefaultsUI(fields) {
  const container = document.getElementById("fieldDefaultsContainer");

  if (fields.length === 0) {
    container.innerHTML =
      '<p class="text-muted small">No additional fields found for this issue type.</p>';
    return;
  }

  // Sort fields: required fields first, then optional
  const sortedFields = [...fields].sort((a, b) => {
    if (a.required && !b.required) return -1;
    if (!a.required && b.required) return 1;
    return a.name.localeCompare(b.name);
  });

  let html = "";

  // Get existing field defaults
  const existingDefaults = appState.jiraSettings?.field_defaults || {};

  sortedFields.forEach((field) => {
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

  container.innerHTML = html;
}

// ============================================================================
// Tab Progress Indicators
// ============================================================================

function updateTabBadge(tabId, type, count = null) {
  const badge = document.getElementById(`${tabId}-badge`);
  if (!badge) return;

  if (type === "complete") {
    badge.innerHTML = '<span class="badge bg-success ms-1">✓</span>';
  } else if (type === "count" && count !== null) {
    badge.innerHTML = `<span class="badge bg-primary ms-1">${count}</span>`;
  } else if (type === "warning") {
    badge.innerHTML = '<span class="badge bg-warning ms-1">!</span>';
  } else if (type === "clear") {
    badge.innerHTML = "";
  }
}
