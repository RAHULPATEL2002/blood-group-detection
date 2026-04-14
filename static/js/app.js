function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return "—";
    const sizes = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let index = 0;
    while (value >= 1024 && index < sizes.length - 1) {
        value /= 1024;
        index += 1;
    }
    return `${value.toFixed(1)} ${sizes[index]}`;
}

function bindTemperatureSync() {
    const range = document.getElementById("temperatureRange");
    const input = document.getElementById("temperatureInput");
    if (!range || !input) return;

    const sync = (value) => {
        range.value = value;
        input.value = value;
    };

    range.addEventListener("input", (event) => {
        sync(event.target.value);
    });

    input.addEventListener("input", (event) => {
        sync(event.target.value);
    });
}

function bindImagePreview() {
    const input = document.getElementById("imageInput");
    const preview = document.getElementById("previewImage");
    const placeholder = document.getElementById("previewPlaceholder");
    const fileName = document.getElementById("fileName");
    const fileSize = document.getElementById("fileSize");
    const fileStatus = document.getElementById("fileStatus");

    if (!input || !preview || !placeholder) return;

    input.addEventListener("change", () => {
        const file = input.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            preview.src = event.target?.result || "";
            preview.classList.add("show");
            placeholder.style.display = "none";
        };
        reader.readAsDataURL(file);

        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = formatBytes(file.size);
        if (fileStatus) fileStatus.textContent = "Ready for analysis";
    });
}

function bindFormOverlay() {
    const form = document.getElementById("scanForm");
    const overlay = document.getElementById("loadingOverlay");
    if (!form || !overlay) return;

    form.addEventListener("submit", (event) => {
        const input = document.getElementById("imageInput");
        if (!input?.files?.length) {
            event.preventDefault();
            return;
        }
        overlay.classList.add("show");
    });
}

document.addEventListener("DOMContentLoaded", () => {
    bindTemperatureSync();
    bindImagePreview();
    bindFormOverlay();
});
